"""Incremental bioformats2raw building: `Bf2RawBuilder.prepare_series()` and co.

The store must be a valid collection after every call (or not exist yet).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

import yaozarrs
from yaozarrs import DimSpec, v05, v06
from yaozarrs.write import v05 as w05
from yaozarrs.write import v06 as w06
from yaozarrs.write._common import _arrays, _builders, _groups

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

np = pytest.importorskip("numpy")

WRITERS: list[str] = []
if importlib.util.find_spec("zarr") is not None:
    if int(importlib.metadata.version("zarr").split(".")[0]) >= 3:
        WRITERS.append("zarr")
if importlib.util.find_spec("tensorstore") is not None:
    WRITERS.append("tensorstore")
if not WRITERS:
    pytest.skip("No supported Zarr writer found", allow_module_level=True)
needs_zarr = pytest.mark.skipif("zarr" not in WRITERS, reason="needs zarr-python")

YX = [DimSpec(name="y", scale=0.5, unit="micrometer"), DimSpec(name="x", scale=0.5)]
CYX = [DimSpec(name="c"), *YX]
SPEC_2D = ((16, 16), "uint8")
SPEC_3D = ((2, 32, 32), "uint16")


@pytest.fixture(params=["v05", "v06"])
def ver(request: pytest.FixtureRequest) -> tuple[ModuleType, ModuleType]:
    """The models and the write module of an OME-Zarr version."""
    return {"v05": (v05, w05), "v06": (v06, w06)}[request.param]


@pytest.fixture(params=WRITERS)
def writer(request: pytest.FixtureRequest) -> str:
    return request.param


def _image(models: ModuleType, dims: list[DimSpec] = YX, n_levels: int = 1) -> Any:
    ms = models.Multiscale.from_dims(dims, n_levels=n_levels)
    return models.Image(multiscales=[ms])


def _listing(dest: Path) -> list[str]:
    zarr_json = json.loads((dest / "OME" / "zarr.json").read_bytes())
    return zarr_json["attributes"]["ome"]["series"]


def _set(arr: Any, index: Any, value: Any) -> None:
    if hasattr(arr, "write"):  # tensorstore
        arr[index].write(value).result()
    else:
        arr[index] = value


def _get(arr: Any, index: Any) -> Any:
    return arr[index].read().result() if hasattr(arr, "read") else arr[index]


def _snapshot(path: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def _bf2raw_store(w: ModuleType, models: ModuleType, dest: Path, *names: str) -> None:
    images = {n: (_image(models), np.zeros((16, 16), "uint8")) for n in names}
    w.write_bioformats2raw(dest, images, writer=WRITERS[0])


def test_valid_after_every_call(tmp_path: Path, ver: Any, writer: str) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    builder = w.Bf2RawBuilder(
        dest, writer=writer, ome_xml="<OME/>", extra_attributes={"a": 1}
    )
    assert not dest.exists()  # (constructing writes nothing)
    assert builder.series == ()

    path, arrays = builder.prepare_series("0", _image(models, CYX), SPEC_3D)
    assert path == dest / "0"
    assert set(arrays) == {"0"}
    yaozarrs.validate_zarr_store(dest)
    _set(arrays["0"], 0, 1)

    builder.prepare_series("group_1/array_1_0", _image(models), SPEC_2D)  # (nested)
    yaozarrs.validate_zarr_store(dest)
    assert (dest / "group_1" / "zarr.json").exists()

    builder.write_image("1", _image(models), np.ones((16, 16), "uint8"))
    yaozarrs.validate_zarr_store(dest)

    builder.add_series("2", _image(models), SPEC_2D)
    builder.add_series("3", _image(models, n_levels=2), [SPEC_2D, ((8, 8), "uint8")])
    assert "2" not in builder.series  # (only queued)
    _, prepared = builder.prepare()
    assert set(prepared) == {"2/0", "3/0", "3/1"}
    yaozarrs.validate_zarr_store(dest)

    builder.add_series("4", _image(models), SPEC_2D)
    _, prepared = builder.prepare()  # (prepare() is repeatable)
    assert set(prepared) == {"4/0"}
    yaozarrs.validate_zarr_store(dest)
    assert (_get(arrays["0"], 0) == 1).all()  # (earlier data is kept)

    expected = ["0", "group_1/array_1_0", "1", "2", "3", "4"]
    assert _listing(dest) == expected
    assert builder.series == tuple(expected)
    assert (dest / "OME" / "METADATA.ome.xml").read_text() == "<OME/>"
    assert json.loads((dest / "zarr.json").read_bytes())["attributes"]["a"] == 1


@needs_zarr
@pytest.mark.parametrize("existing", ["none", "overwrite", "extend"])
def test_valid_at_every_write(
    tmp_path: Path, ver: Any, existing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before each write, the store is absent or valid: so a crash leaves it so."""
    models, w = ver
    dest = tmp_path / "out.zarr"
    if existing != "none":
        _bf2raw_store(w, models, dest, "0", "1")
    checked: list[str] = []

    def checking(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # (until the root zarr.json is written, there's no store at all)
            if (dest / "zarr.json").exists():
                yaozarrs.validate_zarr_store(dest)
            checked.append(func.__name__)
            return func(*args, **kwargs)

        return wrapper

    for module, name in [
        (_groups, "_write_group_json"),
        (_builders, "_write_group_json"),
        (_builders, "_write_to_array"),
        (_arrays, "_create_array_zarr"),
    ]:
        monkeypatch.setattr(module, name, checking(getattr(module, name)))

    overwrite = existing == "overwrite"
    builder = w.Bf2RawBuilder(
        dest, writer="zarr", ome_xml="<OME/>", overwrite=overwrite
    )
    # (not "0": a numbered series is valid even without OME/zarr.json)
    builder.prepare_series(
        "s", _image(models, n_levels=2), [SPEC_2D, ((8, 8), "uint8")]
    )
    builder.write_image("a/b", _image(models), np.ones((16, 16), "uint8"))
    builder.add_series("c", _image(models), SPEC_2D)
    builder.prepare()
    yaozarrs.validate_zarr_store(dest)
    before = ["0", "1"] if existing == "extend" else []
    assert _listing(dest) == [*before, "s", "a/b", "c"]
    assert len(checked) > 10


def test_written_series_has_data_when_listed(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    seen: list[list[str] | None] = []
    original = _arrays._write_to_array

    def _write_to_array(*args: Any, **kwargs: Any) -> None:
        seen.append(_listing(dest) if (dest / "OME" / "zarr.json").exists() else None)
        original(*args, **kwargs)

    builder = w.Bf2RawBuilder(dest, writer=WRITERS[0])
    builder.write_image("0", _image(models), np.ones((16, 16), "uint8"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_builders, "_write_to_array", _write_to_array)
        builder.write_image("1", _image(models), np.ones((16, 16), "uint8"))
    assert seen == [["0"]]  # ("1" wasn't listed yet while its data was written)


def test_failure_cleans_up(tmp_path: Path, ver: Any, writer: str) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    builder = w.Bf2RawBuilder(dest, writer=writer)
    builder.prepare_series("0", _image(models), SPEC_2D)
    before = _snapshot(dest)

    with pytest.MonkeyPatch.context() as mp:
        # (e.g. Ctrl-C while writing the data)
        mp.setattr(_builders, "_write_to_array", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            builder.write_image("1", _image(models), np.ones((16, 16), "uint8"))
    assert _snapshot(dest) == before
    yaozarrs.validate_zarr_store(dest)

    builder.write_image("1", _image(models), np.ones((16, 16), "uint8"))  # (retry)
    assert _listing(dest) == ["0", "1"]


def test_failed_prepare_adds_nothing(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    fail = {"path": "b"}

    def create(path: Path, **kwargs: Any) -> Any:
        if path.parent.name == fail["path"]:
            raise RuntimeError("disk full")
        return _arrays._get_create_func(WRITERS[0])(path, **kwargs)

    builder = w.Bf2RawBuilder(dest, writer=create)
    builder.add_series("a", _image(models), SPEC_2D)
    builder.add_series("b", _image(models), SPEC_2D)
    with pytest.raises(RuntimeError, match="disk full"):
        builder.prepare()
    assert not (dest / "a").exists() and not (dest / "b").exists()
    assert builder.series == ()

    fail["path"] = ""
    builder.add_series("a", _image(models), SPEC_2D)  # (to retry, add them again)
    builder.add_series("b", _image(models), SPEC_2D)
    _, arrays = builder.prepare()
    assert set(arrays) == {"a/0", "b/0"}
    assert _listing(dest) == ["a", "b"]


def test_failed_first_series_leaves_nothing(tmp_path: Path, ver: Any) -> None:
    # (so the same call can simply be run again)
    models, w = ver
    dest = tmp_path / "out.zarr"
    _bf2raw_store(w, models, dest, "0")
    images = {"s": (_image(models), np.ones((16, 16), "uint8"))}
    with pytest.MonkeyPatch.context() as mp:
        # (e.g. Ctrl-C while writing the data)
        mp.setattr(_builders, "_write_to_array", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            w.write_bioformats2raw(dest, images, writer=WRITERS[0], overwrite=True)
    assert not dest.exists()

    w.write_bioformats2raw(dest, images, writer=WRITERS[0], overwrite=True)
    yaozarrs.validate_zarr_store(dest)
    assert _listing(dest) == ["s"]


def test_failed_root_write(tmp_path: Path, ver: Any) -> None:
    # (if the root zarr.json can't be written, the first series isn't added)
    models, w = ver
    dest = tmp_path / "out.zarr"
    builder = w.Bf2RawBuilder(dest, writer=WRITERS[0])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_builders, "_write_group_json", Mock(side_effect=OSError("full")))
        with pytest.raises(OSError, match="full"):
            builder.prepare_series("s", _image(models), SPEC_2D)
    assert builder.series == ()
    assert not (dest / "s").exists()

    builder.prepare_series("s", _image(models), SPEC_2D)  # (retry)
    yaozarrs.validate_zarr_store(dest)
    assert _listing(dest) == ["s"]


def test_name_errors(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    builder = w.Bf2RawBuilder(dest, writer=WRITERS[0])
    builder.prepare_series("a", _image(models), SPEC_2D)
    builder.write_image("c", _image(models), np.ones((16, 16), "uint8"))
    builder.add_series("d", _image(models), SPEC_2D)
    for name, match in [
        ("a", "already exists"),
        ("c", "already exists"),
        ("d", "already added via add_series"),
    ]:
        with pytest.raises(ValueError, match=match):
            builder.prepare_series(name, _image(models), SPEC_2D)
    (dest / "x").mkdir()  # (made by someone else: it's not replaced or cleaned up)
    (dest / "x" / "mine.txt").write_text("mine")
    with pytest.raises(FileExistsError, match="already exists"):
        builder.prepare_series("x", _image(models), SPEC_2D)
    assert (dest / "x" / "mine.txt").exists()
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize(
    "name", ["OME", "ome/x", "../outside", "a//b", "__x", ".", "x/zarr.json"]
)
def test_invalid_names(tmp_path: Path, ver: Any, name: str) -> None:
    models, w = ver
    builder = w.Bf2RawBuilder(tmp_path / "out.zarr", writer=WRITERS[0])
    with pytest.raises(ValueError, match="Invalid"):
        builder.prepare_series(name, _image(models), SPEC_2D)
    with pytest.raises(ValueError, match="Invalid"):
        builder.add_series(name, _image(models), SPEC_2D)
    assert not list(tmp_path.iterdir())  # (nothing written, inside or out)


def test_extend_existing_collection(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    _bf2raw_store(w, models, dest, "0", "1")
    before = _snapshot(dest)
    builder = w.Bf2RawBuilder(
        dest, writer=WRITERS[0], ome_xml="<OME/>", extra_attributes={"a": 1}
    )
    assert builder.series == ("0", "1")
    assert _snapshot(dest) == before  # (constructing writes nothing)
    with pytest.raises(ValueError, match="already exists"):
        builder.prepare_series("0", _image(models), SPEC_2D)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_builders, "_write_to_array", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            builder.write_image("x", _image(models), np.ones((16, 16), "uint8"))
    assert _snapshot(dest) == before  # (a failure never removes the collection)

    with pytest.MonkeyPatch.context() as mp:  # (merging `extra_attributes` fails)
        mp.setattr(_builders, "_write_group_json", Mock(side_effect=OSError("full")))
        with pytest.raises(OSError, match="full"):
            builder.prepare_series("x", _image(models), SPEC_2D)
    yaozarrs.validate_zarr_store(dest)
    assert builder.series == ("0", "1")
    assert _listing(dest) == ["0", "1"]

    builder.prepare_series("2", _image(models), SPEC_2D)
    yaozarrs.validate_zarr_store(dest)
    assert _listing(dest) == ["0", "1", "2"]
    assert (dest / "OME" / "METADATA.ome.xml").read_text() == "<OME/>"
    root = json.loads((dest / "zarr.json").read_bytes())["attributes"]
    assert root["a"] == 1 and root["ome"]["bioformats2raw.layout"] == 3

    # a one-shot write doesn't add to it
    with pytest.raises(FileExistsError, match="already exists"):
        _bf2raw_store(w, models, dest, "3")


def _numbered(dest: Path, w: ModuleType, models: ModuleType) -> None:
    # (a collection without a `series` list: its series are its numbered groups)
    _bf2raw_store(w, models, dest, "0")
    (dest / "OME" / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group"})
    )


def _other_version(dest: Path, w: ModuleType, models: ModuleType) -> None:
    other_models, other_w = (v06, w06) if w is w05 else (v05, w05)
    _bf2raw_store(other_w, other_models, dest, "0")


def _plate(dest: Path, w: ModuleType, models: ModuleType) -> None:
    # (a plate, though it has `bioformats2raw.layout`: the plate wins)
    _bf2raw_store(w, models, dest, "0")
    root = json.loads((dest / "zarr.json").read_bytes())
    root["attributes"]["ome"]["plate"] = {}
    (dest / "zarr.json").write_text(json.dumps(root))


def _other_dir(dest: Path, w: ModuleType, models: ModuleType) -> None:
    dest.mkdir()
    (dest / "mine.txt").write_text("mine")


@pytest.mark.parametrize("make", [_numbered, _other_version, _plate, _other_dir])
def test_other_content_is_refused(tmp_path: Path, ver: Any, make: Callable) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    make(dest, w, models)
    before = _snapshot(dest)
    with pytest.raises(FileExistsError, match="Use overwrite=True"):
        w.Bf2RawBuilder(dest, writer=WRITERS[0])
    assert _snapshot(dest) == before


def test_overwrite(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    _bf2raw_store(w, models, dest, "0", "1")
    before = _snapshot(dest)

    # it's replaced once, when the first series is created
    builder = w.Bf2RawBuilder(dest, writer=WRITERS[0], overwrite=True)
    assert _snapshot(dest) == before
    _, arrays = builder.prepare_series("x", _image(models), SPEC_2D)
    _set(arrays["0"], 0, 5)
    builder.add_series("y", _image(models), SPEC_2D)
    builder.prepare()
    builder.add_series("z", _image(models), SPEC_2D)
    builder.prepare()  # (used to delete everything again)
    assert _listing(dest) == ["x", "y", "z"]
    assert not (dest / "0").exists()
    assert (_get(arrays["0"], 0) == 5).all()

    # (but something that isn't a Zarr group isn't replaced)
    other = tmp_path / "other"
    _other_dir(other, w, models)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        w.Bf2RawBuilder(other, writer=WRITERS[0], overwrite=True).prepare_series(
            "s", _image(models), SPEC_2D
        )
    assert (other / "mine.txt").exists()


def test_leftover_temp_files_are_harmless(tmp_path: Path, ver: Any) -> None:
    # (as left by a process killed mid-write)
    models, w = ver
    dest = tmp_path / "out.zarr"
    _bf2raw_store(w, models, dest, "0")
    for directory in (dest, dest / "OME"):
        (directory / f".zarr.json.{'0' * 8}.tmp").write_text("{")
    yaozarrs.validate_zarr_store(dest)
    w.Bf2RawBuilder(dest, writer=WRITERS[0]).prepare_series(
        "1", _image(models), SPEC_2D
    )
    yaozarrs.validate_zarr_store(dest)


def test_metadata_is_replaced_into_place(
    tmp_path: Path, ver: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # (each file is written complete next to it, then renamed over it)
    models, w = ver
    dest = tmp_path / "out.zarr"
    replaced: list[Path] = []
    real = _groups._replace

    def spy(src: Path, dst: Path) -> None:
        assert src.parent == dst.parent
        if dst.suffix == ".json":
            json.loads(src.read_bytes())
        replaced.append(dst)
        real(src, dst)

    monkeypatch.setattr(_groups, "_replace", spy)
    builder = w.Bf2RawBuilder(dest, writer=WRITERS[0], ome_xml="<OME/>")
    builder.prepare_series("0", _image(models), SPEC_2D)
    builder.prepare_series("1", _image(models), SPEC_2D)
    assert {dest / "zarr.json", dest / "OME" / "METADATA.ome.xml"} <= set(replaced)
    assert replaced.count(dest / "OME" / "zarr.json") == 2


@pytest.mark.parametrize("case", ["win32", "linux", "interrupted"])
def test_failed_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    # (on Windows, a file another program has open can't be replaced: retry, then
    # say; and whatever fails, the file is untouched, with no temporary file left)
    target = tmp_path / "zarr.json"
    target.write_text("old")
    monkeypatch.setattr(_groups.sys, "platform", "win32" if case == "win32" else "")
    monkeypatch.setattr(_groups.time, "sleep", lambda _: None)
    error = KeyboardInterrupt() if case == "interrupted" else PermissionError(13, "")
    monkeypatch.setattr(_groups.os, "replace", replace := Mock(side_effect=error))
    with pytest.raises(type(error)) as info:
        _groups._atomic_write(target, "new")
    if case == "win32":
        assert replace.call_count == 8
        assert "open in another program" in str(info.value)
        assert info.value.__cause__ is error
    else:
        assert replace.call_count == 1
        assert info.value is error
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["zarr.json"]


def test_per_series_array_settings(tmp_path: Path, ver: Any, writer: str) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    builder = w.Bf2RawBuilder(dest, writer=writer, chunks=(1, 16, 16))
    builder.prepare_series("3d", _image(models, CYX), SPEC_3D)
    with pytest.raises(ValueError, match="one entry per dimension"):
        builder.prepare_series("2d", _image(models), SPEC_2D)  # (the builder's chunks)
    builder.prepare_series("2d", _image(models), SPEC_2D, chunks=(8, 8))
    builder.prepare_series(
        "sharded",
        _image(models),
        ((32, 32), "uint8"),
        chunks=(8, 8),
        shards=(16, 16),
        compression="none",
    )
    yaozarrs.validate_zarr_store(dest)

    def meta(name: str) -> dict:
        return json.loads((dest / name / "0" / "zarr.json").read_bytes())

    def grid(name: str) -> list[int]:
        return meta(name)["chunk_grid"]["configuration"]["chunk_shape"]

    assert grid("3d") == [1, 16, 16]
    assert grid("2d") == [8, 8]
    assert grid("sharded") == [16, 16]
    codec_names = json.dumps(meta("sharded")["codecs"])
    assert "sharding_indexed" in codec_names
    assert "blosc" not in codec_names and "zstd" not in codec_names
