"""Incremental bioformats2raw building: `Bf2RawBuilder.prepare_series()` and co.

The store must be a valid collection after every call, and at any moment once it
exists (even if the process dies mid-call).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import threading
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


def _ome(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_bytes())["attributes"]["ome"]


def _listing(dest: Path) -> list[str]:
    return _ome(dest / "OME")["series"]


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


# ---------------------------------------------------------------- the happy path


def test_valid_after_every_call(tmp_path: Path, ver: Any, writer: str) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    builder = w.Bf2RawBuilder(dest, writer=writer, ome_xml="<OME/>")
    assert not dest.exists()  # (constructing writes nothing)
    assert builder.series == ()

    path, arrays = builder.prepare_series("0", _image(models, CYX), SPEC_3D)
    assert path == dest / "0"
    assert set(arrays) == {"0"}
    yaozarrs.validate_zarr_store(dest)
    _set(arrays["0"], 0, 1)

    # nested, with its own array settings
    builder.prepare_series(
        "group_1/array_1_0",
        _image(models),
        SPEC_2D,
        chunks=(8, 8),
        compression="zstd",
    )
    yaozarrs.validate_zarr_store(dest)
    assert json.loads((dest / "group_1" / "zarr.json").read_bytes())["node_type"]

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

    # a later builder adds to the collection
    resumed = w.Bf2RawBuilder(dest, writer=writer)
    assert resumed.series == builder.series
    resumed.prepare_series("5", _image(models), SPEC_2D)
    yaozarrs.validate_zarr_store(dest)

    expected = ["0", "group_1/array_1_0", "1", "2", "3", "4", "5"]
    assert _listing(dest) == expected
    assert resumed.series == tuple(expected)
    assert (dest / "OME" / "METADATA.ome.xml").read_text() == "<OME/>"
    assert not list(dest.rglob("*.tmp"))
    assert repr(resumed) == "<Bf2RawBuilder: 7 images>"


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


# ---------------------------------------------------------------- crashes


class _Crash(BaseException):
    """A process dying: nothing after it runs (cleanup is disabled too)."""


def _build_new(w: ModuleType, models: ModuleType, dest: Path) -> None:
    builder = w.Bf2RawBuilder(dest, writer="zarr", ome_xml="<OME/>")
    builder.prepare_series(
        "0", _image(models, n_levels=2), [SPEC_2D, ((8, 8), "uint8")]
    )
    builder.write_image("a/b", _image(models), np.ones((16, 16), "uint8"))


def _build_overwrite(w: ModuleType, models: ModuleType, dest: Path) -> None:
    builder = w.Bf2RawBuilder(dest, writer="zarr", overwrite=True)
    builder.prepare_series("0", _image(models), SPEC_2D)
    builder.prepare_series("1", _image(models), SPEC_2D)


SEQUENCES = {
    "new": (None, _build_new),
    "overwrite": (lambda w, m, d: _bf2raw_store(w, m, d, "0", "1"), _build_overwrite),
}


@needs_zarr
@pytest.mark.parametrize("sequence", SEQUENCES)
def test_crash_at_any_step(
    tmp_path: Path, ver: Any, sequence: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a crash at any step, the store is absent or valid (and extendable)."""
    models, w = ver
    setup, build = SEQUENCES[sequence]
    steps = {"n": 0, "crash_at": -1}

    def counted(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if steps["n"] == steps["crash_at"]:
                raise _Crash
            steps["n"] += 1
            return func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(_builders, "_discard", lambda path: None)
    for module, name in [
        (_groups, "_atomic_write"),
        (_builders, "_atomic_write"),
        (_arrays, "_create_array_zarr"),
    ]:
        monkeypatch.setattr(module, name, counted(getattr(module, name)))

    for crash_at in range(100):
        dest = tmp_path / f"crash_{crash_at}.zarr"
        steps.update(n=-(10**9), crash_at=-1)  # (setup steps aren't counted)
        if setup is not None:
            setup(w, models, dest)
        steps.update(n=0, crash_at=crash_at)
        try:
            build(w, models, dest)
        except _Crash:
            pass
        else:
            break  # (every step has been crashed at)
        steps.update(crash_at=-1)

        # (until the root zarr.json is written, there's no store at all)
        if (dest / "zarr.json").exists():
            yaozarrs.validate_zarr_store(dest)
            resumed = w.Bf2RawBuilder(dest, writer="zarr")
            resumed.prepare_series("resumed", _image(models), SPEC_2D)
            yaozarrs.validate_zarr_store(dest)
    else:  # pragma: no cover
        pytest.fail("the sequence never completed")
    assert crash_at > 3


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
    assert not (dest / "1").exists()
    assert _snapshot(dest) == before
    yaozarrs.validate_zarr_store(dest)

    builder.write_image("1", _image(models), np.ones((16, 16), "uint8"))  # (retry)
    yaozarrs.validate_zarr_store(dest)
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


# ---------------------------------------------------------------- concurrency


def test_never_invalid_while_building(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    problems: list[BaseException] = []
    done = threading.Event()

    def observe() -> None:
        while not done.is_set():
            # (until the root zarr.json is written, there's no store at all)
            existed = (dest / "zarr.json").exists()
            try:
                yaozarrs.validate_zarr_store(dest)
            except FileNotFoundError as e:
                if existed:  # pragma: no cover
                    problems.append(e)
            except Exception as e:  # pragma: no cover
                problems.append(e)

    observer = threading.Thread(target=observe)
    observer.start()
    try:
        builder = w.Bf2RawBuilder(dest, writer=WRITERS[0])
        for i in range(30):
            builder.prepare_series(str(i), _image(models), SPEC_2D)
    finally:
        done.set()
        observer.join()
    assert not problems
    assert len(_listing(dest)) == 30


# ---------------------------------------------------------------- names


@pytest.mark.parametrize(
    ("first", "second", "error", "match"),
    [
        ("a", "a", ValueError, "already created via prepare_series"),
        ("a", "a/b", ValueError, "can't be nested in"),
        ("a/b", "a", FileExistsError, "isn't a series"),
        (None, "OME", ValueError, "metadata"),
        (None, "ome/x", ValueError, "metadata"),
        (None, "a//b", ValueError, "Invalid image path"),
        (None, "__a", ValueError, "Invalid image path"),
        (None, "..", ValueError, "Invalid image path"),
    ],
)
def test_name_errors(
    tmp_path: Path,
    ver: Any,
    first: str | None,
    second: str,
    error: type[Exception],
    match: str,
) -> None:
    models, w = ver
    builder = w.Bf2RawBuilder(tmp_path / "out.zarr", writer=WRITERS[0])
    if first is not None:
        builder.prepare_series(first, _image(models), SPEC_2D)
    with pytest.raises(error, match=match):
        builder.prepare_series(second, _image(models), SPEC_2D)


def test_existing_series_names(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    _bf2raw_store(w, models, dest, "0")
    before = _snapshot(dest)
    builder = w.Bf2RawBuilder(dest, writer=WRITERS[0])
    with pytest.raises(FileExistsError, match="already exists"):
        builder.prepare_series("0", _image(models), SPEC_2D)
    with pytest.raises(FileExistsError, match="already exists"):
        builder.add_series("0", _image(models), SPEC_2D)
    assert _snapshot(dest) == before


def test_unlisted_path_is_not_replaced(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    _bf2raw_store(w, models, dest, "0")
    (dest / "leftover").mkdir()
    builder = w.Bf2RawBuilder(dest, writer=WRITERS[0])
    with pytest.raises(FileExistsError, match="isn't a series"):
        builder.prepare_series("leftover", _image(models), SPEC_2D)
    assert (dest / "leftover").exists()


# ---------------------------------------------------------------- existing stores


def _v2_group(dest: Path) -> None:
    dest.mkdir()
    (dest / ".zgroup").write_text('{"zarr_format": 2}')


def _plain_root(dest: Path) -> None:
    dest.mkdir()
    _groups._write_group_json(dest)


def _plate(dest: Path) -> None:
    dest.mkdir()
    (dest / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ome": {"version": "0.5", "plate": {}}},
            }
        )
    )


def _other_dir(dest: Path) -> None:
    dest.mkdir()
    (dest / "notes.txt").write_text("mine")


def _numbered(dest: Path) -> None:
    # (a collection without a `series` list: its series are its numbered groups)
    other_models, other_w = (v06, w06) if dest.name == "v06" else (v05, w05)
    _bf2raw_store(other_w, other_models, dest, "0")
    version = _ome(dest)["version"]
    _groups._write_group_json(dest / "OME", {"ome": {"version": version}})


def _other_version(dest: Path) -> None:
    other_models, other_w = (v06, w06) if dest.name == "v05" else (v05, w05)
    _bf2raw_store(other_w, other_models, dest, "0")


@pytest.mark.parametrize(
    "make", [_v2_group, _plain_root, _plate, _other_dir, _numbered, _other_version]
)
def test_refuses_other_content(tmp_path: Path, ver: Any, make: Callable) -> None:
    models, w = ver
    dest = tmp_path / models.__name__.rsplit(".", 1)[-1]
    make(dest)
    before = _snapshot(dest)
    with pytest.raises(FileExistsError):
        w.Bf2RawBuilder(dest, writer=WRITERS[0])
    # (with overwrite, groups are replaced, but not unrelated directories)
    builder = w.Bf2RawBuilder(dest, writer=WRITERS[0], overwrite=True)
    if make is _other_dir:
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            builder.prepare_series("0", _image(models), SPEC_2D)
        assert _snapshot(dest) == before
    else:
        builder.prepare_series("0", _image(models), SPEC_2D)
        yaozarrs.validate_zarr_store(dest)


def test_overwrite_replaces_once(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    _bf2raw_store(w, models, dest, "0", "1")
    builder = w.Bf2RawBuilder(dest, writer=WRITERS[0], overwrite=True)
    assert _listing(dest) == ["0", "1"]  # (nothing is deleted until a series is made)

    _, arrays = builder.prepare_series("x", _image(models), SPEC_2D)
    _set(arrays["0"], 0, 5)
    builder.add_series("y", _image(models), SPEC_2D)
    builder.prepare()
    builder.add_series("z", _image(models), SPEC_2D)
    builder.prepare()  # (used to delete everything again)
    assert _listing(dest) == ["x", "y", "z"]
    assert not (dest / "0").exists()
    assert (_get(arrays["0"], 0) == 5).all()


def test_write_bioformats2raw_still_refuses_existing(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    _bf2raw_store(w, models, dest, "0")
    with pytest.raises(FileExistsError, match="already exists"):
        _bf2raw_store(w, models, dest, "1")
    with pytest.raises(ValueError, match="Series '1'"):  # (series are checked first)
        w.write_bioformats2raw(dest, {"1": (_image(models), np.zeros(4, "uint8"))})


def test_ome_xml_and_extra_attributes(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    xml = dest / "OME" / "METADATA.ome.xml"
    builder = w.Bf2RawBuilder(
        dest, writer=WRITERS[0], ome_xml="<OME>1</OME>", extra_attributes={"a": 1}
    )
    builder.prepare_series("0", _image(models), SPEC_2D)
    assert xml.read_text() == "<OME>1</OME>"

    # extending a collection replaces its OME-XML, and merges extra attributes into
    # its root
    later = w.Bf2RawBuilder(
        dest, writer=WRITERS[0], ome_xml="<OME>2</OME>", extra_attributes={"b": 2}
    )
    later.prepare_series("1", _image(models), SPEC_2D)
    assert xml.read_text() == "<OME>2</OME>"
    attrs = json.loads((dest / "zarr.json").read_bytes())["attributes"]
    assert attrs["a"] == 1 and attrs["b"] == 2
    assert attrs["ome"]["bioformats2raw.layout"] == 3
    yaozarrs.validate_zarr_store(dest)


def test_leftover_temp_files_are_harmless(tmp_path: Path, ver: Any) -> None:
    models, w = ver
    dest = tmp_path / "out.zarr"
    w.Bf2RawBuilder(dest, writer=WRITERS[0]).prepare_series(
        "0", _image(models), SPEC_2D
    )
    for directory in (dest, dest / "OME"):
        (directory / f".zarr.json.{'0' * 32}.tmp").write_text("{")
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_replace_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    # (on Windows, a file another program has open can't be replaced: retry, then say)
    monkeypatch.setattr(_groups.sys, "platform", platform)
    monkeypatch.setattr(_groups.time, "sleep", lambda _: None)
    denied = PermissionError(13, "Access is denied")
    monkeypatch.setattr(_groups.os, "replace", replace := Mock(side_effect=denied))
    with pytest.raises(PermissionError) as info:
        _groups._atomic_write(tmp_path / "zarr.json", "{}")
    if platform == "win32":
        assert replace.call_count == 6
        assert "open in another program" in str(info.value)
        assert info.value.__cause__ is denied
    else:
        assert replace.call_count == 1
        assert info.value is denied
    assert not list(tmp_path.iterdir())  # (the temporary file is removed)


# ---------------------------------------------------------------- arrays


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
