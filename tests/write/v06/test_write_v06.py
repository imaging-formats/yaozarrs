"""Tests for OME-Zarr v0.6 write functionality.

The writing machinery is shared with v0.5 (and tested in depth there); these tests
check that every v0.6 workflow produces a store that validates as v0.6.
"""

from __future__ import annotations

import doctest
import importlib.metadata
import importlib.util
import json
from typing import TYPE_CHECKING, Any

import pytest

import yaozarrs
from yaozarrs import DimSpec, v06
from yaozarrs.write.v06 import (
    Bf2RawBuilder,
    LabelsBuilder,
    PlateBuilder,
    _write,
    prepare_image,
    write_bioformats2raw,
    write_image,
    write_plate,
)

if TYPE_CHECKING:
    from pathlib import Path

    from yaozarrs.typing import ZarrWriter

np = pytest.importorskip("numpy")

WRITERS: list[ZarrWriter] = []
if importlib.util.find_spec("zarr") is not None:
    if int(importlib.metadata.version("zarr").split(".")[0]) >= 3:
        WRITERS.append("zarr")
if importlib.util.find_spec("tensorstore") is not None:
    WRITERS.append("tensorstore")
if not WRITERS:
    pytest.skip("No supported Zarr writer found", allow_module_level=True)

CYX = [
    DimSpec(name="c"),
    DimSpec(name="y", scale=0.5, unit="micrometer"),
    DimSpec(name="x", scale=0.5, unit="micrometer"),
]


def _image(n_levels: int = 1, **kwargs: Any) -> v06.Image:
    return v06.Image(
        multiscales=[v06.Multiscale.from_dims(CYX, n_levels=n_levels, **kwargs)]
    )


def _pyramid(n_levels: int, dtype: str = "uint16") -> list[np.ndarray]:
    return [np.zeros((2, 64 // 2**i, 64 // 2**i), dtype=dtype) for i in range(n_levels)]


def _label_image(**kwargs: Any) -> v06.LabelImage:
    ms = v06.Multiscale.from_dims(CYX[1:], **kwargs)
    return v06.LabelImage(multiscales=[ms], image_label=v06.ImageLabel())


def _ome(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text())["attributes"]["ome"]


def _validate(path: Path) -> None:
    # (warnings are errors in this test suite, so this also checks for warnings)
    yaozarrs.validate_zarr_store(path)
    for zarr_json in path.rglob("zarr.json"):
        if ome := json.loads(zarr_json.read_text()).get("attributes", {}).get("ome"):
            assert ome["version"] == "0.6", zarr_json


# ---------------------------------------------------------------- images


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_multiscale(tmp_path: Path, writer: ZarrWriter) -> None:
    dest = write_image(tmp_path / "img.zarr", _image(3), _pyramid(3), writer=writer)
    _validate(dest)

    ms = _ome(dest)["multiscales"][0]
    assert [cs["name"] for cs in ms["coordinateSystems"]] == ["intrinsic"]
    # every level maps into the intrinsic coordinate system...
    for i, ds in enumerate(ms["datasets"]):
        (transform,) = ds["coordinateTransformations"]
        assert transform["input"] == {"path": str(i)}
        assert transform["output"] == {"name": "intrinsic"}
    # ... and the downsampled ones carry a (half-pixel) translation, as a sequence
    assert ms["datasets"][0]["coordinateTransformations"][0]["type"] == "scale"
    seq = ms["datasets"][1]["coordinateTransformations"][0]
    assert [t["type"] for t in seq["transformations"]] == ["scale", "translation"]
    assert seq["transformations"][1]["translation"] == [0, 0.25, 0.25]


@pytest.mark.parametrize("writer", WRITERS)
def test_dimension_names(tmp_path: Path, writer: ZarrWriter) -> None:
    # (no longer required in v0.6, but written, as in v0.5, from the intrinsic axes)
    image = _image(coordinate_system="physical")
    dest = write_image(tmp_path / "img.zarr", image, _pyramid(1), writer=writer)
    _validate(dest)
    array_json = json.loads((dest / "0" / "zarr.json").read_text())
    assert array_json["dimension_names"] == ["c", "y", "x"]


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_5d(tmp_path: Path, writer: ZarrWriter) -> None:
    dims = [
        DimSpec(name="t", scale=2.0, unit="second"),
        DimSpec(name="c"),
        DimSpec(name="z", scale=1.0, unit="micrometer", scale_factor=1.0),
        DimSpec(name="y", scale=0.5, unit="micrometer"),
        DimSpec(name="x", scale=0.5, unit="micrometer"),
    ]
    image = v06.Image(multiscales=[v06.Multiscale.from_dims(dims, n_levels=2)])
    data = [np.zeros((2, 2, 4, 32, 32), "uint8"), np.zeros((2, 2, 4, 16, 16), "uint8")]
    dest = write_image(tmp_path / "img.zarr", image, data, writer=writer)
    _validate(dest)
    array_json = json.loads((dest / "1" / "zarr.json").read_text())
    assert array_json["dimension_names"] == ["t", "c", "z", "y", "x"]


@pytest.mark.parametrize("writer", WRITERS)
def test_extra_coordinate_systems(tmp_path: Path, writer: ZarrWriter) -> None:
    ms = v06.Multiscale.from_dims(CYX).model_dump(exclude_unset=True, by_alias=True)
    ms["coordinateSystems"].append(
        {
            "name": "stage",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
        }
    )
    ms["coordinateTransformations"] = [
        {
            "type": "translation",
            "translation": [0, 100, 200],
            "input": {"name": "intrinsic"},
            "output": {"name": "stage"},
        }
    ]
    image = v06.Image(multiscales=[v06.Multiscale.model_validate(ms)])
    dest = write_image(tmp_path / "img.zarr", image, _pyramid(1), writer=writer)
    _validate(dest)
    written = _ome(dest)["multiscales"][0]
    assert written["coordinateTransformations"][0]["output"] == {"name": "stage"}


@pytest.mark.parametrize("writer", WRITERS)
def test_prepare_image(tmp_path: Path, writer: ZarrWriter) -> None:
    specs = [(a.shape, a.dtype) for a in _pyramid(2)]
    dest, arrays = prepare_image(tmp_path / "img.zarr", _image(2), specs, writer=writer)
    assert set(arrays) == {"0", "1"}
    _validate(dest)


# ---------------------------------------------------------------- labels


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_with_labels(tmp_path: Path, writer: ZarrWriter) -> None:
    labels = {
        "cells": (
            _label_image(n_levels=2),
            [np.zeros((64, 64), "uint8"), np.zeros((32, 32), "uint8")],
        )
    }
    image = v06.Image(multiscales=[v06.Multiscale.from_dims(CYX[1:], n_levels=2)])
    data = [np.zeros((64, 64), "uint16"), np.zeros((32, 32), "uint16")]
    dest = write_image(tmp_path / "img.zarr", image, data, labels=labels, writer=writer)
    _validate(dest)
    assert _ome(dest / "labels") == {"version": "0.6", "labels": ["cells"]}
    assert "image-label" in _ome(dest / "labels" / "cells")


@pytest.mark.parametrize("writer", WRITERS)
def test_image_to_label_transform(tmp_path: Path, writer: ZarrWriter) -> None:
    # the writer adds no image -> label transformation itself, but one can be given
    ms = v06.Multiscale.from_dims(CYX[1:]).model_dump(exclude_unset=True, by_alias=True)
    ms["coordinateTransformations"] = [
        {
            "type": "identity",
            "input": {"name": "intrinsic"},
            "output": {"name": "intrinsic", "path": "labels/cells"},
        }
    ]
    image = v06.Image(multiscales=[v06.Multiscale.model_validate(ms)])
    labels = {"cells": (_label_image(), np.zeros((64, 64), "uint8"))}
    data = np.zeros((64, 64), "uint16")
    dest = write_image(tmp_path / "img.zarr", image, data, labels=labels, writer=writer)
    _validate(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_labels_builder(tmp_path: Path, writer: ZarrWriter) -> None:
    image = v06.Image(multiscales=[v06.Multiscale.from_dims(CYX[1:])])
    dest = write_image(tmp_path / "img.zarr", image, np.zeros((64, 64), "uint16"))
    builder = LabelsBuilder(dest / "labels", writer=writer)
    builder.write_label("cells", _label_image(), np.zeros((64, 64), "uint8"))
    # (a second builder picks up the existing labels group)
    builder = LabelsBuilder(dest / "labels", writer=writer)
    builder.add_label("nuclei", _label_image(), ((64, 64), "uint8"))
    _, arrays = builder.prepare()
    assert set(arrays) == {"nuclei/0"}
    assert _ome(dest / "labels")["labels"] == ["cells", "nuclei"]
    _validate(dest)


# ---------------------------------------------------------------- plates


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate(tmp_path: Path, writer: ZarrWriter) -> None:
    images = {
        (row, col, fov): (_image(), _pyramid(1))
        for row in "AB"
        for col in ("1", "2")
        for fov in ("0", "1")
    }
    dest = write_plate(
        tmp_path / "plate.zarr", images, plate={"name": "p"}, writer=writer
    )
    _validate(dest)
    plate = _ome(dest)["plate"]
    assert plate["name"] == "p"
    assert [w["path"] for w in plate["wells"]] == ["A/1", "A/2", "B/1", "B/2"]
    assert _ome(dest / "A" / "1")["well"]["images"] == [{"path": "0"}, {"path": "1"}]


@pytest.mark.parametrize("writer", WRITERS)
def test_plate_builder(tmp_path: Path, writer: ZarrWriter) -> None:
    plate = v06.Plate(
        plate=v06.PlateDef(
            columns=[v06.Column(name="1")],
            rows=[v06.Row(name="A"), v06.Row(name="B")],
            wells=[
                v06.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
                v06.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
            ],
        )
    )
    builder = PlateBuilder(tmp_path / "plate.zarr", plate=plate, writer=writer)
    builder.add_well(
        row="A", col="1", images={"0": (_image(), ((2, 64, 64), "uint16"))}
    )
    builder.add_well(
        row="B", col="1", images={"0": (_image(), ((2, 64, 64), "uint16"))}
    )
    dest, arrays = builder.prepare()
    assert set(arrays) == {"A/1/0/0", "B/1/0/0"}
    _validate(dest)


def test_plate_field_names(tmp_path: Path) -> None:
    # v0.6 node names: only [A-Za-z0-9_.-], not only periods, no "__" prefix
    images = {
        ("A", "1", "0"): (_image(), _pyramid(1)),
        ("B", "1", "f 0"): (_image(), _pyramid(1)),
    }
    with pytest.raises(ValueError, match="FieldOfView"):
        write_plate(tmp_path / "plate.zarr", images)
    # (checked before anything is written)
    assert not (tmp_path / "plate.zarr").exists()


@pytest.mark.parametrize("writer", WRITERS)
def test_plate_builder_write_well(tmp_path: Path, writer: ZarrWriter) -> None:
    builder = PlateBuilder(tmp_path / "plate.zarr", writer=writer)
    builder.write_well(row="A", col="2", images={"0": (_image(), _pyramid(1))})
    builder.write_well(row="C", col="1", images={"0": (_image(), _pyramid(1))})
    _validate(builder.root_path)
    plate = _ome(builder.root_path)["plate"]
    assert [r["name"] for r in plate["rows"]] == ["A", "B", "C"]
    assert [w["path"] for w in plate["wells"]] == ["A/2", "C/1"]


# ---------------------------------------------------------------- bioformats2raw


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw(tmp_path: Path, writer: ZarrWriter) -> None:
    images = {"0": (_image(2), _pyramid(2)), "1": (_image(), _pyramid(1))}
    dest = write_bioformats2raw(
        tmp_path / "b.zarr", images, ome_xml="<OME/>", writer=writer
    )
    _validate(dest)
    assert _ome(dest) == {"version": "0.6", "bioformats2raw.layout": 3}
    assert _ome(dest / "OME") == {"version": "0.6", "series": ["0", "1"]}


@pytest.mark.parametrize("writer", WRITERS)
def test_bf2raw_builder_prepare(tmp_path: Path, writer: ZarrWriter) -> None:
    builder = Bf2RawBuilder(tmp_path / "b.zarr", writer=writer)
    builder.add_series("0", _image(), ((2, 64, 64), "uint16"))
    dest, arrays = builder.prepare()
    assert set(arrays) == {"0/0"}
    _validate(dest)


# ---------------------------------------------------------------- doctests

finder = doctest.DocTestFinder()


@pytest.mark.skipif("zarr" not in WRITERS, reason="zarr writer not available")
@pytest.mark.parametrize(
    "case",
    [test for test in finder.find(_write) if test.examples],
    ids=lambda t: t.name.split(".")[-1],
)
def test_write_doctests_parametrized(
    tmp_path: Path,
    case: doctest.DocTest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.REPORTING_FLAGS
    )
    # put all paths inside the test tmp_path
    monkeypatch.setattr(_write, "Path", lambda p: tmp_path / p)
    runner.run(case)
    if runner.failures > 0:
        captured = capsys.readouterr().out.split("******************")[-1]
        pytest.fail(f"Doctest {case.name} failed:\n\n{captured}")

    for result in tmp_path.glob("*.zarr"):
        if result.is_dir() and (result / "zarr.json").exists():
            _validate(result)
