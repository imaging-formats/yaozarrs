"""Each writer only accepts its own version's models."""

from __future__ import annotations

import importlib
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest

from yaozarrs import DimSpec, v04, v05, v06
from yaozarrs.write import v05 as write_v05
from yaozarrs.write import v06 as write_v06

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import ModuleType

np = pytest.importorskip("numpy")

V04_IMAGE = v04.Image.model_validate(
    {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1, 1]}
                        ],
                    }
                ],
            }
        ]
    }
)
V04_PLATE = v04.Plate.model_validate(
    {
        "plate": {
            "version": "0.4",
            "columns": [{"name": "1"}],
            "rows": [{"name": "A"}],
            "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
        }
    }
)


def _v05_image() -> v05.Image:
    return v05.Image.model_validate(
        {
            "multiscales": [
                {
                    "axes": [
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1, 1]}
                            ],
                        }
                    ],
                }
            ]
        }
    )


# (writer module, a valid image for it, an image from another version, the
# other version's plate, and the name the error should give the expected type)
def _v06_image() -> v06.Image:
    dims = [DimSpec(name="y"), DimSpec(name="x")]
    return v06.Image(multiscales=[v06.Multiscale.from_dims(dims)])


V05_PLATE = v05.Plate.model_validate(
    V04_PLATE.model_dump(exclude={"plate": {"version"}})
)

CASES = [
    (write_v05, _v05_image, V04_IMAGE, V04_PLATE, "v05"),
    (write_v06, _v06_image, _v05_image(), V05_PLATE, "v06"),
]
IDS = ["v05", "v06"]
DATA = np.zeros((4, 4), dtype=np.uint8)
SPEC = ((4, 4), "uint8")


def _calls(
    mod: ModuleType, good: Callable[[], Any], bad: Any, plate: Any
) -> dict[str, tuple[str, Callable[[Path], Any]]]:
    """{name: (the model kind that is rejected, call)}."""
    field = {("A", "1", "0"): (bad, DATA)}
    return {
        "write_image": ("Image", lambda p: mod.write_image(p, bad, DATA)),
        "write_image_label": (
            "Image",
            lambda p: mod.write_image(p, good(), DATA, labels={"x": (bad, DATA)}),
        ),
        "prepare_image": ("Image", lambda p: mod.prepare_image(p, bad, SPEC)),
        "write_plate": ("Image", lambda p: mod.write_plate(p, field)),
        "write_plate_second": (
            # (a good field first: nothing may be written before the bad one is found)
            "Image",
            lambda p: mod.write_plate(p, {("A", "1", "0"): (good(), DATA), **field}),
        ),
        "write_plate_plate": (
            "Plate",
            lambda p: mod.write_plate(
                p, {("A", "1", "0"): (good(), DATA)}, plate=plate
            ),
        ),
        "PlateBuilder": ("Plate", lambda p: mod.PlateBuilder(p, plate=plate)),
        "PlateBuilder.write_well": (
            "Image",
            lambda p: mod.PlateBuilder(p).write_well(
                row="A", col="1", images={"0": (bad, DATA)}
            ),
        ),
        "PlateBuilder.add_well": (
            "Image",
            lambda p: mod.PlateBuilder(p).add_well(
                row="A", col="1", images={"0": (bad, SPEC)}
            ),
        ),
        "write_bioformats2raw": (
            "Image",
            lambda p: mod.write_bioformats2raw(p, {"0": (bad, DATA)}),
        ),
        "write_bioformats2raw_second": (
            "Image",
            lambda p: mod.write_bioformats2raw(
                p, {"0": (good(), DATA), "1": (bad, DATA)}
            ),
        ),
        "Bf2RawBuilder.write_image": (
            "Image",
            lambda p: mod.Bf2RawBuilder(p).write_image("0", bad, DATA),
        ),
        "Bf2RawBuilder.add_series": (
            "Image",
            lambda p: mod.Bf2RawBuilder(p).add_series("0", bad, SPEC),
        ),
        "LabelsBuilder.write_label": (
            "Image",
            lambda p: mod.LabelsBuilder(p).write_label("x", bad, DATA),
        ),
        "LabelsBuilder.add_label": (
            "Image",
            lambda p: mod.LabelsBuilder(p).add_label("x", bad, SPEC),
        ),
    }


CALL_NAMES = list(_calls(write_v05, _v05_image, V04_IMAGE, V04_PLATE))


@pytest.mark.parametrize("call", CALL_NAMES)
@pytest.mark.parametrize(("mod", "good", "bad", "plate", "version"), CASES, ids=IDS)
def test_other_version_model_rejected(
    tmp_path: Path,
    mod: ModuleType,
    good: Callable[[], Any],
    bad: Any,
    plate: Any,
    version: str,
    call: str,
) -> None:
    dest = tmp_path / "out.zarr"
    kind, func = _calls(mod, good, bad, plate)[call]
    other = type(bad).__module__.split(".")[1]
    match = f"expected a {version}.{kind}, got {other}.{kind}"
    with pytest.raises(TypeError, match=re.escape(match)):
        func(dest)
    # nothing was written before the check
    assert not dest.exists()


@pytest.mark.parametrize(("mod", "good", "bad", "plate", "version"), CASES, ids=IDS)
def test_plain_image_as_label(
    tmp_path: Path,
    mod: ModuleType,
    good: Callable[[], Any],
    bad: Any,
    plate: Any,
    version: str,
) -> None:
    # `image-label` is a SHOULD, so an Image without it is still accepted as a label
    pytest.importorskip("zarr")
    dest = mod.write_image(
        tmp_path / "out.zarr", good(), DATA, labels={"x": (good(), DATA)}
    )
    assert (dest / "labels" / "x" / "zarr.json").exists()


@pytest.mark.parametrize(("mod", "good", "bad", "plate", "version"), CASES, ids=IDS)
def test_accepted_models(
    tmp_path: Path,
    mod: ModuleType,
    good: Callable[[], Any],
    bad: Any,
    plate: Any,
    version: str,
) -> None:
    pytest.importorskip("zarr")
    image = good()
    # a subclass (LabelImage) is still an Image
    models = importlib.import_module(f"yaozarrs.{version}")
    label_image = models.LabelImage.model_validate(
        {**image.model_dump(), "image-label": {}}
    )
    dest = mod.write_image(tmp_path / "a.zarr", label_image, DATA)
    assert (dest / "zarr.json").exists()
    # and plate metadata can be any mapping
    images = {("A", "1", "0"): (good(), DATA)}
    dest = mod.write_plate(
        tmp_path / "b.zarr", images, plate=MappingProxyType({"name": "x"})
    )
    assert (dest / "A" / "1" / "0" / "zarr.json").exists()


@pytest.mark.parametrize(("mod", "good", "bad", "plate", "version"), CASES, ids=IDS)
def test_bad_field_writes_nothing(
    tmp_path: Path,
    mod: ModuleType,
    good: Callable[[], Any],
    bad: Any,
    plate: Any,
    version: str,
) -> None:
    # (a later field that doesn't match its metadata fails before the first is written)
    dest = tmp_path / "out.zarr"
    images = {("A", "1", "0"): (good(), DATA), ("B", "1", "0"): (good(), [DATA, DATA])}
    with pytest.raises(ValueError, match="Number of data arrays"):
        mod.write_plate(dest, images)
    assert not dest.exists()


def test_not_a_model(tmp_path: Path) -> None:
    not_a_plate: Any = "A/1"
    with pytest.raises(TypeError, match=r"plate: expected a v05\.Plate, got str"):
        write_v05.PlateBuilder(tmp_path / "p.zarr", plate=not_a_plate)


@pytest.mark.parametrize(("mod", "good", "bad", "plate", "version"), CASES, ids=IDS)
def test_data_checked_before_writing(
    tmp_path: Path,
    mod: ModuleType,
    good: Callable[[], Any],
    bad: Any,
    plate: Any,
    version: str,
) -> None:
    dest = tmp_path / "out.zarr"
    # data with the wrong number of dimensions for the axes
    with pytest.raises(ValueError, match=r"has 3 dimensions .*, but the image has 2"):
        mod.write_image(dest, good(), np.zeros((1, 4, 4), "uint8"))
    with pytest.raises(ValueError, match=r"has 1 dimensions .*, but the image has 2"):
        mod.prepare_image(dest, good(), ((4,), "uint8"))
    # The pixels of the label images MUST be integer data types
    labels = {"cells": (good(), np.zeros((4, 4), "float32"))}
    with pytest.raises(ValueError, match="MUST have an integer data type, got float32"):
        mod.write_image(dest, good(), DATA, labels=labels)
    with pytest.raises(ValueError, match="integer data type"):
        mod.LabelsBuilder(dest / "labels").add_label("x", good(), ((4, 4), "float64"))
    assert not dest.exists()
