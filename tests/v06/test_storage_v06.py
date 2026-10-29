from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from yaozarrs import validate_zarr_store
from yaozarrs._storage import StorageErrorType, StorageValidationError

pytest.importorskip("fsspec")

if TYPE_CHECKING:
    from collections.abc import Callable

DATA = Path(__file__).parent.parent / "data" / "v06"
STORES = ["bf2raw", "image_2d", "image_tczyx", "image_transforms", "image_with_labels"]
STORES += ["image_zyx_oriented", "plate"]


def _write_array(path: Path, ndim: int, dtype: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [8] * ndim,
        "data_type": dtype,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8] * ndim}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    (path / "zarr.json").write_text(json.dumps(meta))


def make_store(tmp_path: Path, name: str) -> Path:
    """Copy a tests/data/v06 store, adding array metadata for every dataset."""
    root = shutil.copytree(DATA / f"{name}.ome.zarr", tmp_path / f"{name}.ome.zarr")
    for zarr_json in list(root.rglob("zarr.json")):
        ome = json.loads(zarr_json.read_text())["attributes"].get("ome", {})
        dtype = "uint8" if "image-label" in ome else "uint16"
        for ms in ome.get("multiscales", ()):
            ndim = len(ms["coordinateSystems"][0]["axes"])
            for ds in ms["datasets"]:
                _write_array(zarr_json.parent / ds["path"], ndim, dtype)
    return root


def edit(subpath: str, fn: Callable[[dict], Any]) -> Callable[[Path], None]:
    """Return a mutator that edits the zarr.json at `subpath` in place."""

    def _edit(root: Path) -> None:
        zarr_json = root / subpath / "zarr.json"
        data = json.loads(zarr_json.read_text())
        fn(data)
        zarr_json.write_text(json.dumps(data))

    return _edit


def _ms(data: dict) -> dict:
    return data["attributes"]["ome"]["multiscales"][0]


def _label_ref(data: dict) -> dict:
    return _ms(data)["coordinateTransformations"][0]["output"]


def _drop_label_channel_axis(root: Path) -> None:
    # a valid 3D label image, referenced from a 4D image by an identity
    def _drop(data: dict) -> None:
        ms = _ms(data)
        ms["coordinateSystems"][0]["axes"].pop(0)
        for ds in ms["datasets"]:
            t = ds["coordinateTransformations"][0]
            for step in t.get("transformations", [t]):  # (scale, or a sequence)
                step.get("scale", step.get("translation")).pop(0)
            _write_array(root / "labels" / "nuclei" / ds["path"], 3, "uint8")

    edit("labels/nuclei", _drop)(root)


def _rename_labels_group(root: Path) -> None:
    # a labels group not called "labels", found via the transformation's `path`
    (root / "labels").rename(root / "segmentations")
    edit("", lambda d: _label_ref(d).update(path="segmentations/nuclei"))(root)


def _copy_label_outside_labels_group(root: Path) -> None:
    # an image with the referenced coordinate system, but not in a labels group
    shutil.copytree(root / "labels" / "nuclei", root / "other" / "nuclei")
    (root / "other" / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    edit("", lambda d: _label_ref(d).update(path="other/nuclei"))(root)


@pytest.mark.parametrize("name", STORES)
def test_valid_stores(tmp_path: Path, name: str) -> None:
    validate_zarr_store(make_store(tmp_path, name))


def test_labels_group_found_via_transformation(tmp_path: Path) -> None:
    root = make_store(tmp_path, "image_with_labels")
    _rename_labels_group(root)
    validate_zarr_store(root)
    # the whole labels group is validated, not just the referenced label image
    edit("segmentations/cells/s0", lambda d: d.update(data_type="float32"))(root)
    with pytest.raises(StorageValidationError, match="label_non_integer_dtype"):
        validate_zarr_store(root)


def test_dimension_names_not_checked(tmp_path: Path) -> None:
    # (0.6 has no rule about them; 0.5.2 required them to match the axes names)
    root = make_store(tmp_path, "image_2d")
    edit("scale0/image", lambda d: d.update(dimension_names=["a", "b"]))(root)
    validate_zarr_store(root)


INVALID = [
    (
        "image_2d",
        lambda root: shutil.rmtree(root / "scale1"),
        StorageErrorType.dataset_path_not_found,
    ),
    (
        "image_2d",
        edit("scale1/image", lambda d: d.update(shape=[8, 8, 8])),
        StorageErrorType.dataset_dimension_mismatch,
    ),
    # Every Zarr array referred to by a `path` MUST have the same ... datatype
    (
        "image_2d",
        edit("scale1/image", lambda d: d.update(data_type="float32")),
        StorageErrorType.dataset_dtype_mismatch,
    ),
    (
        "image_with_labels",
        edit("labels/cells/s0", lambda d: d.update(data_type="float32")),
        StorageErrorType.label_non_integer_dtype,
    ),
    # transformations to a coordinate system in a child labels group
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(path="labels/missing")),
        StorageErrorType.coordinate_system_path_not_found,
    ),
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(path="s0")),
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(path="labels")),
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "image_with_labels",
        _copy_label_outside_labels_group,
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(path="labels/nuclei/s0")),
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(name="missing")),
        StorageErrorType.coordinate_system_not_found,
    ),
    (
        "image_with_labels",
        _drop_label_channel_axis,
        StorageErrorType.transform_dimension_mismatch,
    ),
    (
        "plate",
        lambda root: shutil.rmtree(root / "B" / "2"),
        StorageErrorType.well_path_not_found,
    ),
    (
        "plate",
        lambda root: shutil.rmtree(root / "A" / "1" / "1"),
        StorageErrorType.field_path_not_found,
    ),
    (
        "bf2raw",
        lambda root: shutil.rmtree(root / "1"),
        StorageErrorType.series_path_not_found,
    ),
]


@pytest.mark.parametrize(
    "name, mutate, err_type", INVALID, ids=[str(c[2]) for c in INVALID]
)
def test_invalid_stores(
    tmp_path: Path, name: str, mutate: Callable[[Path], None], err_type: Any
) -> None:
    root = make_store(tmp_path, name)
    mutate(root)
    with pytest.raises(StorageValidationError, match=str(err_type)) as exc:
        validate_zarr_store(root)
    assert {e["type"] for e in exc.value.errors()} == {str(err_type)}
