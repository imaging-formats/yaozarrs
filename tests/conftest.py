"""Test script to verify the OME-ZARR writer functions work correctly."""

from __future__ import annotations

import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Literal

import pytest


def dd() -> Any:
    try:
        from yaozarrs import _demo_data
    except Exception:
        pytest.skip("ome-zarr not installed", allow_module_level=True)
    else:
        return _demo_data


@pytest.fixture
def write_demo_ome(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Path]:
    _dd = dd()

    def _write_demo(
        type: Literal[
            "image", "labels", "plate", "image-with-labels", "bioformats2raw"
        ],
        **kwargs: Any,
    ) -> Path:
        ome_version = kwargs.setdefault("version", "0.5")
        if ome_version == "0.5" and version("zarr").startswith("2"):
            pytest.skip("zarr v2 does not support OME-Zarr v0.5")

        path = tmp_path_factory.mktemp(f"demo_{type}.zarr")
        if type in ("image", "image-with-labels", "bioformats2raw"):
            if type == "bioformats2raw":
                _dd.write_ome_bf2raw(path, **kwargs)
            else:
                _dd.write_ome_image(path, **kwargs)
            if type == "image-with-labels":
                _dd.write_ome_labels(path, **kwargs)
        elif type == "labels":
            # XXX: write_labels is special (in ome-zarr) ...
            # it creates a group "labels"
            _dd.write_ome_labels(path, **kwargs)
            path = path / "labels"
        elif type == "plate":
            _dd.write_ome_plate(path, **kwargs)
        else:
            raise ValueError(f"Unknown type: {type}")
        return path

    return _write_demo


@pytest.fixture
def complex_ome_zarr(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if version("zarr").startswith("2"):
        pytest.skip("zarr v2 does not support OME-Zarr v0.5")

    _dd = dd()
    path = tmp_path_factory.mktemp("complex.ome.zarr")
    _dd.write_ome_plate(
        path, rows=list("ABC"), columns=list("123"), fields_per_well=2, version="0.5"
    )

    # Add labels to some fields using the proper API
    # This creates proper ome.labels metadata in the field's labels group
    for well_row, well_col, field_idx in ["A10", "A11", "A30", "B10", "B30", "C10"]:
        field_path = path / well_row / well_col / field_idx
        _dd.write_ome_labels(field_path, version="0.5")

    return path


def _update_zarr_metadata(
    path: Path, subpath: str | tuple[str, ...], key: tuple[str, ...], value: Any
) -> None:
    """Update zarr.json metadata at a given path."""
    if isinstance(subpath, tuple):
        zarr_json_path = Path(path, *subpath, "zarr.json")
    else:
        zarr_json_path = path / subpath / "zarr.json"

    data: dict = json.loads(zarr_json_path.read_text())
    *first, last = key
    d = data
    for k in first:
        d = d[k]
    d[last] = value
    zarr_json_path.write_text(json.dumps(data))


@pytest.fixture
def complex_ome_zarr_broken(complex_ome_zarr: Path) -> Path:
    plate_path = complex_ome_zarr

    # Error 1: Remove a dataset (field image) - dataset_path_not_found
    # Remove resolution level 0 from field A/1/0
    shutil.rmtree(plate_path / "A" / "1" / "0" / "0")

    # Error 2: Make a well into an array instead of group - well_path_not_group
    # Break well A/2 at the well level
    _update_zarr_metadata(plate_path, ("A", "2"), ("node_type",), "array")

    # Error 3: Wrong dimension count for dataset - dataset_dimension_mismatch
    # Use a different well (B/1) so it's discoverable
    _update_zarr_metadata(
        plate_path,
        ("B", "1", "0", "0"),
        ("shape",),
        [10, 10, 10, 10, 10],  # Wrong number of dimensions
    )

    # Error 4: Make dataset into group instead of array - dataset_not_array
    # Use well B/2, field 0, dataset 0
    _update_zarr_metadata(plate_path, ("B", "2", "0", "0"), ("node_type",), "group")

    # Error 5: Wrong dimension_names in array attributes - dimension_names_mismatch
    # Use well B/2, field 1 to keep it separate from error 4
    _update_zarr_metadata(
        plate_path,
        ("B", "2", "1", "0"),
        ("attributes",),
        {"dimension_names": ["wrong", "names", "here"]},
    )

    # Error 6: Make a field path into array - field_path_not_group
    # Use well B/3, field 1
    _update_zarr_metadata(plate_path, ("B", "3", "1"), ("node_type",), "array")

    # Error 7: Remove a field entirely - field_path_not_found
    # Use well C/1, field 1
    shutil.rmtree(plate_path / "C" / "1" / "1")

    # Error 8: Make a field into invalid group - field_image_invalid
    # Use well C/2, field 0 - empty multiscales
    _update_zarr_metadata(
        plate_path, ("C", "2", "0"), ("attributes", "ome", "multiscales"), []
    )

    # Error 9: Invalid well metadata - well_invalid
    # Use well C/3 - corrupt well metadata
    _update_zarr_metadata(plate_path, ("C", "3"), ("attributes", "ome"), {})

    # Error 10: Remove label path - label_path_not_found
    # Remove the "annotations" label from A/1/0 (which has labels)
    # The labels group references this path, so it will fail validation
    shutil.rmtree(plate_path / "A" / "1" / "0" / "labels" / "annotations")

    # Error 11: Make label path into array - label_path_not_group
    # Convert A/1/1's annotations label to an array instead of a group
    _update_zarr_metadata(
        plate_path, ("A", "1", "1", "labels", "annotations"), ("node_type",), "array"
    )

    # Error 12: Invalid label image metadata - label_image_invalid
    # Clear the label image metadata for A/3/0's annotations
    _update_zarr_metadata(
        plate_path, ("A", "3", "0", "labels", "annotations"), ("attributes", "ome"), {}
    )

    # Error 13: Non-integer dtype for label - label_non_integer_dtype
    # Set float dtype for B/1/0's label array (resolution level 0)
    _update_zarr_metadata(
        plate_path,
        ("B", "1", "0", "labels", "annotations", "0"),
        ("data_type",),
        "float32",
    )

    # Error 14: Make labels parent into array - labels_not_group
    # Convert B/3/0's labels group to an array
    _update_zarr_metadata(
        plate_path, ("B", "3", "0", "labels"), ("node_type",), "array"
    )

    return complex_ome_zarr
