"""Test script to verify the OME-ZARR writer functions work correctly."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Callable, Literal, TypeAlias

import pytest

import yaozarrs

if TYPE_CHECKING:
    import zarr

    VersionStr: TypeAlias = Literal["0.4", "0.5"]

VERSIONS = []
with suppress(ImportError):
    import zarr

    # Only test v0.5 if zarr v3 is available
    zarr_major_version = int(zarr.__version__.split(".")[0])
    VERSIONS = ["0.4"]
    if zarr_major_version >= 3:
        VERSIONS.append("0.5")

ZATTRS = {"0.4": ".zattrs", "0.5": "zarr.json"}


@pytest.mark.parametrize("version", VERSIONS)
def test_image_dataset(version: VersionStr, write_demo_ome: Callable) -> None:
    """Test image dataset creation and validation."""

    # Create the dataset
    image_path = write_demo_ome(
        "image",
        version=version,
        shape=(2, 3, 64, 64),  # t, c, y, x
        axes="tcyx",
        channel_names=["DAPI", "GFP", "RFP"],
        channel_colors=[0x0000FF, 0x00FF00, 0xFF0000],
        num_levels=3,
    )

    # Check multiscale levels
    for i in range(3):
        assert zarr.open_array(str(image_path), mode="r", path=str(i))

    metadata_file = image_path / ZATTRS[version]
    with open(metadata_file) as f:
        metadata = json.load(f)

    yaozarrs.validate_ome_json(json.dumps(metadata))


@pytest.mark.parametrize("version", VERSIONS)
def test_labels_dataset(version: VersionStr, write_demo_ome: Callable) -> None:
    """Test labels dataset creation and validation."""
    # Create the dataset
    labels_path = write_demo_ome(
        "labels",
        version=version,
        labels_name="annotations",
        shape=(64, 64),
        num_labels=3,
        label_colors=[
            (255, 0, 0, 255),  # Red
            (0, 255, 0, 255),  # Green
            (0, 0, 255, 255),  # Blue
        ],
        num_levels=2,
    )

    # Check labels group
    labels_group = zarr.open_group(str(labels_path), mode="r")
    annotations_group = labels_group["annotations"]
    assert isinstance(labels_group, zarr.Group)
    assert isinstance(annotations_group, zarr.Group)
    for key in ["0", "1"]:  # Label arrays
        assert isinstance(annotations_group[key], zarr.Array)

    metadata_file = labels_path / ZATTRS[version]
    with open(metadata_file) as f:
        metadata = json.load(f)

    yaozarrs.validate_ome_json(json.dumps(metadata))
    # yaozarrs.from_uri(labels_path / ZATTRS[version])


@pytest.mark.parametrize("version", VERSIONS)
def test_plate_dataset(version: VersionStr, write_demo_ome: Callable) -> None:
    """Test plate dataset creation and validation."""

    # Create the dataset
    plate_path = write_demo_ome(
        "plate",
        version=version,
        plate_name="test-plate",
        rows=["A", "B"],
        columns=["1", "2", "3"],
        image_shape=(2, 32, 32),  # c, y, x
        image_axes="cyx",
        fields_per_well=2,
        acquisitions=[
            {"id": 0, "name": "acq1"},
            {"id": 1, "name": "acq2"},
        ],
        num_levels=2,
    )

    # Check wells
    for row in ["A", "B"]:
        for col in ["1", "2", "3"]:
            well_path = f"{row}/{col}"
            well_group = zarr.open_group(str(plate_path), path=well_path)

            # Check fields
            for field in ["0", "1"]:
                field_group = well_group[field]
                assert isinstance(field_group, zarr.Group)
                assert isinstance(field_group["0"], zarr.Array)

    yaozarrs.validate_ome_uri(plate_path / ZATTRS[version])
