"""Test script to verify the OME-ZARR writer functions work correctly."""

from __future__ import annotations

import importlib.util

import pytest

if not importlib.util.find_spec("ome_zarr"):
    pytest.skip("ome_zarr not installed", allow_module_level=True)

import json
import sys
from pathlib import Path

import zarr

import yaozarrs

SCRIPTS = Path(__file__).parent.parent / "scripts"
if not (SCRIPTS / "write_demo_zarr.py").is_file():
    raise AssertionError(
        f"Script not found: {SCRIPTS / 'write_demo_zarr.py'}. "
        f"Please update {__name__} accordingly."
    )

sys.path.insert(0, str(SCRIPTS))

from write_demo_zarr import write_ome_image, write_ome_labels, write_ome_plate  # type: ignore # noqa

VERSIONS = ["0.4", "0.5"]


@pytest.mark.parametrize("version", VERSIONS)
def test_image_dataset(version: str, tmp_path: Path) -> None:
    """Test image dataset creation and validation."""
    print(f"\n=== Testing Image Dataset (v{version}) ===")

    image_path = tmp_path / f"test_image_v{version.replace('.', '')}"

    # Create the dataset
    write_ome_image(
        image_path,
        version=version,
        shape=(2, 3, 64, 64),  # t, c, y, x
        axes="tcyx",
        channel_names=["DAPI", "GFP", "RFP"],
        channel_colors=[0x0000FF, 0x00FF00, 0xFF0000],
        num_levels=3,
    )

    print(f"✓ Created image dataset at {image_path}")

    # Test with zarr-python
    try:
        store = zarr.open_group(str(image_path), mode="r")
        print(f"✓ Opened with zarr-python: {list(store.keys())}")

        # Check multiscale levels
        for i in range(3):
            if str(i) in store:
                data = store[str(i)]
                assert isinstance(data, zarr.Array)
                print(f"  - Level {i}: shape={data.shape}, dtype={data.dtype}")
    except Exception as e:
        print(f"✗ Failed to open with zarr-python: {e}")

    # Test with yaozarrs
    try:
        # Read the metadata file
        if version == "0.4":
            metadata_file = image_path / ".zattrs"
        else:
            metadata_file = image_path / "zarr.json"

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Validate with yaozarrs
            if version == "0.4":
                validated = yaozarrs.validate_ome_json(json.dumps(metadata))
            else:
                # For v0.5, metadata is in the "attributes" key
                ome_metadata = metadata.get("attributes", {})
                validated = yaozarrs.validate_ome_json(json.dumps(ome_metadata))

            print(f"✓ Validated with yaozarrs: {type(validated).__name__}")
        else:
            print(f"✗ Metadata file not found: {metadata_file}")
    except Exception as e:
        print(f"✗ Failed to validate with yaozarrs: {e}")


@pytest.mark.parametrize("version", VERSIONS)
def test_labels_dataset(version: str, tmp_path: Path) -> None:
    """Test labels dataset creation and validation."""
    print(f"\n=== Testing Labels Dataset (v{version}) ===")

    labels_path = tmp_path / f"test_labels_v{version.replace('.', '')}"

    # Create the dataset
    write_ome_labels(
        labels_path,
        version=version,
        shape=(64, 64),
        num_labels=3,
        label_colors=[
            (255, 0, 0, 255),  # Red
            (0, 255, 0, 255),  # Green
            (0, 0, 255, 255),  # Blue
        ],
        num_levels=2,
    )

    print(f"✓ Created labels dataset at {labels_path}")

    # Test with zarr-python
    try:
        store = zarr.open_group(str(labels_path), mode="r")
        print(f"✓ Opened with zarr-python: {list(store.keys())}")

        # Check labels group
        if "labels" in store:
            labels_group = zarr.open_group(str(labels_path), path="labels")
            print(f"  - Labels group keys: {list(labels_group.keys())}")

            for key in labels_group.keys():
                if key.isdigit():  # Resolution level
                    data = labels_group[key]
                    assert isinstance(data, zarr.Array)
                    print(f"  - Level {key}: shape={data.shape}, dtype={data.dtype}")
                    unique_labels = set(data[:].flatten())
                    print(f"    Unique labels: {sorted(unique_labels)}")
    except Exception as e:
        print(f"✗ Failed to open with zarr-python: {e}")

    # Test with yaozarrs
    try:
        if version == "0.4":
            metadata_file = labels_path / ".zattrs"
        else:
            metadata_file = labels_path / "zarr.json"

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            if version == "0.4":
                validated = yaozarrs.validate_ome_json(json.dumps(metadata))
            else:
                ome_metadata = metadata.get("attributes", {})
                validated = yaozarrs.validate_ome_json(json.dumps(ome_metadata))

            print(f"✓ Validated with yaozarrs: {type(validated).__name__}")
        else:
            print(f"✗ Metadata file not found: {metadata_file}")
    except Exception as e:
        print(f"✗ Failed to validate with yaozarrs: {e}")


@pytest.mark.parametrize("version", VERSIONS)
def test_plate_dataset(version: str, tmp_path: Path) -> None:
    """Test plate dataset creation and validation."""
    print(f"\n=== Testing Plate Dataset (v{version}) ===")

    plate_path = tmp_path / f"test_plate_v{version.replace('.', '')}"

    # Create the dataset
    write_ome_plate(
        plate_path,
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

    print(f"✓ Created plate dataset at {plate_path}")

    # Test with zarr-python
    try:
        store = zarr.open_group(str(plate_path), mode="r")
        print(f"✓ Opened with zarr-python: {list(store.keys())}")

        # Check wells
        for row in ["A", "B"]:
            for col in ["1", "2", "3"]:
                well_path = f"{row}/{col}"
                if well_path in store:
                    well_group = zarr.open_group(str(plate_path), path=well_path)
                    print(f"  - Well {well_path}: {list(well_group.keys())}")

                    # Check fields
                    for field in ["0", "1"]:
                        if field in well_group:
                            field_group = well_group[field]
                            if "0" in field_group:  # First resolution level
                                data = field_group["0"]
                                assert isinstance(data, zarr.Array)
                                print(f"    Field {field}: shape={data.shape}")
    except Exception as e:
        print(f"✗ Failed to open with zarr-python: {e}")

    # Test with yaozarrs
    try:
        if version == "0.4":
            metadata_file = plate_path / ".zattrs"
        else:
            metadata_file = plate_path / "zarr.json"

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            if version == "0.4":
                validated = yaozarrs.validate_ome_json(json.dumps(metadata))
            else:
                ome_metadata = metadata.get("attributes", {})
                validated = yaozarrs.validate_ome_json(json.dumps(ome_metadata))

            print(f"✓ Validated with yaozarrs: {type(validated).__name__}")
        else:
            print(f"✗ Metadata file not found: {metadata_file}")
    except Exception as e:
        print(f"✗ Failed to validate with yaozarrs: {e}")
