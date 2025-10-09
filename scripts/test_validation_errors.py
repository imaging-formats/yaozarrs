#!/usr/bin/env python
"""Script to test validation with multiple error types in a single dataset.

Creates a temporary Plate dataset and introduces multiple validation errors
to test the validate_zarr_store function and CLI.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from yaozarrs import validate_zarr_store
from yaozarrs._demo_data import write_ome_labels, write_ome_plate


def update_zarr_metadata(
    path: Path, subpath: str | tuple[str, ...], key: tuple[str, ...], value: Any
) -> None:
    """Update zarr.json metadata at a given path.

    Parameters
    ----------
    path : Path
        Root path of the zarr store
    subpath : str | tuple[str, ...]
        Subpath to the zarr.json file
    key : tuple[str, ...]
        Nested key path to update
    value : any
        Value to set
    """
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


def main() -> None:
    """Create a plate dataset with multiple validation errors."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        plate_path = tmp_path / "test_plate.zarr"

        print(f"Creating plate dataset at: {plate_path}")
        # Create a valid plate dataset with 3x3 wells
        write_ome_plate(
            plate_path,
            version="0.5",
            rows=["A", "B", "C"],
            columns=["1", "2", "3"],
            fields_per_well=2,
        )

        # Add labels to some fields manually
        print("Adding labels to some fields...")
        for well_row, well_col, field_idx in [
            ("A", "1", "0"),
            ("A", "1", "1"),
            ("A", "3", "0"),
            ("B", "1", "0"),
            ("B", "3", "0"),
            ("C", "1", "0"),
        ]:
            field_path = plate_path / well_row / well_col / field_idx
            labels_path = field_path / "labels" / "annotations"
            write_ome_labels(
                labels_path,
                version="0.5",
                parent_image_path="../..",
                shape=(64, 64),
                axes="yx",
                num_levels=2,
            )

        print("\nIntroducing validation errors...")

        # Error 1: Remove a dataset (field image) - dataset_path_not_found
        print("  1. Removing field dataset A/1/0/0")
        shutil.rmtree(plate_path / "A" / "1" / "0" / "0")

        # Error 2: Make a well into an array instead of group - well_path_not_group
        print("  2. Converting well A/2 to array type")
        update_zarr_metadata(plate_path, ("A", "2"), ("node_type",), "array")

        # Error 3: Wrong dimension count for dataset - dataset_dimension_mismatch
        print("  3. Changing dataset dimensions for A/2/0/0")
        update_zarr_metadata(
            plate_path,
            ("A", "2", "0", "0"),
            ("shape",),
            [10, 10, 10, 10, 10],  # Wrong number of dimensions
        )

        # Error 4: Make dataset into group instead of array - dataset_not_array
        print("  4. Converting dataset A/2/1/0 to group type")
        update_zarr_metadata(
            plate_path,
            ("A", "2", "1", "0"),
            ("node_type",),
            "group",
        )

        # Error 5: Wrong dimension_names in array attributes - dimension_names_mismatch
        print("  5. Setting wrong dimension_names for A/3/0/0")
        update_zarr_metadata(
            plate_path,
            ("A", "3", "0", "0"),
            ("attributes",),
            {"dimension_names": ["wrong", "names", "here"]},
        )

        # Error 6: Make a field path into array - field_path_not_group
        print("  6. Converting field A/3/1 to array type")
        update_zarr_metadata(
            plate_path,
            ("A", "3", "1"),
            ("node_type",),
            "array",
        )

        # Error 7: Remove a field entirely - field_path_not_found
        print("  7. Removing entire field B/1/1")
        shutil.rmtree(plate_path / "B" / "1" / "1")

        # Error 8: Make a field into invalid group - field_image_invalid
        print("  8. Making field B/2/0 invalid by removing multiscales")
        update_zarr_metadata(
            plate_path,
            ("B", "2", "0"),
            ("attributes", "ome", "multiscales"),
            [],
        )

        # Error 9: Invalid well metadata - well_invalid
        print("  9. Clearing well B/2 metadata")
        update_zarr_metadata(
            plate_path,
            ("B", "2"),
            ("attributes", "ome"),
            {},
        )

        # Error 10: Remove label path - label_path_not_found
        # Target the actual label image: field/labels/annotations/labels/annotations
        print(" 10. Removing label path A/1/0/labels/annotations/labels/annotations")
        shutil.rmtree(
            plate_path
            / "A"
            / "1"
            / "0"
            / "labels"
            / "annotations"
            / "labels"
            / "annotations"
        )

        # Error 11: Make label path into array - label_path_not_group
        # Target the actual label image group
        print(
            " 11. Converting A/1/1/labels/annotations/labels/annotations to array type"
        )
        update_zarr_metadata(
            plate_path,
            ("A", "1", "1", "labels", "annotations", "labels", "annotations"),
            ("node_type",),
            "array",
        )

        # Error 12: Invalid label image metadata - label_image_invalid
        # Target the actual label image group
        print(
            " 12. Clearing label image metadata for "
            "B/1/0/labels/annotations/labels/annotations"
        )
        update_zarr_metadata(
            plate_path,
            ("B", "1", "0", "labels", "annotations", "labels", "annotations"),
            ("attributes", "ome"),
            {},
        )

        # Error 13: Non-integer dtype for label - label_non_integer_dtype
        # Note: B/3/0 has labels based on our setup
        # The structure is field/labels/annotations/labels/annotations/0
        print(
            " 13. Setting float dtype for B/3/0/labels/annotations/labels/annotations/0"
        )
        update_zarr_metadata(
            plate_path,
            ("B", "3", "0", "labels", "annotations", "labels", "annotations", "0"),
            ("data_type",),
            "float32",
        )

        # Error 14: Make labels parent into array - labels_not_group
        # Note: C/1/0 has labels based on our setup
        # Target the inner labels group: field/labels/annotations/labels
        print(" 14. Converting C/1/0/labels/annotations/labels to array type")
        update_zarr_metadata(
            plate_path,
            ("C", "1", "0", "labels", "annotations", "labels"),
            ("node_type",),
            "array",
        )

        print("\n" + "=" * 80)
        print("Running validation...")
        print("=" * 80 + "\n")

        # Run validation and catch the error
        validate_zarr_store(plate_path)


if __name__ == "__main__":
    main()
