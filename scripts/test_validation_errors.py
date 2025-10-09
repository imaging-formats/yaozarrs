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

from yaozarrs import validate_zarr_store
from yaozarrs._demo_data import write_ome_plate


def update_zarr_metadata(
    path: Path, subpath: str | tuple[str, ...], key: tuple[str, ...], value: any
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
        # Create a valid plate dataset with 2x2 wells
        write_ome_plate(
            plate_path,
            version="0.5",
            rows=["A", "B"],
            columns=["1", "2"],
            fields_per_well=2,
        )

        print("\nIntroducing validation errors...")

        # Error 1: Remove a dataset (field image) - dataset_path_not_found
        print("  1. Removing field dataset A/1/0/0")
        shutil.rmtree(plate_path / "A" / "1" / "0" / "0")

        # Error 2: Make a well into an array instead of group - well_path_not_group
        print("  2. Converting well A/2 to array type")
        update_zarr_metadata(plate_path, ("A", "2"), ("node_type",), "array")

        # Error 3: Remove a field entirely - field_path_not_found
        print("  3. Removing entire field B/1/1")
        shutil.rmtree(plate_path / "B" / "1" / "1")

        # Error 4: Make a field into invalid group - field_image_invalid
        print("  4. Making field B/2/0 invalid by removing multiscales")
        update_zarr_metadata(
            plate_path,
            ("B", "2", "0"),
            ("attributes", "ome", "multiscales"),
            [],
        )

        # Error 5: Wrong dimension count for dataset
        print("  5. Changing dataset dimensions for B/2/1/0")
        update_zarr_metadata(
            plate_path,
            ("B", "2", "1", "0"),
            ("shape",),
            [10, 10, 10, 10, 10],  # Wrong number of dimensions
        )

        print("\n" + "=" * 80)
        print("Running validation...")
        print("=" * 80 + "\n")

        # Run validation and catch the error
        validate_zarr_store(plate_path)


if __name__ == "__main__":
    main()
