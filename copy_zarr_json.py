#!/usr/bin/env python3
"""
Script to mirror directory structure and copy only zarr.json files.

This script copies the directory structure from the source to destination,
but only copies files named 'zarr.json'. Empty directories are not created.
"""

import os
import shutil
from pathlib import Path


def copy_zarr_json_files(source_dir: str, dest_dir: str) -> None:
    """
    Mirror directory structure and copy only zarr.json files.

    Args:
        source_dir: Source directory to copy from
        dest_dir: Destination directory to copy to
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Track directories that contain zarr.json files
    copied_files = []

    # Walk through source directory
    for root, dirs, files in os.walk(source_path):
        # Check if this directory contains zarr.json
        if "zarr.json" in files:
            # Calculate relative path from source
            rel_path = Path(root).relative_to(source_path)

            # Create destination directory path
            dest_subdir = dest_path / rel_path

            # Create directory if it doesn't exist
            dest_subdir.mkdir(parents=True, exist_ok=True)

            # Copy zarr.json file
            source_file = Path(root) / "zarr.json"
            dest_file = dest_subdir / "zarr.json"

            shutil.copy2(source_file, dest_file)
            copied_files.append(str(dest_file))
            print(f"Copied: {rel_path}/zarr.json")

    print(f"\nTotal files copied: {len(copied_files)}")
    return copied_files


def main():
    source_dir = "/Users/talley/HMS Dropbox/Talley Lambert/Shared/Data/ngff_examples/v0.5/"
    dest_dir = "/Users/talley/dev/self/yaozarrs/tests/data/"

    # Check if source directory exists
    if not Path(source_dir).exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return

    # Create destination directory if it doesn't exist
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    print(f"Copying zarr.json files from:")
    print(f"  Source: {source_dir}")
    print(f"  Dest:   {dest_dir}")
    print()

    try:
        copied_files = copy_zarr_json_files(source_dir, dest_dir)
        print("\nCopy completed successfully!")

    except Exception as e:
        print(f"Error during copy: {e}")


if __name__ == "__main__":
    main()