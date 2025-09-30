from pathlib import Path
from typing import Callable, cast

import pytest

try:
    import zarr
except ImportError:
    pytest.skip("zarr not installed", allow_module_level=True)

from yaozarrs.v05._storage import (
    ErrorDetails,
    StorageValidationError,
    validate_zarr_store,
)


def test_validate_invalid_storage(tmp_path: Path) -> None:
    """Test validation with intentionally broken storage."""
    with pytest.raises((ValueError, KeyError, OSError)):
        validate_zarr_store(tmp_path.name)


def test_validate_missing_zarr_file() -> None:
    """Test validation with non-existent zarr file."""
    # Use a path that doesn't trigger filesystem creation attempts
    with pytest.raises(FileNotFoundError):
        validate_zarr_store("./nonexistent_zarr_directory")


def test_validation_error_cases(tmp_path: Path) -> None:
    """Test various validation error scenarios."""

    # Create a zarr group with invalid OME metadata
    root = zarr.group(tmp_path / "invalid_group")

    # Test case 1: Group with missing datasets
    invalid_metadata = {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": [
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {
                            "path": "nonexistent_array",  # This array doesn't exist
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0]}
                            ],
                        }
                    ],
                }
            ],
        }
    }

    root.attrs.update(invalid_metadata)

    # This should raise StorageValidationError due to missing dataset
    with pytest.raises(StorageValidationError) as exc_info:
        validate_zarr_store(root)

    # Check that error details are properly formatted
    errors = cast("StorageValidationError", exc_info.value).errors()
    assert len(errors) >= 1
    assert any("not found" in error["msg"].lower() for error in errors)


def test_storage_validation_error() -> None:
    """Test the StorageValidationError class."""

    # Test error creation and formatting
    errors: list[ErrorDetails] = [
        {
            "type": "test_error",
            "loc": ("ome", "multiscales", 0),
            "msg": "Test error message",
            "input": "test_input",
        },
        {
            "type": "another_error",
            "loc": ("ome", "datasets", 1, "path"),
            "msg": "Another error message",
            "input": "another_input",
        },
    ]

    error = StorageValidationError(errors)

    # Test that error message is generated
    assert "2 validation error(s) for storage structure:" in str(error)
    assert "Test error message" in str(error)
    assert "Another error message" in str(error)

    # Test errors() method
    filtered_errors = error.errors()
    assert len(filtered_errors) == 2
    assert filtered_errors[0]["type"] == "test_error"
    assert filtered_errors[1]["type"] == "another_error"

    # Test filtering options
    no_input_errors = error.errors(include_input=False)
    assert "input" not in no_input_errors[0]

    # Test title property
    assert error.title == "StorageValidationError"


@pytest.mark.parametrize(
    "uri",
    [
        # "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr",
        "/Users/talley/Downloads/zarr/6001240_labels.zarr",
        # "/Users/talley/Downloads/zarr/3.66.9-6.141020_15-41-29.00.ome.zarr",
        # "/Users/talley/Downloads/zarr/76-45.ome.zarr",
    ],
)
def test_validate_storage(uri: str) -> None:
    """Test basic validation functionality."""
    # Test with real zarr file that should pass validation
    validate_zarr_store(uri)


@pytest.mark.parametrize("type", ["image", "labels", "plate"])
def test_validate_demo_storage(type: str, write_demo_ome: Callable) -> None:
    """Test validation on demo OME-Zarr files."""
    path = write_demo_ome(type, version="0.5")
    validate_zarr_store(path)
