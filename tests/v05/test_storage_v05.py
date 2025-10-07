import json
import shutil
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, cast

import pytest

try:
    import zarr
except ImportError:
    pytest.skip("zarr not installed", allow_module_level=True)
try:
    import fsspec  # noqa: F401
except ImportError:
    pytest.skip("fsspec not installed", allow_module_level=True)

try:
    from aiohttp.client_exceptions import ClientConnectorError

    connection_exceptions: tuple[type[Exception], ...] = (ClientConnectorError,)
except ImportError:
    connection_exceptions = ()

from yaozarrs.v05._storage import (
    ErrorDetails,
    StorageErrorType,
    StorageValidationError,
    validate_zarr_store,
)


def test_validate_missing_zarr_file() -> None:
    """Test validation with non-existent zarr file."""
    # Use a path that doesn't trigger filesystem creation attempts
    # ValueError in zarr2, FileNotFoundError in zarr3
    with pytest.raises((FileNotFoundError, ValueError)):
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


URIS: list[str] = []
if version("zarr").startswith("3"):
    URIS += [
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0010/76-45.ome.zarr",
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0026/3.66.9-6.141020_15-41-29.00.ome.zarr",
    ]
    for local in [
        "~/Downloads/zarr/6001240_labels.zarr",
        "~/Downloads/zarr/76-45.ome.zarr",
        "~/Downloads/zarr/3.66.9-6.141020_15-41-29.00.ome.zarr",
    ]:
        if Path(local).expanduser().exists():
            URIS.append(local)


@pytest.mark.parametrize("uri", URIS)
def test_validate_storage(uri: str) -> None:
    """Test basic validation functionality."""
    # Test with real zarr file that should pass validation
    try:
        validate_zarr_store(uri)
    except connection_exceptions:
        pytest.xfail("No internet")


@pytest.mark.parametrize("type", ["image", "labels", "plate", "image-with-labels"])
def test_validate_valid_demo_storage(type: str, write_demo_ome: Callable) -> None:
    """Test validation on demo OME-Zarr files."""

    path = write_demo_ome(type, version="0.5")
    try:
        validate_zarr_store(path)
    except connection_exceptions:
        pytest.xfail("No internet")


def test_validate_null_storage(tmp_path: Path) -> None:
    """Test validation with intentionally broken storage."""
    with pytest.raises(OSError, match="No zarr metadata found"):
        validate_zarr_store(tmp_path.name)


@dataclass
class StorageTestCase:
    err_type: StorageErrorType
    kwargs: dict
    mutator: Callable[[Path], Any]


def update_meta(subpath: str | tuple[str, ...], key: tuple[str, ...], value: Any):
    def _arr2group(path: Path):
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

    return _arr2group


MULTI_SCALE = {
    "name": "timelapse",
    "axes": [{"name": "x", "type": "space"}, {"name": "y", "type": "space"}],
    "datasets": [
        {
            "path": "0",
            "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}],
        },
    ],
}
MULTI_SCALE2 = {
    "name": "timelapse2",
    "axes": [{"name": "x", "type": "space"}, {"name": "y", "type": "space"}],
    "datasets": [
        {
            "path": "1",
            "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}],
        },
    ],
}

IMAGE_META = {"version": "0.5", "multiscales": [MULTI_SCALE]}

# bf2raw_no_images = auto()
# bf2raw_path_not_found = auto()
# bf2raw_path_not_group = auto()
# bf2raw_invalid_image = auto()
# series_path_not_found = auto()
# series_path_not_group = auto()
# series_invalid_image = auto()
# invalid_labels_metadata = auto()


@pytest.mark.parametrize(
    "case",
    [
        StorageTestCase(
            StorageErrorType.dataset_path_not_found,
            {"type": "image"},
            lambda p: shutil.rmtree(p / "0"),
        ),
        StorageTestCase(
            StorageErrorType.dataset_not_array,
            {"type": "image"},
            update_meta("0", ("node_type",), "group"),
        ),
        StorageTestCase(
            StorageErrorType.dataset_dimension_mismatch,
            {"type": "image"},
            update_meta("0", ("shape",), [10] * 5),
        ),
        StorageTestCase(
            StorageErrorType.dimension_names_mismatch,
            {"type": "image"},
            update_meta("0", ("attributes",), {"dimension_names": ["a", "b", "c"]}),
        ),
        StorageTestCase(
            StorageErrorType.well_path_not_found,
            {"type": "plate"},
            lambda p: shutil.rmtree(p / "A" / "1"),
        ),
        StorageTestCase(
            StorageErrorType.well_path_not_group,
            {"type": "plate"},
            update_meta(("A", "1"), ("node_type",), "array"),
        ),
        StorageTestCase(
            StorageErrorType.invalid_well,
            {"type": "plate"},
            update_meta(("A", "1"), ("attributes", "ome"), IMAGE_META),
        ),
        StorageTestCase(
            StorageErrorType.invalid_well,
            {"type": "plate"},
            update_meta(("A", "1"), ("attributes", "ome"), {}),
        ),
        StorageTestCase(
            StorageErrorType.field_path_not_found,
            {"type": "plate"},
            lambda p: shutil.rmtree(p / "A" / "1" / "0"),
        ),
        StorageTestCase(
            StorageErrorType.field_path_not_group,
            {"type": "plate"},
            update_meta(("A", "1", "0"), ("node_type",), "array"),
        ),
        StorageTestCase(
            StorageErrorType.label_path_not_found,
            {"type": "labels"},
            lambda p: shutil.rmtree(p / "annotations"),
        ),
        StorageTestCase(
            StorageErrorType.label_path_not_group,
            {"type": "labels"},
            update_meta("annotations", ("node_type",), "array"),
        ),
        StorageTestCase(
            StorageErrorType.invalid_label_image,
            {"type": "labels"},
            update_meta("annotations", ("attributes", "ome"), IMAGE_META),
        ),
        StorageTestCase(
            StorageErrorType.invalid_label_image,
            {"type": "labels"},
            update_meta("annotations", ("attributes", "ome"), {}),
        ),
        StorageTestCase(
            StorageErrorType.label_multiscale_count_mismatch,
            {"type": "image-with-labels"},
            update_meta(
                ("labels", "annotations"),
                ("attributes", "ome", "multiscales"),
                [MULTI_SCALE, MULTI_SCALE2],
            ),
        ),
        StorageTestCase(
            StorageErrorType.label_dataset_count_mismatch,
            {"type": "image-with-labels"},
            update_meta(
                ("labels", "annotations"),
                ("attributes", "ome", "multiscales"),
                [MULTI_SCALE],
            ),
        ),
        StorageTestCase(
            StorageErrorType.label_non_integer_dtype,
            {"type": "image-with-labels"},
            update_meta(("labels", "annotations", "0"), ("data_type",), "float32"),
        ),
        StorageTestCase(
            StorageErrorType.labels_not_group,
            {"type": "image-with-labels"},
            update_meta("labels", ("node_type",), "array"),
        ),
        StorageTestCase(
            StorageErrorType.invalid_labels_metadata,
            {"type": "image-with-labels"},
            update_meta("labels", ("attributes", "ome", "multiscales"), []),
        ),
        # StorageTestCase(
        #     StorageErrorType.invalid_label_image_source,
        #     {"type": "image-with-labels"},
        #     lambda x: None,
        # ),
        # StorageTestCase(
        #     StorageErrorType.label_image_source_not_found,
        #     {"type": "image-with-labels"},
        #     lambda x: None,
        # ),
    ],
    ids=lambda x: x.err_type,
)
def test_validate_invalid_storage(
    case: StorageTestCase, write_demo_ome: Callable
) -> None:
    path = write_demo_ome(**case.kwargs)
    case.mutator(path)
    try:
        with pytest.raises(StorageValidationError, match=str(case.err_type)):
            validate_zarr_store(path)
    except connection_exceptions:
        pytest.xfail("No internet")
