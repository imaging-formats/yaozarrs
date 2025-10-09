import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from yaozarrs._zarr import open_group

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

from yaozarrs import validate_zarr_store
from yaozarrs._storage import ErrorDetails, StorageErrorType, StorageValidationError


@contextmanager
def xfail_internet_error() -> Iterator[None]:
    """Decorator to xfail tests on internet connection errors."""
    try:
        yield
    except connection_exceptions:
        pytest.xfail("No internet")


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
            "ctx": {"fs_path": "test.zarr/0", "expected": "array"},
        },
        {
            "type": "another_error",
            "loc": ("ome", "datasets", 1, "path"),
            "msg": "Another error message",
            "ctx": {"fs_path": "test.zarr/1", "expected": "group", "found": "array"},
        },
    ]

    error = StorageValidationError(errors)

    # Test that error message is generated (now uses class title)
    assert "2 validation error(s) for StorageValidationError" in str(error)
    assert "Test error message" in str(error)
    assert "Another error message" in str(error)
    # Also verify the new dot-notation format is used
    assert "ome.multiscales.0" in str(error)
    assert "ome.datasets.1.path" in str(error)

    # Test errors() method
    filtered_errors = error.errors()
    assert len(filtered_errors) == 2
    assert filtered_errors[0]["type"] == "test_error"
    assert filtered_errors[1]["type"] == "another_error"

    # Test filtering options (context should be present)
    filtered = error.errors(include_context=True)
    assert "ctx" in filtered[0]

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
    with xfail_internet_error():
        validate_zarr_store(uri)


@pytest.mark.parametrize(
    "type",
    ["image", "labels", "plate", "image-with-labels", "bioformats2raw"],
)
def test_validate_valid_demo_storage(type: str, write_demo_ome: Callable) -> None:
    """Test validation on demo OME-Zarr files."""

    path = write_demo_ome(type, version="0.5")
    with xfail_internet_error():
        validate_zarr_store(path)

    if type == "image-with-labels":
        # Also validate with labels only
        validate_zarr_store(path / "labels")
        validate_zarr_store(path / "labels" / "annotations")
    elif type == "plate":
        # Validate a well only
        validate_zarr_store(path / "A" / "1")


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_group_from_group(write_demo_ome: Callable, version: str) -> None:
    """Test validation with intentionally broken storage."""
    path = write_demo_ome("image", version=version)
    group = open_group(path)
    assert open_group(group) is group


def test_validate_valid_demo_storage_bf2raw_ome_group(write_demo_ome: Callable) -> None:
    """Test validation on demo OME-Zarr files."""

    path = write_demo_ome("bioformats2raw", version="0.5", write_ome_group=True)
    with xfail_internet_error():
        validate_zarr_store(path)


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
            StorageErrorType.well_invalid,
            {"type": "plate"},
            update_meta(("A", "1"), ("attributes", "ome"), IMAGE_META),
        ),
        StorageTestCase(
            StorageErrorType.well_invalid,
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
            StorageErrorType.field_image_invalid,
            {"type": "plate"},
            update_meta(("A", "1", "0"), ("attributes", "ome"), {}),
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
            StorageErrorType.label_image_invalid,
            {"type": "labels"},
            update_meta("annotations", ("attributes", "ome"), IMAGE_META),
        ),
        StorageTestCase(
            StorageErrorType.label_image_invalid,
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
            StorageErrorType.labels_metadata_invalid,
            {"type": "image-with-labels"},
            update_meta("labels", ("attributes", "ome", "multiscales"), []),
        ),
        StorageTestCase(
            StorageErrorType.label_image_source_invalid,
            {"type": "labels"},
            lambda p: (
                # Create a source group with LabelsGroup metadata (not Image)
                zarr.group(
                    p / "source_dummy",
                    attributes={"ome": {"version": "0.5", "labels": ["foo"]}},
                ),
                # Point the label to this invalid source
                update_meta(
                    ("annotations",),
                    ("attributes", "ome", "image-label"),
                    {"source": {"image": "../source_dummy"}},
                )(p),
            )[1],
        ),
        StorageTestCase(
            StorageErrorType.label_image_source_not_found,
            {"type": "labels"},
            lambda p: update_meta(
                ("annotations",),
                ("attributes", "ome", "image-label"),
                {"source": {"image": "../nonexistent"}},
            )(p),
        ),
        StorageTestCase(
            StorageErrorType.bf2raw_no_images,
            {"type": "bioformats2raw"},
            lambda p: shutil.rmtree(p / "0"),
        ),
        StorageTestCase(
            StorageErrorType.bf2raw_path_not_group,
            {"type": "bioformats2raw"},
            update_meta("0", ("node_type",), "array"),
        ),
        StorageTestCase(
            StorageErrorType.bf2raw_invalid_image,
            {"type": "bioformats2raw"},
            update_meta("0", ("attributes", "ome", "multiscales"), []),
        ),
        StorageTestCase(
            StorageErrorType.series_path_not_found,
            {"type": "bioformats2raw", "write_ome_group": True},
            update_meta("OME", ("attributes", "ome", "series"), ["1"]),
        ),
        StorageTestCase(
            StorageErrorType.series_path_not_group,
            {"type": "bioformats2raw", "write_ome_group": True},
            update_meta("0", ("node_type",), "array"),
        ),
        StorageTestCase(
            StorageErrorType.series_invalid_image,
            {"type": "bioformats2raw", "write_ome_group": True},
            update_meta("0", ("attributes", "ome", "multiscales"), []),
        ),
    ],
    ids=lambda x: x.err_type,
)
def test_validate_invalid_storage(
    case: StorageTestCase, write_demo_ome: Callable
) -> None:
    path = write_demo_ome(**case.kwargs)
    case.mutator(path)
    with xfail_internet_error():
        with pytest.raises(StorageValidationError, match=str(case.err_type)):
            validate_zarr_store(path)


def test_validate_complex_ome_zarr(complex_ome_zarr: Path) -> None:
    """Test validation on demo OME-Zarr files."""

    with xfail_internet_error():
        validate_zarr_store(complex_ome_zarr)


def test_validate_complex_ome_zarr_broken(complex_ome_zarr_broken: Path) -> None:
    """Test validation on demo OME-Zarr files."""

    with pytest.raises(StorageValidationError) as err:
        validate_zarr_store(complex_ome_zarr_broken)

    err_types = {e["type"] for e in err.value.errors()}
    assert err_types == {
        str(StorageErrorType.dataset_path_not_found),
        str(StorageErrorType.well_path_not_group),
        str(StorageErrorType.dataset_dimension_mismatch),
        str(StorageErrorType.dataset_not_array),
        str(StorageErrorType.dimension_names_mismatch),
        str(StorageErrorType.field_path_not_group),
        str(StorageErrorType.field_path_not_found),
        str(StorageErrorType.field_image_invalid),
        str(StorageErrorType.well_invalid),
        str(StorageErrorType.label_path_not_found),
        str(StorageErrorType.label_path_not_group),
        str(StorageErrorType.label_image_invalid),
        str(StorageErrorType.label_non_integer_dtype),
        str(StorageErrorType.labels_not_group),
    }
    assert len(err.value.errors()) == 14

    print(err.value)
