import json
import re
import shutil
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from yaozarrs._zarr import _CachedMapper, open_group

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

from yaozarrs import bioformats2raw_series, v04, v05, validate_zarr_store
from yaozarrs._errors import (
    ErrorDetails,
    StorageErrorType,
    StorageValidationError,
    StorageValidationWarning,
)
from yaozarrs.v05._storage import StorageValidatorV05


@contextmanager
def xfail_internet_error() -> Generator[None, None, None]:
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
    errors = exc_info.value.errors()
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
def test_validate_valid_demo_storage(
    type: str, write_demo_ome: Callable, as_store: Callable
) -> None:
    """Test validation on demo OME-Zarr files."""

    path = write_demo_ome(type, version="0.5")
    with xfail_internet_error():
        validate_zarr_store(as_store(path))

    if type == "image-with-labels":
        # Also validate with labels only
        validate_zarr_store(as_store(path / "labels"))
        validate_zarr_store(as_store(path / "labels" / "annotations"))
    elif type == "plate":
        # Validate a well only
        validate_zarr_store(as_store(path / "A" / "1"))


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


def _strip_version(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_version(v) for k, v in obj.items() if k != "version"}
    return [_strip_version(v) for v in obj] if isinstance(obj, list) else obj


@pytest.mark.parametrize("type", ["image", "labels", "plate", "image-with-labels"])
def test_validate_v04_without_version(type: str, write_demo_ome: Callable) -> None:
    """`version` is only a SHOULD in v0.4."""
    path = write_demo_ome(type, version="0.4")
    for zattrs in path.rglob(".zattrs"):
        zattrs.write_text(json.dumps(_strip_version(json.loads(zattrs.read_text()))))

    group = validate_zarr_store(path)
    assert group.ome_version() == "0.4"


def test_ome_version_ignores_top_level_keys_in_zarr_v3(tmp_path: Path) -> None:
    """v0.4 is zarr v2 only, so these aren't OME metadata in a zarr v3 group."""
    meta = {"zarr_format": 3, "node_type": "group", "attributes": {"plate": {}}}
    (tmp_path / "zarr.json").write_text(json.dumps(meta))
    assert open_group(tmp_path).ome_version() is None


def test_validate_unknown_version(write_demo_ome: Callable) -> None:
    path = write_demo_ome("image", version="0.4")
    zattrs = path / ".zattrs"
    attrs = json.loads(zattrs.read_text())
    attrs["multiscales"][0]["version"] = "0.3"
    zattrs.write_text(json.dumps(attrs))

    with pytest.raises(StorageValidationError, match=r"version '0\.3'") as exc_info:
        validate_zarr_store(path)
    assert exc_info.value.errors()[0]["type"] == "unsupported_version"


@pytest.mark.parametrize("ome", [{"scene": {}}, {"plate": {}}])
def test_validate_missing_version(tmp_path: Path, ome: dict) -> None:
    """A zarr v3 group with OME metadata but no version is invalid, not 'not OME'."""
    meta = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}}
    (tmp_path / "zarr.json").write_text(json.dumps(meta))

    with pytest.raises(StorageValidationError, match="must declare its 'version'") as e:
        validate_zarr_store(tmp_path)
    assert e.value.errors()[0]["type"] == "version_missing"
    assert e.value.errors()[0]["loc"] == ("ome", "version")


def _coerce_scale(zarr_json: Path) -> None:
    """Make a scale in this image a string, which is only accepted by coercion."""
    meta = json.loads(zarr_json.read_text())
    dataset = meta["attributes"]["ome"]["multiscales"][0]["datasets"][0]
    dataset["coordinateTransformations"][0]["scale"][0] = "1"
    zarr_json.write_text(json.dumps(meta))


def test_validate_strict_rejects_coerced_root_metadata(
    write_demo_ome: Callable,
) -> None:
    path = write_demo_ome("image", version="0.5")
    _coerce_scale(path / "zarr.json")

    validate_zarr_store(path)
    with pytest.raises(StorageValidationError) as exc:
        validate_zarr_store(path, strict=True)
    # one entry per pydantic error, located from the root
    errors = exc.value.errors()
    assert all(err["loc"][0] == "ome" for err in errors)
    assert any("scale" in err["loc"] for err in errors)


def test_invalid_root_metadata_keeps_warnings() -> None:
    """Root metadata errors are StorageValidationErrors, and warnings still come out."""
    axes = [{"name": n, "type": "space", "unit": "lightyear"} for n in "yx"]
    ome = {"version": "0.5", "multiscales": [{"axes": axes, "datasets": []}]}
    zarr_json = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}}
    store = {"zarr.json": json.dumps(zarr_json).encode()}

    with pytest.warns(StorageValidationWarning, match="lightyear"):
        with pytest.raises(StorageValidationError, match="datasets"):
            validate_zarr_store(store)


def test_validate_strict_rejects_coerced_child_metadata(
    write_demo_ome: Callable,
) -> None:
    path = write_demo_ome("plate", version="0.5")
    _coerce_scale(path / "A" / "1" / "0" / "zarr.json")

    # (the same group, to check that the cached result isn't reused)
    group = open_group(path)
    group.validate()
    with pytest.raises(StorageValidationError):
        group.validate(strict=True)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize("type", ["image", "labels", "plate", "image-with-labels"])
def test_validate_strict_accepts_valid_stores(
    type: str, version: str, write_demo_ome: Callable
) -> None:
    validate_zarr_store(write_demo_ome(type, version=version), strict=True)


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
            "path": "s0",
            "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}],
        },
    ],
}
MULTI_SCALE2 = {
    "name": "timelapse2",
    "axes": [{"name": "x", "type": "space"}, {"name": "y", "type": "space"}],
    "datasets": [
        {
            "path": "s1",
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
            lambda p: shutil.rmtree(p / "s0"),
        ),
        StorageTestCase(
            StorageErrorType.dataset_not_array,
            {"type": "image"},
            update_meta("s0", ("node_type",), "group"),
        ),
        StorageTestCase(
            StorageErrorType.dataset_dimension_mismatch,
            {"type": "image"},
            update_meta("s0", ("shape",), [10] * 5),
        ),
        StorageTestCase(
            StorageErrorType.dimension_names_mismatch,
            {"type": "image"},
            update_meta("s0", ("dimension_names",), ["a", "b", "c", "d"]),
        ),
        StorageTestCase(
            # e.g. tifffile writes Y1, X1 for level 1 (issue #60)
            StorageErrorType.dimension_names_mismatch,
            {"type": "image"},
            update_meta("s1", ("dimension_names",), ["c1", "z1", "y1", "x1"]),
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
            # the plate has several acquisitions, but the fields don't say which
            StorageErrorType.field_acquisition_missing,
            {"type": "plate"},
            update_meta(
                (),
                ("attributes", "ome", "plate", "acquisitions"),
                [{"id": 0}, {"id": 1}],
            ),
        ),
        StorageTestCase(
            # no "ome" key at all
            StorageErrorType.well_invalid,
            {"type": "plate"},
            update_meta(("A", "1"), ("attributes",), {}),
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
            update_meta("annotations", ("attributes", "ome"), {}),
        ),
        StorageTestCase(
            # no "ome" key at all
            StorageErrorType.label_image_invalid,
            {"type": "labels"},
            update_meta("annotations", ("attributes",), {}),
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
            update_meta(("labels", "annotations", "s0"), ("data_type",), "float32"),
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
    case: StorageTestCase, write_demo_ome: Callable, as_store: Callable
) -> None:
    path = write_demo_ome(**case.kwargs)
    case.mutator(path)
    with xfail_internet_error():
        with pytest.raises(StorageValidationError, match=str(case.err_type)):
            validate_zarr_store(as_store(path))


def _drop_key(zarr_json: Path, *keys: str) -> None:
    data = json.loads(zarr_json.read_text())
    *parents, last = keys
    d = data
    for k in parents:
        d = d[k]
    del d[last]
    zarr_json.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "case",
    [
        StorageTestCase(
            StorageErrorType.label_image_invalid,
            {"type": "labels"},
            lambda p: _drop_key(
                p / "annotations" / "zarr.json", "attributes", "ome", "image-label"
            ),
        ),
        StorageTestCase(
            # valid OME metadata, but not a labels group
            StorageErrorType.labels_metadata_invalid,
            {"type": "image-with-labels"},
            update_meta("labels", ("attributes", "ome"), IMAGE_META),
        ),
        StorageTestCase(
            # zarr-python omits dimension_names unless explicitly passed
            StorageErrorType.dimension_names_missing,
            {"type": "image"},
            lambda p: _drop_key(p / "s0" / "zarr.json", "dimension_names"),
        ),
    ],
    ids=lambda x: x.err_type,
)
def test_validate_invalid_storage_warns(
    case: StorageTestCase, write_demo_ome: Callable, as_store: Callable
) -> None:
    path = write_demo_ome(**case.kwargs)
    case.mutator(path)
    with xfail_internet_error():
        with pytest.warns(StorageValidationWarning, match=str(case.err_type)):
            validate_zarr_store(as_store(path))


def _write_bf2raw_v04(root: Path, image: Path) -> Path:
    shutil.copytree(image, root / "0")
    for sub, attrs in (("", {"bioformats2raw.layout": 3}), ("OME", {"series": ["0"]})):
        (root / sub).mkdir(exist_ok=True)
        (root / sub / ".zgroup").write_text('{"zarr_format": 2}')
        (root / sub / ".zattrs").write_text(json.dumps(attrs))
    return root


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_validate_ome_group_directly(
    version: str, write_demo_ome: Callable, tmp_path: Path
) -> None:
    """Series paths are relative to the parent of the OME group."""
    if version == "0.4":
        # (the demo writer can't write a v0.4 bioformats2raw store)
        image = write_demo_ome("image", version="0.4")
        path = _write_bf2raw_v04(tmp_path / "bf2raw.zarr", image)
    else:
        path = write_demo_ome("bioformats2raw", write_ome_group=True)
    validate_zarr_store(path / "OME")
    validate_zarr_store(open_group(path)["OME"])

    shutil.rmtree(path / "0")
    with pytest.raises(StorageValidationError, match="series_path_not_found"):
        validate_zarr_store(path / "OME")


def test_bf2raw_series_errors_propagate(
    write_demo_ome: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_demo_ome("bioformats2raw", write_ome_group=True)
    monkeypatch.setattr(
        StorageValidatorV05, "visit_series", Mock(side_effect=RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        validate_zarr_store(path)


def test_bf2raw_invalid_ome_group_uses_numbered_paths(write_demo_ome: Callable) -> None:
    path = write_demo_ome("bioformats2raw", write_ome_group=True)
    update_meta("OME", ("attributes",), {})(path)
    validate_zarr_store(path)


def _bf2raw_plate(write_demo_ome: Callable) -> tuple[Path, list[str]]:
    """A bioformats2raw plate (without an `OME` group), and its image paths."""
    path = write_demo_ome("plate")
    update_meta("", ("attributes", "ome", "bioformats2raw.layout"), 3)(path)
    return path, bioformats2raw_series(path)


def _write_series(path: Path, series: list[str]) -> None:
    (path / "OME").mkdir()
    ome = {"version": "0.5", "series": series}
    doc = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}}
    (path / "OME" / "zarr.json").write_text(json.dumps(doc))


def test_bf2raw_plate_series(write_demo_ome: Callable) -> None:
    path, images = _bf2raw_plate(write_demo_ome)
    validate_zarr_store(path)
    _write_series(path, images[::-1])  # (any order)
    validate_zarr_store(path)


def test_bf2raw_plate_series_mismatch_warns(write_demo_ome: Callable) -> None:
    path, images = _bf2raw_plate(write_demo_ome)
    _write_series(path, images[1:])
    with pytest.warns(StorageValidationWarning, match="series_plate_mismatch"):
        validate_zarr_store(path)


def test_bf2raw_plate_series_bad_path(write_demo_ome: Callable) -> None:
    path, images = _bf2raw_plate(write_demo_ome)
    _write_series(path, [*images, "nope"])
    with pytest.warns(StorageValidationWarning, match="series_plate_mismatch"):
        with pytest.raises(StorageValidationError, match="series_path_not_found"):
            validate_zarr_store(path)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize("root", [".", "labels", "labels/annotations"])
def test_validate_label_image_arrays(
    version: str, root: str, write_demo_ome: Callable, as_store: Callable
) -> None:
    path = write_demo_ome("image-with-labels", version=version)
    shutil.rmtree(path / "labels" / "annotations" / "s0")
    with pytest.raises(StorageValidationError, match="dataset_path_not_found"):
        validate_zarr_store(as_store(path / root))


def test_validate_complex_ome_zarr(complex_ome_zarr: Path) -> None:
    """Test validation on demo OME-Zarr files."""

    with xfail_internet_error():
        validate_zarr_store(complex_ome_zarr)


def test_validate_complex_ome_zarr_broken(
    complex_ome_zarr_broken: Path, as_store: Callable
) -> None:
    """Test validation on demo OME-Zarr files."""

    with pytest.raises(StorageValidationError) as err:
        validate_zarr_store(as_store(complex_ome_zarr_broken))

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


# ---------------------------------------------------------------------------
# v0.4/v0.5 scale/translation "stored either as a list of floats (`scale`) or as
# binary data at a location in this container (`path`)"
# ---------------------------------------------------------------------------

YX = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
VERSIONS = pytest.mark.parametrize("mod", [v04, v05], ids=["v04", "v05"])


@VERSIONS
def test_transform_at_path_model(mod: Any) -> None:
    ds = {"path": "0", "coordinateTransformations": [{"type": "scale", "path": "s"}]}
    ms = mod.Multiscale(axes=YX, datasets=[ds])
    (scale,) = ms.datasets[0].coordinateTransformations
    assert scale.ndim is None
    assert ms.model_dump(exclude_unset=True)["datasets"] == [ds]
    # (an inline vector dumps as before, without `path`)
    assert mod.ScaleTransformation(scale=[1, 2]).model_dump() == {
        "type": "scale",
        "scale": [1.0, 2.0],
    }
    for bad in ({"type": "scale"}, {"type": "scale", "scale": [1, 1], "path": "s"}):
        with pytest.raises(ValidationError, match="requires exactly one of"):
            mod.Dataset(path="0", coordinateTransformations=[bad])
    # (the union is discriminated: one error, under the class name; and without
    # `type`, as `DimSpec` writes them, the values tell which it is)
    with pytest.raises(ValidationError) as e:
        mod.Dataset(
            path="0", coordinateTransformations=[{"type": "scale", "scale": [1]}]
        )
    (err,) = e.value.errors()
    assert err["loc"] == (
        "coordinateTransformations",
        0,
        "ScaleTransformation",
        "scale",
    )
    ts = [{"scale": [1, 1]}, {"translation": [0, 0]}]
    ds = mod.Dataset(path="0", coordinateTransformations=ts)
    assert [t.type for t in ds.coordinateTransformations] == ["scale", "translation"]
    # (but nothing is guessed when the values don't say which it is)
    for ambiguous in ({"path": "t"}, {"scale": [1, 1], "translation": [0, 0]}):
        with pytest.raises(ValidationError, match="Unable to extract tag"):
            mod.Dataset(path="0", coordinateTransformations=[ambiguous])


def _node_files(version: str, node: Any, dims: Any = None) -> dict[str, Any]:
    """Metadata files of a group (`node`: its OME attributes) or array (its shape)."""
    node = list(node) if isinstance(node, tuple) else node
    if isinstance(node, dict):  # a group
        if version == "0.4":
            return {".zgroup": {"zarr_format": 2}, ".zattrs": node}
        attrs = {"ome": {"version": "0.5", **node}} if node else {}
        return {
            "zarr.json": {"zarr_format": 3, "node_type": "group", "attributes": attrs}
        }
    if version == "0.4":
        meta = {"zarr_format": 2, "shape": node, "chunks": node, "dtype": "<f8"}
        extra = {"compressor": None, "fill_value": 0, "order": "C", "filters": None}
        return {".zarray": {**meta, **extra}}
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    meta = {"zarr_format": 3, "node_type": "array", "shape": node, "codecs": codecs}
    grid = {"name": "regular", "configuration": {"chunk_shape": node}}
    extra = {"data_type": "float64", "chunk_grid": grid, "fill_value": 0}
    extra["chunk_key_encoding"] = {"name": "default"}
    return {
        "zarr.json": {**meta, **extra, **({"dimension_names": dims} if dims else {})}
    }


def _vector_store(
    root: Path, version: str, vec: Any = (2,), *, path: str = "vec", where: str = "ds"
) -> None:
    """A YX image with its scale (`where="ds"`) or a translation of the multiscale
    (`"ms"`) stored at `path`. `vec` is that array's shape, or {} for a group."""
    stored = {"type": "scale" if where == "ds" else "translation", "path": path}
    inline = {"type": "scale", "scale": [1, 1]}
    ds = {
        "path": "s0",
        "coordinateTransformations": [stored if where == "ds" else inline],
    }
    ms: dict[str, Any] = {"axes": YX, "datasets": [ds]}
    if where == "ms":
        ms["coordinateTransformations"] = [inline, stored]
    if version == "0.4":
        ms["version"] = "0.4"
    for name, node in {"": {"multiscales": [ms]}, "s0": [8, 8], "vec": vec}.items():
        (root / name).mkdir(parents=True, exist_ok=True)
        dims = ["y", "x"] if name == "s0" else None
        for file, meta in _node_files(version, node, dims).items():
            (root / name / file).write_text(json.dumps(meta))


@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize(
    "vec, where, problem",
    [
        ([2], "ds", None),
        ([2], "ms", None),
        ([3], "ds", "has shape [3], expected 2"),
        ([3], "ms", "has shape [3], expected 2"),
        ({}, "ds", "is not a Zarr array"),
    ],
)
def test_vector_at_path_storage(
    tmp_path: Path, version: str, vec: Any, where: str, problem: str | None
) -> None:
    _vector_store(tmp_path, version, vec, where=where)
    if problem is None:
        validate_zarr_store(tmp_path)
        return
    with pytest.raises(StorageValidationError, match=re.escape(problem)) as e:
        validate_zarr_store(tmp_path)
    loc = ("datasets", 0, "coordinateTransformations", 0, "path")
    if where == "ms":
        loc = ("coordinateTransformations", 1, "path")
    prefix = ("ome", "multiscales", 0) if version == "0.5" else ("multiscales", 0)
    assert [tuple(err["loc"]) for err in e.value.errors()] == [(*prefix, *loc)]


# ("binary data at a location in this container": not necessarily a Zarr array in
# this group, so a path that doesn't resolve to one only warns)
@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize("path", ["../vec", "nope"])
def test_unresolved_vector_path_warns(tmp_path: Path, version: str, path: str) -> None:
    _vector_store(tmp_path, version, path=path)
    with pytest.warns(StorageValidationWarning, match="isn't checked"):
        validate_zarr_store(tmp_path)


def test_vector_data_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _vector_store(tmp_path, "0.5")
    read: list[str] = []
    for name in ("get", "getitems"):
        real = getattr(_CachedMapper, name)

        def spy(self: Any, key: Any, *args: Any, _real: Any = real) -> Any:
            read.extend([key] if isinstance(key, str) else key)
            return _real(self, key, *args)

        monkeypatch.setattr(_CachedMapper, name, spy)
    validate_zarr_store(tmp_path)
    assert {k for k in read if k.startswith("vec/")} == {"vec/zarr.json"}
