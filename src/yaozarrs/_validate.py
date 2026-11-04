from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias, TypeVar, overload

from pydantic import Discriminator, Tag, TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError

from . import v04, v05, v06
from ._base import MISSING_VERSION, REQUIRE_VERSION, type_adapter
from ._errors import (
    ErrorDetails,
    StorageErrorType,
    StorageValidationError,
    StorageValidationWarning,
    ValidationResult,
    ValidationWarning,
)
from ._zarr import ZarrGroup, open_group
from .v06._version import VERSIONS as V06_VERSIONS

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping
    from pathlib import Path

    from pydantic_core import InitErrorDetails

AnyOMEGroup: TypeAlias = (
    v04.OMEZarrGroupJSON | v05.OMEZarrGroupJSON | v06.OMEZarrGroupJSON
)
AnyOME: TypeAlias = (
    AnyOMEGroup
    | v05.OMEMetadata
    | v05.OMEAttributes
    | v06.OMEMetadata
    | v06.OMEAttributes
)
T = TypeVar("T", bound=AnyOME)

# OME version strings -> the package that models them
_VERSIONS = {"0.4": "v04", "0.5": "v05", **dict.fromkeys(V06_VERSIONS, "v06")}


def _find_version(obj: Any) -> tuple[Any, tuple[str | int, ...]] | None:
    """Return the OME version that `obj` declares (if any), and where it is."""
    loc: tuple[str | int, ...] = ()
    if isinstance(obj, dict) and isinstance(obj.get("attributes"), dict):
        obj, loc = obj["attributes"].get("ome"), ("attributes", "ome")  # zarr.json
    elif isinstance(obj, dict) and "ome" in obj:
        obj, loc = obj["ome"], ("ome",)  # the attributes of a zarr.json document
    if not isinstance(obj, dict):
        return None
    if obj.get("version") is not None:
        return obj["version"], (*loc, "version")
    # (v0.4 declares it inside the metadata object, e.g. multiscales[0].version)
    for key in ("multiscales", "plate", "well", "image-label"):
        inner, inner_loc = obj.get(key), (*loc, key)
        if isinstance(inner, list) and inner:
            inner, inner_loc = inner[0], (*inner_loc, 0)
        if isinstance(inner, dict) and inner.get("version") is not None:
            return inner["version"], (*inner_loc, "version")
    return None


def _declared_version(obj: Any) -> str | None:
    """Return the package for the OME version that `obj` declares (if any).

    An unknown version returns a tag that matches no model, so that it gets rejected.
    """
    if (found := _find_version(obj)) is None:
        return None
    version = found[0]
    return _VERSIONS.get(str(version), f"version {version!r}")


def _discriminate_group(obj: Any) -> str:
    return _declared_version(obj) or "unknown"


def _discriminate_any(obj: Any) -> str:
    version = _declared_version(obj)
    if version not in ("v05", "v06"):
        return version or "unknown"
    if "attributes" in obj:
        return f"{version}-group"
    return f"{version}-attributes" if "ome" in obj else f"{version}-metadata"


# The default validation targets.  A document that declares a version is validated
# only against that version's models: in a plain union, pydantic validates it against
# every member (running their validators, and warnings, even for members that end up
# failing), and lists every member's errors.  Documents without a version are tried
# against all of them.
_AnyOMEGroupByVersion: TypeAlias = Annotated[
    Annotated[v04.OMEZarrGroupJSON, Tag("v04")]
    | Annotated[v05.OMEZarrGroupJSON, Tag("v05")]
    | Annotated[v06.OMEZarrGroupJSON, Tag("v06")]
    | Annotated[AnyOMEGroup, Tag("unknown")],
    Discriminator(_discriminate_group),
]
_AnyOMEByVersion: TypeAlias = Annotated[
    Annotated[v04.OMEZarrGroupJSON, Tag("v04")]
    | Annotated[v05.OMEZarrGroupJSON, Tag("v05-group")]
    | Annotated[v05.OMEAttributes, Tag("v05-attributes")]
    | Annotated[v05.OMEMetadata, Tag("v05-metadata")]
    | Annotated[v06.OMEZarrGroupJSON, Tag("v06-group")]
    | Annotated[v06.OMEAttributes, Tag("v06-attributes")]
    | Annotated[v06.OMEMetadata, Tag("v06-metadata")]
    | Annotated[AnyOME, Tag("unknown")],
    Discriminator(_discriminate_any),
]


# The version-independent functions can't guess a document's version: the v0.5 and
# v0.6 models would each fill in their own, so every document they share (e.g. a
# plate) would pass as the first of them.  They require it instead (as every
# document validated from JSON must), which still leaves the documents that don't
# need one: v0.4 metadata, and v0.5 labels groups.
_CONTEXT = {REQUIRE_VERSION: True}


def _missing_version_error(obj: Any) -> ValidationError | None:
    """A single error for an `ome` object without a version (instead of one per model).

    `obj` can be a zarr.json document or its attributes.
    """
    loc: tuple[str, ...] = ("ome",)
    if isinstance(obj, dict) and isinstance(obj.get("attributes"), dict):
        obj, loc = obj["attributes"], ("attributes", "ome")
    if not isinstance(obj, dict) or not isinstance(ome := obj.get("ome"), dict):
        return None
    if "version" in ome:
        return None
    error = PydanticCustomError("missing_version", MISSING_VERSION)
    line: InitErrorDetails = {"type": error, "loc": (*loc, "version"), "input": ome}
    return ValidationError.from_exception_data("OME metadata", [line])


def _unsupported_version_error(obj: Any) -> ValidationError | None:
    """A single error for a version we don't know (instead of the union's errors)."""
    if (found := _find_version(obj)) is None or str(found[0]) in _VERSIONS:
        return None
    version, loc = found
    supported = ", ".join(v for v in _VERSIONS if v.replace(".", "").isdigit())
    error = PydanticCustomError(
        "unsupported_version",
        "Unsupported OME-Zarr version '{version}' (expected one of {supported})",
        {"version": version, "supported": supported},
    )
    line: InitErrorDetails = {"type": error, "loc": loc, "input": version}
    return ValidationError.from_exception_data("OME metadata", [line])


def _version_error(obj: Any) -> ValidationError | None:
    """A clearer error for a missing or unsupported version, if that's the problem."""
    return _missing_version_error(obj) or _unsupported_version_error(obj)


@overload
def validate_ome_object(
    node: Any, cls: type[T], *, strict: bool | None = None
) -> T: ...
@overload
def validate_ome_object(node: Any, *, strict: bool | None = None) -> AnyOME: ...
def validate_ome_object(
    node: Any, cls: type[T] | Any = None, *, strict: bool | None = None
) -> T | AnyOME:
    """Validate any python object representing an OME-Zarr document or node.

    Without `cls`, a v0.5 or v0.6 document must declare its `version` (as the spec
    requires): it can't be guessed, since many documents are the same in both.  With
    a `cls` of a given version (e.g. `v06.Plate`), a missing version is filled in.

    Parameters
    ----------
    node : Any
        The OME Node instance to validate.
    cls : type[T]
        The class to validate against. Must be a subclass of `BaseModel`.
        If not provided, defaults to `AnyOME`, meaning "any valid OME node object".
    strict : bool | None
        If `True`, reject input that would otherwise be coerced to the right type
        (e.g. `"1"` for an integer). By default, such input is accepted.

    Returns
    -------
    object: T
        The validated (pydantic) model instance.

    Raises
    ------
    pydantic.ValidationError
        If the validation fails.
    """
    if cls is not None:
        return type_adapter(cls).validate_python(node, strict=strict)
    adapter: TypeAdapter[AnyOME] = type_adapter(_AnyOMEByVersion)
    try:
        return adapter.validate_python(node, strict=strict, context=_CONTEXT)
    except ValidationError:
        if error := _version_error(node):
            raise error from None
        raise


@overload
def validate_ome_json(
    data: str | bytes | bytearray, cls: type[T], *, strict: bool | None = None
) -> T: ...
@overload
def validate_ome_json(
    data: str | bytes | bytearray, *, strict: bool | None = None
) -> AnyOME: ...
def validate_ome_json(
    data: str | bytes | bytearray,
    cls: type[T] | Any = None,
    *,
    strict: bool | None = None,
) -> T | AnyOME:
    """Validate any ome-zarr JSON string or bytes literal.

    By default, this will validate `data` against all known OME JSON documents.
    This includes ome-zarr group documents for v04 (found at .zattrs in the zarr group)
    and v05/v06 (found at zarr.json in the zarr group).  For v05 and v06 objects, it
    also detects data that would be valid as the value of the
    `data["attributes"]["ome"]` key inside a zarr.json document.  A v0.5 or v0.6
    document must declare its `version`.

    Parameters
    ----------
    data : str | bytes | bytearray
        The OMENode instance to validate.
    cls : type[T]
        The class to validate against. Must be a subclass of `BaseModel`.
        If not provided, defaults to `OMENode`, meaning any valid OME node object
    strict : bool | None
        If `True`, reject input that would otherwise be coerced to the right type
        (e.g. `"1"` for an integer). By default, such input is accepted.

    Returns
    -------
    object: T
        The validated (pydantic) model instance.

    Raises
    ------
    pydantic.ValidationError
        If the validation fails.
    """
    if cls is not None and cls is not _AnyOMEGroupByVersion:
        return type_adapter(cls).validate_json(data, strict=strict)
    adapter: TypeAdapter[AnyOME] = type_adapter(cls or _AnyOMEByVersion)
    try:
        return adapter.validate_json(data, strict=strict)
    except ValidationError:
        try:
            error = _version_error(json.loads(data))
        except ValueError:  # (not JSON)
            error = None
        if error:
            raise error from None
        raise


@overload
def validate_ome_uri(
    uri: str | os.PathLike, cls: type[T], *, strict: bool | None = None
) -> T: ...
@overload
def validate_ome_uri(
    uri: str | os.PathLike, *, strict: bool | None = None
) -> AnyOMEGroup: ...
def validate_ome_uri(
    uri: str | os.PathLike, cls: type[T] | Any = None, *, strict: bool | None = None
) -> T | AnyOMEGroup:
    """Load and validate root metadata in a OME-Zarr group from a URI or local path.

    !!!important
        This requires that you have installed yaozarrs with the `io` extra, e.g.
        `pip install yaozarrs[io]`.

    This function will attempt to load the OME-Zarr group metadata from the given URI or
    local path. The URI should be a path to a zarr group (directory or URL) with valid
    ome-zarr metadata, or a path directly to the metadata JSON file itself (e.g.
    zarr.json or .zattrs).

    **This _only_ loads and validates the root OME-Zarr group metadata.**  It does not
    traverse the hierarchy, perform structural validation, or validate any child groups
    or arrays.

    Parameters
    ----------
    uri : str | os.PathLike
        The URI or local path to the OME-Zarr group. This can be a file path,
        a directory path, or a URL.
    cls : type[T]
        The class to validate against. Must be a subclass of `BaseModel`.
    strict : bool | None
        If `True`, reject input that would otherwise be coerced to the right type
        (e.g. `"1"` for an integer). By default, such input is accepted.

    Returns
    -------
    AnyOME: T
        An instance of `v06.OMEZarrGroupJSON`, `v05.OMEZarrGroupJSON`,
        `v04.OMEZarrGroupJSON`, or another valid OME-Zarr node type, depending on the
        object detected.

    Raises
    ------
    FileNotFoundError
        If the URI does not point to a valid OME-Zarr group.
    pydantic.ValidationError
        If the loaded metadata is not valid according to the OME-Zarr specification.
    """
    from ._io import read_json_from_uri

    json_content, uri_str = read_json_from_uri(uri)
    cls = cls or _AnyOMEGroupByVersion
    obj = validate_ome_json(json_content, cls, strict=strict)  # type: ignore
    obj.uri = uri_str
    return obj


def validate_zarr_store(
    obj: ZarrGroup | str | Path | Mapping[str, bytes] | Any,
    *,
    strict: bool | None = None,
) -> ZarrGroup:
    """Validate both structure and metadata of an OME-Zarr hierarchy.

    !!!important
        Requires `yaozarrs[io]` or `fsspec` to be installed, unless `obj` is a mapping
        or a [`yaozarrs.ZarrGroup`][] opened from one.

    This is a high-level function to validate both the metadata and the structure
    of a complete OME-Zarr store.  This is the function used by the `yaozarrs validate`
    CLI command.

    Supports OME-Zarr versions 0.4, 0.5 and 0.6.

    Parameters
    ----------
    obj : ZarrGroup | str | Path | Mapping[str, bytes] | Any
        The zarr store to validate. Can be a URI string, a Path, a ZarrGroup instance,
        a zarr.Group object (for backwards compatibility), or a mapping of store keys
        to the bytes of the metadata documents, if you want to do the IO yourself (see
        [`yaozarrs.open_group`][]).  To validate a single document without a store,
        use [`yaozarrs.validate_ome_object`][] or [`yaozarrs.validate_ome_json`][].
    strict : bool | None
        If `True`, reject metadata that would otherwise be coerced to the right type
        (e.g. `"1"` for an integer). By default, such metadata is accepted.

    Returns
    -------
    ZarrGroup
        The opened ZarrGroup if validation is successful.

    Raises
    ------
    StorageValidationError
        If the storage structure is invalid.
    """
    zarr_group = open_group(obj)
    ome_version = zarr_group.ome_version()
    if not ome_version and "ome" in zarr_group.attrs:
        # (OME metadata, so an invalid OME-Zarr group rather than not one at all)
        error = ValidationResult().add_error(
            StorageErrorType.version_missing,
            ("ome", "version"),
            "The OME metadata must declare its 'version'",
        )
        raise StorageValidationError(error.errors)
    if not ome_version:
        raise ValueError(
            f"Unable to determine OME-Zarr version for {zarr_group}. "
            "Is this an OME-Zarr group?"
        )

    # (imported here: each version's storage validator imports this module)
    if ome_version == "0.5":
        from .v05._storage import StorageValidatorV05

        Validator = StorageValidatorV05
    elif ome_version == "0.4":
        from .v04._storage import StorageValidatorV04

        Validator = StorageValidatorV04
    elif ome_version in V06_VERSIONS:
        from .v06._storage import StorageValidatorV06

        Validator = StorageValidatorV06
    else:
        error = ValidationResult().add_error(
            StorageErrorType.unsupported_version,
            ("version",),
            f"Unsupported OME-Zarr version {ome_version!r}",
        )
        raise StorageValidationError(error.errors)

    # Capture RuntimeWarnings from pydantic validators during validation
    # and convert them to StorageValidationWarning
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.filterwarnings("always", category=ValidationWarning)
        result = Validator.validate_group(zarr_group, strict=strict)

    # prepend captured warnings to the result
    for w in captured_warnings:
        details: ErrorDetails = {
            "type": "model_warning",
            "loc": (),
            "msg": str(w.message),
        }
        result.warnings.insert(0, details)

    # Emit warnings for SHOULD directives from structural validation
    if result.warnings:
        # Use a custom formatter to match exception style
        warning_instance = StorageValidationWarning(result.warnings)

        _original_formatwarning = warnings.formatwarning

        def _custom_formatwarning(message, category, *_, **__) -> str:
            # Format like an exception: module.Class: message
            return f"{category.__module__}.{category.__name__}: {message}\n"

        warnings.formatwarning = _custom_formatwarning  # ty: ignore[invalid-assignment]
        try:
            warnings.warn(warning_instance, stacklevel=2)
        finally:
            warnings.formatwarning = _original_formatwarning

    # Raise error if any validation issues found
    if not result.is_valid:
        raise StorageValidationError(result.errors)

    return zarr_group
