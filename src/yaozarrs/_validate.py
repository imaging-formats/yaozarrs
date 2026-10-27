import os
from typing import Annotated, Any, TypeAlias, TypeVar, overload

from pydantic import Discriminator, Tag, TypeAdapter

from . import v04, v05, v06

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
_VERSIONS = {"0.4": "v04", "0.5": "v05"}


def _declared_version(obj: Any) -> str | None:
    """Return the package for the OME version that `obj` declares (if known)."""
    if isinstance(obj, dict) and isinstance(obj.get("attributes"), dict):
        obj = obj["attributes"].get("ome")  # a zarr.json document
    elif isinstance(obj, dict) and "ome" in obj:
        obj = obj["ome"]  # the attributes of a zarr.json document
    if not isinstance(obj, dict):
        return None
    version = obj.get("version")
    if version is None:
        # (v0.4 declares it inside the metadata object, e.g. multiscales[0].version)
        for key in ("multiscales", "plate", "well", "image-label"):
            inner = obj.get(key)
            if isinstance(inner, list) and inner:
                inner = inner[0]
            if isinstance(inner, dict) and "version" in inner:
                version = inner["version"]
                break
    return _VERSIONS.get(version) if isinstance(version, str) else None


def _discriminate_group(obj: Any) -> str:
    return _declared_version(obj) or "unknown"


def _discriminate_any(obj: Any) -> str:
    version = _declared_version(obj)
    if version in (None, "v04"):
        return version or "unknown"
    if "attributes" in obj:
        return f"{version}-group"
    return f"{version}-attributes" if "ome" in obj else f"{version}-metadata"


# The default validation targets.  A document that declares a version is validated
# only against that version's models: in a plain union, pydantic validates it against
# every member (running their validators, and warnings, even for members that end up
# failing), and lists every member's errors.  Documents without a (known) version are
# tried against all of them.
_AnyOMEGroupByVersion: TypeAlias = Annotated[
    Annotated[v04.OMEZarrGroupJSON, Tag("v04")]
    | Annotated[v05.OMEZarrGroupJSON, Tag("v05")]
    | Annotated[AnyOMEGroup, Tag("unknown")],
    Discriminator(_discriminate_group),
]
_AnyOMEByVersion: TypeAlias = Annotated[
    Annotated[v04.OMEZarrGroupJSON, Tag("v04")]
    | Annotated[v05.OMEZarrGroupJSON, Tag("v05-group")]
    | Annotated[v05.OMEAttributes, Tag("v05-attributes")]
    | Annotated[v05.OMEMetadata, Tag("v05-metadata")]
    | Annotated[AnyOME, Tag("unknown")],
    Discriminator(_discriminate_any),
]


@overload
def validate_ome_object(node: Any, cls: type[T]) -> T: ...
@overload
def validate_ome_object(node: Any) -> AnyOME: ...
def validate_ome_object(node: Any, cls: type[T] | Any = None) -> T | AnyOME:
    """Validate any python object representing an OME-Zarr document or node.

    Parameters
    ----------
    node : Any
        The OME Node instance to validate.
    cls : type[T]
        The class to validate against. Must be a subclass of `BaseModel`.
        If not provided, defaults to `AnyOME`, meaning "any valid OME node object".

    Returns
    -------
    object: T
        The validated (pydantic) model instance.

    Raises
    ------
    pydantic.ValidationError
        If the validation fails.
    """
    adapter: TypeAdapter[T | AnyOME] = TypeAdapter(cls or _AnyOMEByVersion)
    return adapter.validate_python(node)


@overload
def validate_ome_json(data: str | bytes | bytearray, cls: type[T]) -> T: ...
@overload
def validate_ome_json(data: str | bytes | bytearray) -> AnyOME: ...
def validate_ome_json(
    data: str | bytes | bytearray, cls: type[T] | Any = None
) -> T | AnyOME:
    """Validate any ome-zarr JSON string or bytes literal.

    By default, this will validate `data` against all known OME JSON documents.
    This includes ome-zarr group documents for v04 (found at .zattrs in the zarr group)
    and v05/v06 (found at zarr.json in the zarr group).  For v05 and v06 objects, it
    also detects data that would be valid as the value of the
    `data["attributes"]["ome"]` key inside a zarr.json document.

    Parameters
    ----------
    data : str | bytes | bytearray
        The OMENode instance to validate.
    cls : type[T]
        The class to validate against. Must be a subclass of `BaseModel`.
        If not provided, defaults to `OMENode`, meaning any valid OME node object

    Returns
    -------
    object: T
        The validated (pydantic) model instance.

    Raises
    ------
    pydantic.ValidationError
        If the validation fails.
    """
    adapter: TypeAdapter[T | AnyOME] = TypeAdapter(cls or _AnyOMEByVersion)
    return adapter.validate_json(data)


@overload
def validate_ome_uri(uri: str | os.PathLike, cls: type[T]) -> T: ...
@overload
def validate_ome_uri(uri: str | os.PathLike) -> AnyOMEGroup: ...
def validate_ome_uri(
    uri: str | os.PathLike, cls: type[T] | Any = None
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
    obj = validate_ome_json(json_content, cls or _AnyOMEGroupByVersion)  # type: ignore
    obj.uri = uri_str
    return obj
