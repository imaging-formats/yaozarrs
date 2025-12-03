import os
from typing import Any, TypeAlias, TypeVar, overload

from pydantic import TypeAdapter

from . import v04, v05

AnyOMEGroup: TypeAlias = v04.OMEZarrGroupJSON | v05.OMEZarrGroupJSON
AnyOME: TypeAlias = AnyOMEGroup | v05.OMEMetadata | v05.OMEAttributes
T = TypeVar("T", bound=AnyOME)


@overload
def validate_ome_object(node: Any, cls: type[T]) -> T: ...
@overload
def validate_ome_object(node: Any) -> AnyOME: ...
def validate_ome_object(node: Any, cls: type[T] | Any = None) -> T | AnyOME:
    """Validate any ome-zarr document or node as a python object.

    Parameters
    ----------
    node : OMENode
        The OMENode instance to validate.
    cls : type[T]
        The class to validate against. Must be a subclass of `BaseModel`.
        If not provided, defaults to `OMENode`, meaning any valid OME node object

    Raises
    ------
    pydantic.ValidationError
        If the validation fails.
    """
    adapter = TypeAdapter[T](cls or AnyOME)
    return adapter.validate_python(node)


@overload
def validate_ome_json(data: str | bytes | bytearray, cls: type[T]) -> T: ...
@overload
def validate_ome_json(data: str | bytes | bytearray) -> AnyOME: ...
def validate_ome_json(
    data: str | bytes | bytearray, cls: type[T] | Any = None
) -> T | AnyOME:
    """Validate any valid ome-zarr JSON data.

    By default, this will validate `data` against all known OME JSON documents.
    This includes ome-zarr group documents for v04 (found at .zattrs in the zarr group)
    and v05 (found at zarr.json in the zarr group).  For v05 objects, it also detects
    data that would valid as the value of the `data["attributes"]["ome"]` key inside
    a v05 zarr.json document.

    Parameters
    ----------
    data : str | bytes | bytearray
        The OMENode instance to validate.
    cls : type[T]
        The class to validate against. Must be a subclass of `BaseModel`.
        If not provided, defaults to `OMENode`, meaning any valid OME node object

    Raises
    ------
    pydantic.ValidationError
        If the validation fails.
    """
    adapter = TypeAdapter[T](cls or AnyOME)
    return adapter.validate_json(data)


@overload
def from_uri(uri: str | os.PathLike, cls: type[T]) -> T: ...
@overload
def from_uri(uri: str | os.PathLike) -> AnyOMEGroup: ...
def from_uri(uri: str | os.PathLike, cls: type[T] | Any = None) -> T | AnyOMEGroup:
    """Load and validate any OME-Zarr group from a URI or local path.

    This function will attempt to load the OME-Zarr group metadata from the given
    URI or local path. It supports both v0.4 and v0.5 of the OME-Zarr specification.
    The URI should be a path to a zarr group (directory or URL) with valid ome-zarr
    metadata, or a path directly to the metadata JSON file itself (e.g. zarr.json or
    .zattrs).

    This requires that you have installed yaozarrs with the `io` extra, e.g.
    `pip install yaozarrs[io]`.

    Parameters
    ----------
    uri : str | os.PathLike
        The URI or local path to the OME-Zarr group. This can be a file path,
        a directory path, or a URL.
    cls : type[T]
        The class to validate against. Must be a subclass of `BaseModel`.

    Returns
    -------
    AnyOME
        An instance of `v05.OMEZarrGroupJSON`, `v04.OMEZarrGroupJSON`, or another
        valid OME-Zarr node type, depending on the object detected.

    Raises
    ------
    FileNotFoundError
        If the URI does not point to a valid OME-Zarr group.
    pydantic.ValidationError
        If the loaded metadata is not valid according to the OME-Zarr specification.
    """
    from ._io import read_json_from_uri

    json_content, uri_str = read_json_from_uri(uri)
    obj = validate_ome_json(json_content, cls or AnyOMEGroup)  # type: ignore
    obj.uri = uri_str
    return obj
