from typing import Any, TypeAlias, TypeVar, overload

from pydantic import TypeAdapter

from . import v04, v05

AnyOME: TypeAlias = v05.OMEZarrGroupJSON | v04.OMEZarrGroupJSON | v05.OMEMetadata
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
