from typing import Any, TypeAlias, TypeVar, overload

from pydantic import TypeAdapter

from .v05 import OMEMetadata, OMEZarrGroupJSON

AnyNode: TypeAlias = OMEZarrGroupJSON | OMEMetadata
T = TypeVar("T", bound=AnyNode)


@overload
def validate_ome_node(node: Any, cls: type[T]) -> T: ...
@overload
def validate_ome_node(node: Any) -> AnyNode: ...
def validate_ome_node(node: Any, cls: type[T] | Any = None) -> T | AnyNode:
    """Validate any ome-zarr document or node.

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
    adapter = TypeAdapter[T](cls or AnyNode)
    return adapter.validate_python(node)
