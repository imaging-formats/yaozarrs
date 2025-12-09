from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypeVar

from pydantic import AfterValidator, BaseModel, Field
from pydantic_core import PydanticCustomError

T = TypeVar("T")

BASIC_TYPES = (str, int, float, bool, type(None), BaseModel)


def _is_json_equivalent(a: Any, b: Any) -> bool:
    if isinstance(a, BASIC_TYPES) and isinstance(b, BASIC_TYPES):
        return bool(a == b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):  # pragma: no cover
        if a.keys() != b.keys():
            return False
        return all(_is_json_equivalent(a[k], b[k]) for k in a)
    if isinstance(a, Sequence) and isinstance(b, Sequence):  # pragma: no cover
        return all(_is_json_equivalent(x, y) for x, y in zip(a, b, strict=True))
    raise TypeError(  # pragma: no cover
        f"Unsupported type for JSON equivalence: {type(a)}"
    )


def _validate_unique_list(v: list[T]) -> list[T]:
    """Validate that all items in the list are unique, using JSON equivalence."""
    for i, a in enumerate(v):
        for j in range(i + 1, len(v)):
            if _is_json_equivalent(a, v[j]):
                raise PydanticCustomError(
                    "listItemsNotUnique",
                    "List items are not unique. Equal items found at indices: {idx}",
                    {"idx": (i, j)},
                )
    return v


# A list that enforces uniqueItems
UniqueList = Annotated[
    list[T],
    AfterValidator(_validate_unique_list),
    Field(json_schema_extra={"uniqueItems": True}),
]
