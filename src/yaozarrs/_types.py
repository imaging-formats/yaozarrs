from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from typing import Annotated, Any, TypeVar

from pydantic import AfterValidator, BaseModel, Field
from pydantic_core import PydanticCustomError

T = TypeVar("T")
K = TypeVar("K", str, int)  # (names, paths, ids)

BASIC_TYPES = (str, int, float, bool, type(None))


def _canonical_key(x: Any) -> Hashable:
    """Return a hashable key that compares equal iff two items are JSON-equivalent.

    This lets uniqueness be checked with a dict (O(N)) rather than by comparing
    every pair (O(N²)); pydantic models are unhashable by default, so they can't
    be put in a set directly.

    Composite keys are tagged by kind so that, e.g., a model can never collide
    with a sequence that happens to canonicalize to the same tuple.
    """
    if isinstance(x, BaseModel):
        # class identity is part of the key because BaseModel.__eq__ compares
        # classes; the fully qualified name keeps same-named models in different
        # modules distinct (e.g. v04.Column vs v05.Column).
        # (the pydantic-core serializer directly: model_dump_json's Python wrappers
        # cost more than the serialization itself)
        cls = type(x)
        dump = cls.__pydantic_serializer__.to_json(x, exclude_none=True)
        return ("model", f"{cls.__module__}.{cls.__qualname__}", dump)
    if isinstance(x, BASIC_TYPES):
        # returned bare: dict lookup uses hash/__eq__, so numeric equivalences
        # (1 == 1.0 == True) collapse just as `a == b` did.
        return x
    if isinstance(x, Mapping):
        # frozenset rather than sorted() because keys need not be orderable
        return ("map", frozenset((k, _canonical_key(v)) for k, v in x.items()))
    if isinstance(x, Sequence):
        return ("seq", tuple(_canonical_key(i) for i in x))
    raise TypeError(f"Unsupported type for JSON equivalence: {type(x)}")


def _validate_unique_list(v: list[T]) -> list[T]:
    """Validate that all items in the list are unique, using JSON equivalence."""
    if len(v) < 2:
        return v
    seen: dict[Hashable, int] = {}
    for i, item in enumerate(v):
        key = _canonical_key(item)
        if (j := seen.get(key)) is not None:
            raise PydanticCustomError(
                "listItemsNotUnique",
                "List items are not unique. Equal items found at indices: {idx}",
                {"idx": (j, i)},
            )
        seen[key] = i
    return v


# Declares `uniqueItems` in the JSON schema without checking it.  For lists whose
# items must have unique names (or paths): that stricter check implies uniqueItems,
# and is much cheaper than comparing whole items.
UNIQUE_ITEMS = Field(json_schema_extra={"uniqueItems": True})

# A list that enforces uniqueItems (no two items may be equal, as JSON)
UniqueList = Annotated[list[T], AfterValidator(_validate_unique_list), UNIQUE_ITEMS]


def duplicates(values: Sequence[K]) -> list[K]:
    """Return the values that occur more than once, sorted."""
    if len(set(values)) == len(values):
        return []
    return sorted(v for v, n in Counter(values).items() if n > 1)
