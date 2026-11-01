"""Tests for the `UniqueList` annotated type."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

import yaozarrs._types
from yaozarrs._types import UniqueList, _canonical_key


class Item(BaseModel):
    name: str
    value: int = 0


class OtherItem(BaseModel):
    name: str
    value: int = 0


class Model(BaseModel):
    items: UniqueList[Item]


class Basic(BaseModel):
    items: UniqueList[Any]


def test_unique_list_accepts_unique_items() -> None:
    model = Model(items=[Item(name="a"), Item(name="b"), Item(name="a", value=1)])
    assert len(model.items) == 3


def test_unique_list_rejects_duplicates() -> None:
    with pytest.raises(ValidationError, match="List items are not unique"):
        Model(items=[Item(name="a"), Item(name="b"), Item(name="a")])


def test_unique_list_reports_first_duplicate_pair() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Model(items=[Item(name="a"), Item(name="b"), Item(name="a"), Item(name="b")])
    # indices are reported lowest-first, and the earliest duplicate pair wins
    assert "(0, 2)" in str(exc_info.value)


def test_unique_list_json_schema() -> None:
    assert Model.model_json_schema()["properties"]["items"]["uniqueItems"] is True


# --- JSON-equivalence semantics -------------------------------------------------
# `BaseModel.__eq__` compares classes, so distinct model classes are never equal,
# even with identical fields -- and same-named classes in different modules must
# not collide either.


def test_distinct_model_classes_are_not_duplicates() -> None:
    class Mixed(BaseModel):
        items: UniqueList[Any]

    assert len(Mixed(items=[Item(name="a"), OtherItem(name="a")]).items) == 2


def test_same_named_models_in_different_modules_are_not_duplicates() -> None:
    """`v04.Column` and `v05.Column` share a class name but are distinct models."""
    from yaozarrs import v04, v05

    a, b = v04.Column(name="1"), v05.Column(name="1")
    assert a != b  # BaseModel.__eq__ compares classes
    assert _canonical_key(a) != _canonical_key(b)
    assert len(Basic(items=[a, b]).items) == 2


@pytest.mark.parametrize(
    "items",
    [
        ["a", "b", "c"],
        [1, 2, 3],
        [None, False, "x"],
        [{"a": 1, "b": 2}, {"b": 2, "a": 1, "c": 3}],
        [{"a": 1}, {"a": 2}],
        [["a"], ["b"]],
        [["a"], ["a", "b"]],  # differing lengths are unequal, not an error
        [[1, [2]], [1, [3]]],
    ],
)
def test_unique_non_model_items(items: list) -> None:
    assert Basic(items=items).items == items


@pytest.mark.parametrize(
    "items",
    [
        ["a", "b", "a"],
        [1, 2, 1],
        [1, 1.0],  # numeric equivalence, as with `a == b`
        [1, True],
        [{"a": 1, "b": 2}, {"b": 2, "a": 1}],  # key order is irrelevant
        [["a"], ["a"]],
        [["a"], ("a",)],  # list/tuple are both JSON arrays
        [{"a": [1, {"b": 2}]}, {"a": [1, {"b": 2}]}],
    ],
)
def test_duplicate_non_model_items(items: list) -> None:
    with pytest.raises(ValidationError, match="List items are not unique"):
        Basic(items=items)


def test_int_keyed_and_str_keyed_mappings_differ() -> None:
    assert Basic(items=[{1: "a"}, {"1": "a"}]).items == [{1: "a"}, {"1": "a"}]


def test_unorderable_mapping_keys() -> None:
    items = [{1: "a", "b": 2}, {"b": 2}]
    assert Basic(items=items).items == items


def test_unsupported_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="Unsupported type for JSON equivalence"):
        Basic(items=[{1, 2}, {3}])


# --- performance ----------------------------------------------------------------


def test_unique_list_is_not_quadratic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation makes one _canonical_key call per item, not one per pair (see #54).

    A wall-clock timing comparison would demonstrate the same thing, but is
    inherently flaky on noisy/shared CI runners. Counting calls instead makes
    this deterministic: a single linear pass calls `_canonical_key` exactly
    `n` times; the old pairwise-`==` approach would call the equivalent
    comparison O(n^2) times.
    """
    calls = 0
    original = yaozarrs._types._canonical_key

    def _counting_key(x: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(x)

    monkeypatch.setattr(yaozarrs._types, "_canonical_key", _counting_key)

    n = 2000
    items = [Item(name=str(i)) for i in range(n)]
    Model(items=items)
    assert calls == n
