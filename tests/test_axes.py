"""The single `Axis` model, and its typed constructors (`SpaceAxis` & co)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

from yaozarrs import v04, v05, v06
from yaozarrs._errors import ValidationWarning

if TYPE_CHECKING:
    from types import ModuleType

# (v0.4 and v0.5 share their axis models)
VERSIONS = pytest.mark.parametrize("v", [v05, v06], ids=["v05", "v06"])
TYPED = [("SpaceAxis", "space"), ("TimeAxis", "time"), ("ChannelAxis", "channel")]


def test_v04_v05_share_axis() -> None:
    assert v04.Axis is v05.Axis
    assert v04.SpaceAxis is v05.SpaceAxis


@VERSIONS
def test_axis_is_callable(v: ModuleType) -> None:
    ax = v.Axis(name="x", type="space", unit="micrometer")
    assert (ax.name, ax.type, ax.unit) == ("x", "space", "micrometer")


@VERSIONS
@pytest.mark.parametrize("ctor, type_", TYPED)
def test_constructors_return_axis(v: ModuleType, ctor: str, type_: str) -> None:
    ax = getattr(v, ctor)(name="a")
    assert type(ax) is v.Axis
    # (the same as a parsed axis)
    assert ax == v.Axis.model_validate({"name": "a", "type": type_})


def test_parsed_axes_are_all_axis() -> None:
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    scale = [{"type": "scale", "scale": [1, 1, 1, 1]}]
    ms = v05.Multiscale.model_validate(
        {"axes": axes, "datasets": [{"path": "0", "coordinateTransformations": scale}]}
    )
    assert all(type(ax) is v05.Axis for ax in ms.axes)


@VERSIONS
def test_constructors_set_only_given_fields(v: ModuleType) -> None:
    assert v.SpaceAxis(name="x").model_fields_set == {"name", "type"}
    custom = v.CustomAxis(name="x").model_dump(exclude_unset=True, exclude_none=False)
    assert custom == {"name": "x"}
    # (the type isn't a default, so it's always dumped)
    assert v.SpaceAxis(name="x").model_dump(exclude_defaults=True)["type"] == "space"


@VERSIONS
def test_constructor_rejects_other_type(v: ModuleType) -> None:
    with pytest.raises(ValueError, match="Expected type 'space', got 'time'"):
        v.SpaceAxis(name="x", type="time")


@VERSIONS
def test_custom_axis_of_a_typed_type(v: ModuleType) -> None:
    # (just an axis of that type: a typed axis isn't a different class anymore)
    assert v.CustomAxis(name="y", type="space") == v.SpaceAxis(name="y")


def test_v06_constructor_fields() -> None:
    ax = v06.ChannelAxis(name="c", longName="channels", discrete=True)
    assert (ax.longName, ax.discrete) == ("channels", True)


@VERSIONS
@pytest.mark.parametrize(
    "type_, unit, warns",
    [
        ("space", "pixel", True),
        ("space", "micrometer", False),
        ("time", "fortnight", True),
        ("time", "second", False),
        # only space and time axes have recommended units
        ("channel", "pixel", False),
        ("displacement", "pixel", False),
        (None, "pixel", False),
    ],
)
def test_unit_warnings(
    v: ModuleType, type_: str | None, unit: str, warns: bool
) -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        v.Axis(name="a", type=type_, unit=unit)
    assert any(w.category is ValidationWarning for w in record) == warns


@VERSIONS
def test_unit_warning_points_at_caller(v: ModuleType) -> None:
    with pytest.warns(ValidationWarning) as record:
        v.SpaceAxis(name="x", unit="pixel")
    assert record[0].filename == __file__
