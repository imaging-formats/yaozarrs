from __future__ import annotations

import re

import pytest
from pydantic import TypeAdapter, ValidationError

from yaozarrs import v06
from yaozarrs._axis import AxesList as V05AxesList
from yaozarrs._validation_warning import ValidationWarning
from yaozarrs.v06._axes import validate_multiscale_axes
from yaozarrs.v06._coordinate_systems import CoordinateSystemsList

X = {"name": "x", "type": "space", "unit": "micrometer"}
Y = {"name": "y", "type": "space", "unit": "micrometer"}
Z = {"name": "z", "type": "space", "unit": "micrometer"}
T = {"name": "t", "type": "time", "unit": "second"}
C = {"name": "c", "type": "channel", "discrete": True}
I = {"name": "i", "type": "array"}  # noqa: E741
J = {"name": "j", "type": "array"}
D = {"name": "d", "type": "displacement", "discrete": True}

AXES = TypeAdapter(list[v06.Axis])

VALID_COORDINATE_SYSTEMS = [
    {"name": "physical", "axes": [Y, X]},
    {"name": "physical", "axes": [T, C, Z, Y, X]},
    {"name": "array", "axes": [I, J]},
    {"name": "with_extras", "axes": [{**Y, "longName": "rows"}, X]},
    # the schema only requires 2-3 space axes, not ordering (multiscales-only)
    {"name": "unordered", "axes": [X, Y, T]},
    # untyped/custom axes are fine alongside 2-3 space axes
    {"name": "custom", "axes": [{"name": "angle", "type": "angle"}, Y, X]},
    # the schema's axis-count rules only apply inside multiscales per the prose
    {"name": "one_axis", "axes": [X]},
    {"name": "untyped", "axes": [{"name": "a"}, {"name": "b"}]},
    {"name": "four_space", "axes": [{**X, "name": "w"}, Z, Y, X]},
    {"name": "space_and_array", "axes": [I, J, Y, X]},
    {"name": "six_axes", "axes": [T, C, Z, Y, X, I]},
]


@pytest.mark.parametrize("obj", VALID_COORDINATE_SYSTEMS)
def test_valid_coordinate_systems(obj: dict) -> None:
    cs = v06.CoordinateSystem.model_validate(obj)
    assert cs.ndim == len(obj["axes"])
    again = v06.CoordinateSystem.model_validate_json(cs.model_dump_json())
    assert again == cs


INVALID_COORDINATE_SYSTEMS = [
    ({"name": "", "axes": [Y, X]}, "at least 1 character"),
    ({"name": "cs", "axes": [{**Y, "name": ""}, X]}, "at least 1 character"),
    # "The values in the `name` fields MUST be unique within the same coordinate
    # system" (differing units don't make them distinct)
    ({"name": "cs", "axes": [{**X, "unit": "nanometer"}, X]}, "Duplicates: ['x']"),
    ({"name": "cs", "axes": []}, "at least 1 item"),
    ({"name": "cs", "axes": [{**X, "discrete": "maybe"}, Y]}, "valid boolean"),
]


@pytest.mark.parametrize("obj, msg", INVALID_COORDINATE_SYSTEMS)
def test_invalid_coordinate_systems(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(msg)):
        v06.CoordinateSystem.model_validate(obj)


def test_axis_discrimination() -> None:
    axes = AXES.validate_python([T, C, Y, X, I, D, {"name": "foo"}])
    types = [type(ax) for ax in axes]
    assert types == [
        v06.TimeAxis,
        v06.ChannelAxis,
        v06.SpaceAxis,
        v06.SpaceAxis,
        v06.CustomAxis,
        v06.CustomAxis,
        v06.CustomAxis,
    ]
    assert axes[5].discrete is True


def test_typed_axis_type_survives_exclude_unset() -> None:
    ax = v06.SpaceAxis(name="x")
    assert ax.model_dump(exclude_unset=True) == {"name": "x", "type": "space"}
    fields = list(ax.model_dump(exclude_none=False))
    assert fields == ["name", "type", "unit", "longName", "discrete"]


def test_unit_warnings() -> None:
    with pytest.warns(ValidationWarning, match="SHOULD be one of"):
        v06.SpaceAxis(name="x", unit="pixel")
    with pytest.warns(ValidationWarning, match="SHOULD be one of"):
        v06.TimeAxis(name="t", unit="fortnight")
    # custom axes accept any unit without warning
    v06.CustomAxis(name="d", type="displacement", unit="pixel")


def test_coordinate_system_names_unique() -> None:
    adapter = TypeAdapter(CoordinateSystemsList)
    cs = {"name": "a", "axes": [Y, X]}
    adapter.validate_python([cs, {**cs, "name": "b"}])
    with pytest.raises(ValidationError, match=re.escape("Duplicates: ['a']")):
        adapter.validate_python([cs, {**cs, "axes": [Z, Y, X]}])


VALID_MULTISCALE_AXES = [
    [Y, X],
    [Z, Y, X],
    [T, Y, X],
    [C, Y, X],
    [T, C, Z, Y, X],
    [D, Y, X],  # displacement axis sits in the channel/custom slot
    [{"name": "q"}, Y, X],  # untyped axis also sits in that slot
    # several channel/custom axes are fine (like v05), within the 5-axis limit
    [C, D, Y, X],
    [{"name": "view", "type": "view"}, {"name": "phase"}, Y, X],
]


@pytest.mark.parametrize("axes", VALID_MULTISCALE_AXES)
def test_valid_multiscale_axes(axes: list[dict]) -> None:
    validate_multiscale_axes(AXES.validate_python(axes))


INVALID_MULTISCALE_AXES = [
    ([X], "between 2 and 5 axes"),
    ([T, C, Z, Y, X, {"name": "q"}], "between 2 and 5 axes"),
    ([I, X], "2 or 3 axes of type 'space'"),
    ([{**X, "name": "w"}, Z, Y, X], "2 or 3 axes of type 'space'"),
    ([T, {**T, "name": "t2"}, Y, X], "at most 1 axis of type 'time'"),
    ([C, {**C, "name": "c2"}, Y, X], "at most 1 axis of type 'channel'"),
    ([Y, X, T], "required order"),
    ([C, T, Y, X], "required order"),
    ([Y, C, X], "required order"),
]


@pytest.mark.parametrize("axes, msg", INVALID_MULTISCALE_AXES)
def test_invalid_multiscale_axes(axes: list[dict], msg: str) -> None:
    with pytest.raises(ValueError, match=re.escape(msg)):
        validate_multiscale_axes(AXES.validate_python(axes))


def test_multiscale_array_axes_warn() -> None:
    # prose: 2-3 space axes; axes.schema: >= 2 array axes are allowed instead
    with pytest.warns(ValidationWarning, match="'array' axes are only allowed"):
        validate_multiscale_axes(AXES.validate_python([I, J]))


@pytest.mark.parametrize(
    "axes, msg",
    [
        # only the space-axis rule is relaxed for array coordinate systems
        ([{"name": f"a{i}", "type": "array"} for i in range(6)], "between 2 and 5"),
        ([T, {**T, "name": "t2"}, I, J], "at most 1 axis of type 'time'"),
        ([I, T, J], "required order"),
    ],
)
def test_multiscale_array_axes_other_rules(axes: list[dict], msg: str) -> None:
    with pytest.warns(ValidationWarning), pytest.raises(ValueError, match=msg):
        validate_multiscale_axes(AXES.validate_python(axes))


@pytest.mark.parametrize("type_", ["space", "time", "channel"])
def test_custom_axis_rejects_known_types(type_: str) -> None:
    with pytest.raises(ValidationError, match=f"Use {type_.title()}Axis"):
        v06.CustomAxis(name="a", type=type_)


PARITY_AXES = [
    [Y, X],
    [T, C, Z, Y, X],
    [C, {"name": "view", "type": "view"}, Y, X],
    [{"name": "view", "type": "view"}, {"name": "phase"}, {"name": "i"}, Y, X],
    [T, C, {"name": "view", "type": "view"}, Y, X],
    [C, {**C, "name": "c2"}, Y, X],
    [T, {**T, "name": "t2"}, Y, X],
    [X],
    [{**X, "name": "w"}, Z, Y, X],
    [T, C, {"name": "q"}, Z, Y, X],
    [Y, X, T],
    [C, T, Y, X],
]


@pytest.mark.parametrize("axes", PARITY_AXES)
def test_multiscale_axes_rules_match_v05(axes: list[dict]) -> None:
    # the multiscale axes rules are worded identically in v0.5 and v0.6, so v06 must
    # accept/reject the same axes as v05 (decisions.md, D23)
    try:
        TypeAdapter(V05AxesList).validate_python(axes)
        v05_ok = True
    except ValidationError:
        v05_ok = False
    try:
        validate_multiscale_axes(AXES.validate_python(axes))
        v06_ok = True
    except (ValueError, ValidationError):
        v06_ok = False
    assert v05_ok == v06_ok
