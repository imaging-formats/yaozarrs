"""Axis models for OME-NGFF v0.6.

In v0.6 the `axes` array no longer lives directly on a multiscale. Instead it is
nested inside a *coordinate system*
(see [`CoordinateSystem`][yaozarrs.v06.CoordinateSystem]).

Changes from v0.5 (per `axes.schema`):

- Only `name` is *required* (v0.5 effectively required `type` too for known types).
- `type` is a free-form string (the schema dropped the `space`/`time`/`channel`
  enum). v0.6 introduces additional types such as `array` (used for array /
  displacement-field coordinate systems).
- Two new optional fields: `longName` and `discrete`.
- `minItems` is now 1 (was 2); the 2-3 space-axis constraint is expressed via a
  `oneOf`/`contains` rule (which also permits >=2 `array` axes as an alternative).
"""

import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from annotated_types import Len
from pydantic import (
    AfterValidator,
    Discriminator,
    Field,
    Tag,
    WrapValidator,
    model_validator,
)
from typing_extensions import get_args

from yaozarrs._base import _BaseModel
from yaozarrs._types import UniqueList
from yaozarrs._validation_warning import ValidationWarning

ValidSpaceUnit: TypeAlias = Literal[
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]

ValidTimeUnit: TypeAlias = Literal[
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
]


class _AxisBase(_BaseModel):
    name: str = Field(description="The name of the axis.", min_length=1)
    # NOTE (v0.6): `longName` and `discrete` are new optional axis fields.
    longName: str | None = Field(
        default=None, description="Longer name or description of the axis."
    )
    discrete: bool | None = Field(
        default=None, description="Whether the dimension is discrete."
    )


# these classes allow us to:
# 1. have "type" be used for discrimination when parsing Axis union types, falling
#    back to CustomAxis when "type" is missing or unrecognized
# 2. have units validated in a type-specific way
# 3. instantiate SpaceAxis, TimeAxis, ChannelAxis without specifying "type" at runtime
#    (even though it's recommended in the schema)

_VALID_SPACE_UNITS = get_args(ValidSpaceUnit)
_VALID_TIME_UNITS = get_args(ValidTimeUnit)


def _warn_if_not_space_unit(v: str) -> str:
    if v not in _VALID_SPACE_UNITS:
        warnings.warn(
            f"Warning: Space axis unit {v!r}, SHOULD be one of {_VALID_SPACE_UNITS}",
            ValidationWarning,
            stacklevel=3,
        )
    return v


def _warn_if_not_time_unit(v: str) -> str:
    if v not in _VALID_TIME_UNITS:
        warnings.warn(
            f"Warning: Time axis unit {v!r}, SHOULD be one of {_VALID_TIME_UNITS}",
            ValidationWarning,
            stacklevel=3,
        )
    return v


# these Annotated types give type hinting and IDE autocompletion,
# but still fallback to string (with a warning) for unrecognized units
SpaceUnits: TypeAlias = Annotated[
    ValidSpaceUnit | str, AfterValidator(_warn_if_not_space_unit)
]
TimeUnits: TypeAlias = Annotated[
    ValidTimeUnit | str, AfterValidator(_warn_if_not_time_unit)
]


class CustomAxis(_AxisBase):
    """Axis with an arbitrary / unrecognized `type` (e.g. `array`), and any unit.

    This is the fallback for any axis whose `type` is not one of the
    well-known `space`/`time`/`channel` values. v0.6 uses `type="array"` axes
    to describe array (index) coordinate systems for displacement fields.
    """

    type: str | None = None  # free-form in v0.6
    unit: str | None = None


class SpaceAxis(_AxisBase):
    """Axis with `type="space"` (units restricted to SpaceUnits)."""

    if TYPE_CHECKING:
        type: Literal["space"] = "space"
    else:
        type: Literal["space"]
    unit: SpaceUnits | None = None

    @model_validator(mode="before")
    @classmethod
    def _inject_type_if_missing(cls, v: Any) -> Any:
        if isinstance(v, dict) and "type" not in v:
            v["type"] = "space"
        return v


class TimeAxis(_AxisBase):
    """Axis with `type="time"` (units restricted to TimeUnits)."""

    if TYPE_CHECKING:
        type: Literal["time"] = "time"
    else:
        type: Literal["time"]
    unit: TimeUnits | None = None

    @model_validator(mode="before")
    @classmethod
    def _inject_type_if_missing(cls, v: Any) -> Any:
        if isinstance(v, dict) and "type" not in v:
            v["type"] = "time"
        return v


class ChannelAxis(_AxisBase):
    """Axis with `type="channel"`."""

    if TYPE_CHECKING:
        type: Literal["channel"] = "channel"
    else:
        type: Literal["channel"]
    unit: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _inject_type_if_missing(cls, v: Any) -> Any:
        if isinstance(v, dict) and "type" not in v:
            v["type"] = "channel"
        return v


def _axis_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        t = v.get("type")
    else:
        t = getattr(v, "type", None)

    if t in ("space", "time", "channel"):
        return t
    return "custom"


Axis: TypeAlias = Annotated[
    Annotated[SpaceAxis, Tag("space")]
    | Annotated[TimeAxis, Tag("time")]
    | Annotated[ChannelAxis, Tag("channel")]
    | Annotated[CustomAxis, Tag("custom")],
    Discriminator(_axis_discriminator),
]


def _validate_axes_list(axes: list[Axis]) -> list[Axis]:
    """Validate a list of Axis for a `CoordinateSystem.axes`.

    Enforces the `axes.schema` `oneOf`: an axes array MUST be *either* a physical
    coordinate system with 2-3 `space` axes, *or* an array coordinate system with
    >=2 `array` axes (exactly one branch). This is a structural schema rule and
    applies to *every* coordinate system (multiscale or scene), so a fully
    type-less axes array is invalid.

    !!! note
        The prose ordering/count rules (<=1 time, <=1 channel/custom, and
        time->channel/custom->space ordering) are scoped by index.md to
        "coordinate systems inside multiscales metadata" -- they are enforced
        separately by
        [`validate_multiscale_axes_ordering`][yaozarrs.v06._axes.validate_multiscale_axes_ordering],
        called from `Multiscale._post_validate`, not here. A scene-level
        coordinate system (e.g. axes ordered `[x, y, t]`) is not subject to them.
    """
    # names MUST be unique within the (coordinate system's) list.
    names = [ax.name for ax in axes]
    if len(names) != len(set(names)):
        raise ValueError(f"Axis names must be unique. Found duplicates in {names}")

    types = [getattr(ax, "type", None) for ax in axes]
    n_space = types.count("space")
    n_array = types.count("array")

    # axes.schema oneOf: EXACTLY one of (2-3 space) / (>=2 array) must hold.
    space_branch = 2 <= n_space <= 3
    array_branch = n_array >= 2
    if space_branch == array_branch:  # both or neither -> oneOf violated
        raise ValueError(
            "An axes array must contain either 2-3 axes of type 'space' or "
            f"at least 2 axes of type 'array' (got {n_space} space, {n_array} array)."
        )
    return axes


def _axis_order_class(ax: Axis) -> int:
    t = getattr(ax, "type", None)
    if t == "time":
        return 0
    if t in ("displacement", "coordinate"):
        # spec (coordinates/displacements): the vector-field axis is inserted
        # after a time axis (if present) and before the spatial axes, and MUST
        # NOT coincide with a channel/custom axis -- i.e. it is a distinct slot
        # from "channel or custom", not folded into the same <=1 count.
        return 2
    if t == "space":
        return 3
    return 1  # channel, or a null/custom type


def validate_multiscale_axes_ordering(axes: list[Axis]) -> None:
    """Enforce the multiscales-scoped axis ordering/count prose rules.

    index.md restricts these rules to "coordinate systems inside multiscales
    metadata" (this also covers the multiscale groups that back a
    `displacements`/`coordinates` vector field, per index.md's `coordinates and
    displacements` section). Callers (`Multiscale._post_validate`) apply this to
    each of a multiscale's declared coordinate systems; it is intentionally NOT
    applied to scene-level coordinate systems by `_validate_axes_list` above.
    """
    types = [getattr(ax, "type", None) for ax in axes]
    if types.count("array") >= 2:
        return  # array coordinate systems: these ordering rules don't apply

    if types.count("time") > 1:
        raise ValueError("There can be at most 1 axis of type 'time'.")
    if (
        sum(1 for t in types if t not in ("time", "space"))
        - types.count("displacement")
        - types.count("coordinate")
        > 1
    ):
        raise ValueError(
            "There can be at most 1 axis of type 'channel' or a null/custom type."
        )
    if (n_vec := types.count("displacement") + types.count("coordinate")) > 1:
        raise ValueError(
            f"There can be at most 1 axis of type 'displacement'/'coordinate', "
            f"got {n_vec}."
        )

    if axes != sorted(axes, key=_axis_order_class):
        raise ValueError(
            "Axes are not in the required order by type. Order must be "
            "[time,] [channel/custom,] [displacement/coordinate,] space."
        )


AxesList: TypeAlias = Annotated[
    UniqueList[Axis],
    # v0.6 axes.schema: minItems 1, maxItems 5 (the `oneOf` in _validate_axes_list
    # imposes an effective minimum of 2).
    Len(min_length=1, max_length=5),
    # hack to get around ordering of multiple after validators
    WrapValidator(lambda v, h: _validate_axes_list(h(v))),
]
