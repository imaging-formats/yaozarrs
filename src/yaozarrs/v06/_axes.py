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

    !!! note "Pragmatic, type-aware validation"
        v0.6 makes the axis `type` *optional* (and RFC-3 / ngff-spec PR #75 moves
        toward fully "unconstrained" axes). Real v0.6 spec examples include axes
        with no `type` at all (e.g. `{"name": "x", "unit": "micrometer"}`). So we
        only apply the v0.5-style structural rules (2-3 space axes, <=1 time, <=1
        channel, ordering) when *every* axis carries a recognized
        space/time/channel type. See `TRICKY_NOTES.md`.
    """
    # names MUST be unique within the (coordinate system's) list.
    names = [ax.name for ax in axes]
    if len(names) != len(set(names)):
        raise ValueError(f"Axis names must be unique. Found duplicates in {names}")

    types = [getattr(ax, "type", None) for ax in axes]

    # The axes.schema `oneOf` allows EITHER a "physical" coordinate system with
    # 2-3 space axes, OR an "array" coordinate system with >=2 `array` axes.
    if any(t == "array" for t in types):
        if sum(t == "array" for t in types) < 2:
            raise ValueError("An 'array' coordinate system must have >=2 array axes.")
        return axes

    # Only enforce the (v0.5-style) space/time/channel structural rules when every
    # axis carries a recognized type; otherwise we cannot reliably apply them and
    # we accept the axes (matching the permissive spec examples).
    known = {"space", "time", "channel"}
    if not all(t in known for t in types):
        return axes

    n_space_axes = types.count("space")
    if n_space_axes < 2 or n_space_axes > 3:
        raise ValueError("There must be 2 or 3 axes of type 'space'.")
    if types.count("time") > 1:
        raise ValueError("There can be at most 1 axis of type 'time'.")
    if types.count("channel") > 1:
        raise ValueError("There can be at most 1 axis of type 'channel'.")

    # The entries SHOULD be ordered by "type" where the "time" axis must come first
    # (if present), followed by the "channel" axis (if present) and the space axes.
    # NOTE: pending RFC-3 ("Unconstrained Axes", ngff-spec PR #75) may relax this.
    type_order = {"time": 0, "channel": 1, "space": 2}
    sorted_axes = sorted(axes, key=lambda ax: type_order.get(ax.type or "", 3))
    if axes != sorted_axes:
        raise ValueError(
            "Axes are not in the required order by type. "
            "Order must be [time,] [channel,] space."
        )
    return axes


AxesList: TypeAlias = Annotated[
    UniqueList[Axis],
    # v0.6 axes.schema: minItems 1, maxItems 5
    Len(min_length=1, max_length=5),
    # hack to get around ordering of multiple after validators
    WrapValidator(lambda v, h: _validate_axes_list(h(v))),
]
