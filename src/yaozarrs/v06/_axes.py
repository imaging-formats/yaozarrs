import warnings
from collections.abc import Sequence
from typing import Annotated, Literal, Self, TypeAlias

from annotated_types import MinLen
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import deprecated

from yaozarrs._base import _BaseModel
from yaozarrs._common._axis import build_axis
from yaozarrs._common._units import (
    ValidSpaceUnit,
    ValidTimeUnit,
    warn_if_unexpected_unit,
)
from yaozarrs._errors import ValidationWarning
from yaozarrs._types import UNIQUE_ITEMS, duplicates

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "Axis",
    "SpaceAxis",
    "TimeAxis",
    "ChannelAxis",
    "CustomAxis",
]

# the axis types the spec recommends (it MAY take other values, for custom types)
AxisType: TypeAlias = Literal[
    "array", "space", "time", "channel", "coordinate", "displacement"
]


class Axis(_BaseModel):
    """An axis of a coordinate system.

    `SpaceAxis`, `TimeAxis` and `ChannelAxis` build one with the `type` filled in
    (and, for IDEs, the units the spec recommends for that type).
    """

    # MUST contain the field `name` ... The value MUST be a non-empty string.
    name: str = Field(min_length=1, description="The name of the axis.")
    # SHOULD contain the field `type`.
    type: AxisType | str | None = Field(
        default=None, description="The type of the axis."
    )
    # SHOULD contain the field `unit`
    unit: str | None = Field(default=None, description="Physical unit of the axis.")
    # MAY contain the field `longName`. The value MUST be a string
    longName: str | None = Field(
        default=None, description="A longer name or description of the axis."
    )
    # MAY contain the field `discrete`. The value MUST be a boolean
    discrete: bool | None = Field(
        default=None,
        description=(
            "Whether the axis is discrete (may only be indexed by integers, e.g. "
            "channels) rather than continuous (interpolation is well-defined)."
        ),
    )

    @model_validator(mode="after")
    def _warn_unexpected_unit(self) -> Self:
        warn_if_unexpected_unit(self.type, self.unit)
        return self


def SpaceAxis(
    *,
    name: str,
    type: Literal["space"] = "space",
    unit: ValidSpaceUnit | str | None = None,
    longName: str | None = None,
    discrete: bool | None = None,
) -> Axis:
    """An `Axis` with `type="space"`."""
    fields = {"type": type, "unit": unit, "longName": longName, "discrete": discrete}
    return build_axis(Axis, "space", name=name, **fields)


def TimeAxis(
    *,
    name: str,
    type: Literal["time"] = "time",
    unit: ValidTimeUnit | str | None = None,
    longName: str | None = None,
    discrete: bool | None = None,
) -> Axis:
    """An `Axis` with `type="time"`."""
    fields = {"type": type, "unit": unit, "longName": longName, "discrete": discrete}
    return build_axis(Axis, "time", name=name, **fields)


def ChannelAxis(
    *,
    name: str,
    type: Literal["channel"] = "channel",
    unit: str | None = None,
    longName: str | None = None,
    discrete: bool | None = None,
) -> Axis:
    """An `Axis` with `type="channel"`."""
    fields = {"type": type, "unit": unit, "longName": longName, "discrete": discrete}
    return build_axis(Axis, "channel", name=name, **fields)


@deprecated("`CustomAxis` is deprecated: use `Axis` instead (it's the same).")
def CustomAxis(
    *,
    name: str,
    type: Literal["array", "coordinate", "displacement"] | str | None = None,
    unit: str | None = None,
    longName: str | None = None,
    discrete: bool | None = None,
) -> Axis:
    """Deprecated: the same as [`Axis`][yaozarrs.v06.Axis] (use that instead)."""
    fields = {"type": type, "unit": unit, "longName": longName, "discrete": discrete}
    return build_axis(Axis, None, name=name, **fields)


def _validate_axes(axes: list[Axis]) -> list[Axis]:
    """Rules for the axes of *any* coordinate system.

    NOTE: axes.schema also requires <= 5 axes and either 2-3 "space" or >= 2 "array"
    axes for *every* coordinate system, but the prose scopes those rules to coordinate
    systems inside multiscales (see `validate_multiscale_axes`).  Where the schema is
    stricter than the prose, we follow the prose.
    """
    # The values in the `name` fields MUST be unique within the same coordinate system.
    if dupes := duplicates([ax.name for ax in axes]):
        raise ValueError(
            f"Axis names must be unique within a coordinate system. Duplicates: {dupes}"
        )

    return axes


AxesList: TypeAlias = Annotated[
    list[Axis],
    UNIQUE_ITEMS,  # (implied by the unique names)
    MinLen(1),
    AfterValidator(_validate_axes),
]


def validate_multiscale_axes(axes: Sequence[Axis]) -> None:
    """Rules for the axes of coordinate systems inside `multiscales` metadata.

    Raises
    ------
    ValueError
        If the axes violate any of the rules.
    """
    # NOTE: the prose says these rules "apply to all coordinate systems inside
    # multiscales metadata", including "`axes` MUST contain 2 or 3 entries of
    # `type:space`".  But axes.schema allows either 2-3 "space" axes or >= 2 "array"
    # axes, so we accept the latter with a warning.
    # (only the space-axis rule is relaxed for those; the other rules still apply)
    n_space = sum(ax.type == "space" for ax in axes)
    array_cs = sum(ax.type == "array" for ax in axes) >= 2 and not 2 <= n_space <= 3
    if array_cs:
        warnings.warn(
            "Coordinate systems inside multiscales MUST contain 2 or 3 axes of type "
            "'space' according to the spec prose; coordinate systems made of 'array' "
            "axes are only allowed by the JSON schema.",
            ValidationWarning,
            stacklevel=2,
        )

    # The length of `axes` must be between 2 and 5
    if not 2 <= len(axes) <= 5:
        raise ValueError(f"There must be between 2 and 5 axes. Found {len(axes)}.")

    # `axes` MUST contain 2 or 3 entries of `type:space`
    if not array_cs and not 2 <= n_space <= 3:
        raise ValueError("There must be 2 or 3 axes of type 'space'.")
    # `axes` MAY contain one additional entry of `type:time`
    if sum(ax.type == "time" for ax in axes) > 1:
        raise ValueError("There can be at most 1 axis of type 'time'.")
    # `axes` MAY contain one additional entry of `type:channel` or a null / custom type.
    # NOTE: this sentence is unchanged from v0.5; like v05 we only limit `channel` axes
    # to one, and allow several custom/untyped axes (within the 5-axis limit).
    if sum(ax.type == "channel" for ax in axes) > 1:
        raise ValueError("There can be at most 1 axis of type 'channel'.")

    # `axes` entries MUST be ordered by `type` where the `time` axis must come first
    # (if present), followed by the `channel` or custom axis (if present) and the
    # axes of type `space`.
    type_order = {"time": 0, "space": 2}
    order = [type_order.get(ax.type or "", 1) for ax in axes]
    if order != sorted(order):
        raise ValueError(
            "Axes are not in the required order by type. "
            "Order must be [time,] [channel or custom,] space."
        )
