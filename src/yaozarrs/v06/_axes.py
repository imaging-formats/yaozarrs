import warnings
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias

from annotated_types import MinLen
from pydantic import (
    Discriminator,
    Field,
    Tag,
    WrapValidator,
    field_validator,
    model_validator,
)

from yaozarrs._base import _BaseModel
from yaozarrs._common._units import SpaceUnits, TimeUnits
from yaozarrs._types import UniqueList
from yaozarrs._validation_warning import ValidationWarning

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "Axis",
    "SpaceAxis",
    "TimeAxis",
    "ChannelAxis",
    "CustomAxis",
]


class _AxisBase(_BaseModel):
    # MUST contain the field `name` ... The value MUST be a non-empty string.
    name: str = Field(min_length=1, description="The name of the axis.")
    # SHOULD contain the field `type`. (narrowed by subclasses)
    type: str | None = Field(default=None, description="The type of the axis.")
    # SHOULD contain the field `unit` (narrowed by subclasses)
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


class _TypedAxis(_AxisBase):
    @model_validator(mode="before")
    @classmethod
    def _inject_type_if_missing(cls, data: Any) -> Any:
        # allows e.g. `SpaceAxis(name="x")`, while still marking `type` as
        # explicitly set (so that it survives `model_dump(exclude_unset=True)`)
        if isinstance(data, dict) and "type" not in data:
            return {**data, "type": cls.model_fields["type"].default}
        return data


class SpaceAxis(_TypedAxis):
    """Axis with `type="space"` (units restricted to SpaceUnits)."""

    type: Literal["space"] = "space"
    unit: SpaceUnits | None = None


class TimeAxis(_TypedAxis):
    """Axis with `type="time"` (units restricted to TimeUnits)."""

    type: Literal["time"] = "time"
    unit: TimeUnits | None = None


class ChannelAxis(_TypedAxis):
    """Axis with `type="channel"`."""

    type: Literal["channel"] = "channel"


class CustomAxis(_AxisBase):
    """Axis with any other `type` (e.g. "array", "coordinate", "displacement") or none.

    The spec recommends `type` be one of "array", "space", "time", "channel",
    "coordinate", or "displacement", but it MAY take other values for custom axis types.
    """

    type: Literal["array", "coordinate", "displacement"] | str | None = None

    @field_validator("type")
    @classmethod
    def _not_a_typed_axis(cls, v: str | None) -> str | None:
        # (an axis of these types can't be used as a CustomAxis: inside an axes list,
        # the `Axis` union expects the matching typed class)
        if v in ("space", "time", "channel"):
            raise ValueError(
                f"Use {v.title()}Axis for axes of type {v!r} (CustomAxis is for other "
                "types)."
            )
        return v


def _axis_discriminator(v: Any) -> str:
    t = v.get("type") if isinstance(v, dict) else getattr(v, "type", None)
    return t if t in ("space", "time", "channel") else "custom"


Axis: TypeAlias = Annotated[
    Annotated[SpaceAxis, Tag("space")]
    | Annotated[TimeAxis, Tag("time")]
    | Annotated[ChannelAxis, Tag("channel")]
    | Annotated[CustomAxis, Tag("custom")],
    Discriminator(_axis_discriminator),
]


def _validate_axes(axes: list[Axis]) -> list[Axis]:
    """Rules for the axes of *any* coordinate system.

    NOTE: axes.schema also requires <= 5 axes and either 2-3 "space" or >= 2 "array"
    axes for *every* coordinate system, but the prose scopes those rules to coordinate
    systems inside multiscales (see `validate_multiscale_axes`).  Where the schema is
    stricter than the prose, we follow the prose.
    """
    # The values in the `name` fields MUST be unique within the same coordinate system.
    names = [ax.name for ax in axes]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(
            f"Axis names must be unique within a coordinate system. Duplicates: {dupes}"
        )

    return axes


AxesList: TypeAlias = Annotated[
    UniqueList[Axis],
    MinLen(1),
    # hack to get around ordering of multiple after validators
    WrapValidator(lambda v, h: _validate_axes(h(v))),
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
