"""Axis building blocks (identical in v0.4 and v0.5)."""

from typing import Annotated, Any, Literal, Self, TypeAlias, TypeVar

from annotated_types import Len
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import deprecated

from yaozarrs._base import _BaseModel
from yaozarrs._types import UNIQUE_ITEMS

from ._units import ValidSpaceUnit, ValidTimeUnit, warn_if_unexpected_unit

_AxisT = TypeVar("_AxisT", bound=_BaseModel)


class Axis(_BaseModel):
    """A dimension (axis) of a physical coordinate space.

    [`SpaceAxis`][], [`TimeAxis`][] and [`ChannelAxis`][] build one with the `type`
    filled in (`SpaceAxis` and `TimeAxis` also hint, for IDEs, the units the spec
    recommends for space and time axes).
    """

    name: str = Field(description="The name of the axis.")
    type: Literal["space", "time", "channel"] | str | None = None  # SHOULD
    unit: str | None = None  # SHOULD

    @model_validator(mode="after")
    def _warn_unexpected_unit(self) -> Self:
        warn_if_unexpected_unit(self.type, self.unit)
        return self


def build_axis(cls: type[_AxisT], fixed_type: str | None, **fields: Any) -> _AxisT:
    """Build `cls` from the `fields` that aren't None (checking a `fixed_type`).

    (only those, so that they're the fields set: `exclude_unset` dumps don't gain
    `None`s)
    """
    if fixed_type is not None and fields["type"] != fixed_type:
        raise ValueError(f"Expected type {fixed_type!r}, got {fields['type']!r}")
    return cls(**{k: v for k, v in fields.items() if v is not None})


def SpaceAxis(
    *,
    name: str,
    type: Literal["space"] = "space",
    unit: ValidSpaceUnit | str | None = None,
) -> Axis:
    """An `Axis` with `type="space"`."""
    return build_axis(Axis, "space", name=name, type=type, unit=unit)


def TimeAxis(
    *,
    name: str,
    type: Literal["time"] = "time",
    unit: ValidTimeUnit | str | None = None,
) -> Axis:
    """An `Axis` with `type="time"`."""
    return build_axis(Axis, "time", name=name, type=type, unit=unit)


def ChannelAxis(
    *, name: str, type: Literal["channel"] = "channel", unit: str | None = None
) -> Axis:
    """An `Axis` with `type="channel"`."""
    return build_axis(Axis, "channel", name=name, type=type, unit=unit)


@deprecated("`CustomAxis` is deprecated: use `Axis` instead (it's the same).")
def CustomAxis(*, name: str, type: str | None = None, unit: str | None = None) -> Axis:
    """Deprecated: the same as [`Axis`][yaozarrs.v05.Axis] (use that instead)."""
    return build_axis(Axis, None, name=name, type=type, unit=unit)


def _validate_axes_list(axes: list[Axis]) -> list[Axis]:
    """Validate a list of Axis for `Multiscale.axes`."""
    # names MUST be unique within the list.
    names = [ax.name for ax in axes]
    if len(names) != len(set(names)):
        raise ValueError(f"Axis names must be unique. Found duplicates in {names}")

    # The "axes" MUST contain 2 or 3 entries of "type:space"
    # and MAY contain one additional entry of "type:time"
    # and MAY contain one additional entry of "type:channel" or a null / custom type.
    n_space_axes = len([ax for ax in axes if ax.type == "space"])
    if n_space_axes < 2 or n_space_axes > 3:
        raise ValueError("There must be 2 or 3 axes of type 'space'.")
    if len([ax for ax in axes if ax.type == "time"]) > 1:
        raise ValueError("There can be at most 1 axis of type 'time'.")
    if len([ax for ax in axes if ax.type == "channel"]) > 1:
        raise ValueError("There can be at most 1 axis of type 'channel'.")

    # The entries MUST be ordered by "type" where the "time" axis must come first (if
    # present), followed by the "channel" or custom axis (if present) and the axes of
    # type "space".
    # custom/null types are ordered alongside "channel" (after time, before space)
    type_order = {"time": 0, "channel": 1, "space": 2}
    sorted_axes = sorted(axes, key=lambda ax: type_order.get(ax.type or "", 1))
    if axes != sorted_axes:
        raise ValueError(
            "Axes are not in the required order by type. "
            "Order must be [time,] [channel,] space."
        )
    return axes


AxesList: TypeAlias = Annotated[
    list[Axis],
    UNIQUE_ITEMS,  # (implied by the unique names)
    Len(min_length=2, max_length=5),
    AfterValidator(_validate_axes_list),
]
