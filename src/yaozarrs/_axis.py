from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from annotated_types import Len
from pydantic import Discriminator, Field, Tag, WrapValidator, model_validator

from ._base import _BaseModel
from ._types import UniqueList

SpaceUnits: TypeAlias = Literal[
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

TimeUnits: TypeAlias = Literal[
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
    name: str = Field(description="The name of the axis.")


# these three classes allow us to:
# 1. have "type" be used for discrimination when parsing Axis union types, falling
#    back to CustomAxis when "type" is missing or unrecognized
# 2. have units validated in a type-specific way
# 3. instantiate SpaceAxis, TimeAxis, ChannelAxis without specifying "type" at runtime
#    (even though it's required in the schema)


class CustomAxis(_AxisBase):
    type: str | None = None  # SHOULD
    unit: str | None = None  # SHOULD


class SpaceAxis(_AxisBase):
    if TYPE_CHECKING:
        type: Literal["space"] = "space"
    else:
        type: Literal["space"]
    unit: SpaceUnits | None = None  # SHOULD

    @model_validator(mode="before")
    @classmethod
    def _inject_type_if_missing(cls, v: Any) -> Any:
        if isinstance(v, dict) and "type" not in v:
            v["type"] = "space"
        return v


class TimeAxis(_AxisBase):
    if TYPE_CHECKING:
        type: Literal["time"] = "time"
    else:
        type: Literal["time"]
    unit: TimeUnits | None = None  # SHOULD

    @model_validator(mode="before")
    @classmethod
    def _inject_type_if_missing(cls, v: Any) -> Any:
        if isinstance(v, dict) and "type" not in v:
            v["type"] = "time"
        return v


class ChannelAxis(_AxisBase):
    if TYPE_CHECKING:
        type: Literal["channel"] = "channel"
    else:
        type: Literal["channel"]
    unit: str | None = None  # SHOULD

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
    type_order = {"time": 0, "channel": 1, None: 1, "space": 2}
    sorted_axes = sorted(axes, key=lambda ax: type_order.get(ax.type, 3))
    if axes != sorted_axes:
        raise ValueError(
            "Axes are not in the required order by type. "
            "Order must be [time,] [channel,] space."
        )
    return axes


AxesList: TypeAlias = Annotated[
    UniqueList[Axis],
    Len(min_length=2, max_length=5),
    # hack to get around ordering of multiple after validators
    WrapValidator(lambda v, h: _validate_axes_list(h(v))),
]
