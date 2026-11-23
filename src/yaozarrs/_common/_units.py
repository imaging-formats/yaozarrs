"""Units recommended by the spec for space and time axes (from UDUNITS-2)."""

import sys
from typing import Literal, TypeAlias, get_args

from yaozarrs._errors import warn

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


_UNITS = {
    "space": ("Space", get_args(ValidSpaceUnit)),
    "time": ("Time", get_args(ValidTimeUnit)),
}


def warn_if_unexpected_unit(axis_type: str | None, unit: str | None) -> None:
    """Warn if `unit` isn't one the spec recommends for an axis of `axis_type`."""
    if unit is None or (expected := _UNITS.get(axis_type or "")) is None:
        return
    kind, valid = expected
    if unit not in valid:
        warn(
            f"Warning: {kind} axis unit {unit!r}, SHOULD be one of {valid}",
            stacklevel=_external_stacklevel(),
        )


def _external_stacklevel() -> int:
    """The `stacklevel` (for our caller) of the first frame outside yaozarrs/pydantic.

    (the axis validators run several frames deep, inside pydantic: this points the
    warning at the user's code instead)
    """
    level, frame = 1, sys._getframe(1)
    while frame and frame.f_globals.get("__name__", "").partition(".")[0] in (
        "yaozarrs",
        "pydantic",
    ):
        level, frame = level + 1, frame.f_back
    return level
