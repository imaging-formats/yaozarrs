"""Units recommended by the spec for space and time axes (from UDUNITS-2)."""

import warnings
from collections.abc import Callable
from typing import Annotated, Literal, TypeAlias, get_args

from pydantic import AfterValidator

from yaozarrs._errors import ValidationWarning

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


def _warn_if_not_one_of(kind: str, valid: tuple[str, ...]) -> Callable[[str], str]:
    def _warn(v: str) -> str:
        if v not in valid:
            warnings.warn(
                f"Warning: {kind} axis unit {v!r}, SHOULD be one of {valid}",
                ValidationWarning,
                stacklevel=3,
            )
        return v

    return _warn


# these Annotated types give type hinting and IDE autocompletion,
# but still fallback to string (with a warning) for unrecognized units
SpaceUnits: TypeAlias = Annotated[
    ValidSpaceUnit | str,
    AfterValidator(_warn_if_not_one_of("Space", get_args(ValidSpaceUnit))),
]
TimeUnits: TypeAlias = Annotated[
    ValidTimeUnit | str,
    AfterValidator(_warn_if_not_one_of("Time", get_args(ValidTimeUnit))),
]
