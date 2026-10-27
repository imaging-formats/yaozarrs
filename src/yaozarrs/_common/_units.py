"""Units recommended by the spec for space and time axes (from UDUNITS-2)."""

import warnings
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator
from typing_extensions import get_args

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
