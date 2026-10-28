from typing import Annotated, TypeAlias

from pydantic import Field, WrapValidator

from yaozarrs._base import _BaseModel
from yaozarrs._types import UniqueList

from ._axes import AxesList

__all__ = ["CoordinateSystem"]


class CoordinateSystem(_BaseModel):
    """A named set of axes in which coordinates (e.g. physical positions) are expressed.

    Coordinate transformations map points between coordinate systems, referring to
    them by `name`.
    """

    # MUST contain the field `name`. The value MUST be a non-empty string ...
    name: str = Field(min_length=1, description="Name of the coordinate system.")
    # MUST contain the field `axes`, whose value is an array of valid `axes`
    axes: AxesList = Field(description="The axes of this coordinate system.")

    @property
    def ndim(self) -> int:
        """Number of axes (dimensions) in this coordinate system."""
        return len(self.axes)


def _validate_names(systems: list[CoordinateSystem]) -> list[CoordinateSystem]:
    # [name] MUST be a non-empty string that is unique among all entries in a
    # `coordinateSystems` array.
    names = [cs.name for cs in systems]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"Coordinate system names must be unique. Duplicates: {dupes}")
    return systems


CoordinateSystemsList: TypeAlias = Annotated[
    UniqueList[CoordinateSystem],
    # hack to get around ordering of multiple after validators
    WrapValidator(lambda v, h: _validate_names(h(v))),
]
