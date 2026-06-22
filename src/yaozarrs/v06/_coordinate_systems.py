"""Coordinate-system models for OME-NGFF v0.6.

A *coordinate system* is a named set of axes. In v0.6 the `axes` array no longer
lives directly on a multiscale; instead a multiscale declares one or more named
coordinate systems, and (dataset / multiscale) transforms map *into* them by
name. The coordinate system that all dataset transforms `output` to is referred
to as the "intrinsic" coordinate system in the spec.
"""

from pydantic import Field

from yaozarrs._base import _BaseModel

from ._axes import AxesList

__all__ = ["CoordinateSystem"]


class CoordinateSystem(_BaseModel):
    """A named set of axes that transforms can map between.

    !!! note "New in v0.6"
        This object did not exist in v0.5, where `axes` lived directly on the
        multiscale. The dataset `coordinateTransformations` now `output` to a
        coordinate system referenced here by `name`.
    """

    name: str = Field(
        min_length=1,
        description="Name of the coordinate system (unique among all coordinate "
        "systems in the document).",
    )
    axes: AxesList = Field(
        description="Ordered list of axes defining this coordinate system."
    )
