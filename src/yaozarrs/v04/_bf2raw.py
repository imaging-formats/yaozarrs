from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import ZarrGroupModel


class Bf2Raw(ZarrGroupModel):
    bioformats2raw_layout: Literal[3] = Field(
        alias="bioformats2raw.layout",
        description="The top-level identifier metadata added by bioformats2raw",
    )


class Series(ZarrGroupModel):
    """Model for the ome group that contains OME-XML metadata."""

    series: Annotated[list[str], MinLen(1)] = Field(
        description="An array of the same length and the same order as "
        "the images defined in the OME-XML"
    )
