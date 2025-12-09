"""OME-Zarr group produced by bioformats2raw to contain OME metadata."""

from typing import Annotated

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import ZarrGroupModel


class Series(ZarrGroupModel):
    """Model for the ome group that contains OME-XML metadata."""

    series: Annotated[list[str], MinLen(1)] = Field(
        description="An array of the same length and the same order as "
        "the images defined in the OME-XML"
    )
