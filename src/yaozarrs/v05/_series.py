from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import _BaseModel


class Series(_BaseModel):
    version: Literal["0.5"] = "0.5"
    # NOTE
    # this one is confusing... the spec says:
    # "The OME-Zarr Metadata in the zarr.json file within the OME group MAY
    # contain the "series" key:"
    # but without it, it's simply 'version', and no longer distinguishable from
    # other OME-Zarr metadata.  So we make it required here.
    # open an issue...
    series: Annotated[list[str], MinLen(1)] = Field(
        description="An array of the same length and the same order as "
        "the images defined in the OME-XML"
    )
