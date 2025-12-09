from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import _BaseModel
from yaozarrs._types import UniqueList


class FieldOfView(_BaseModel):
    path: str = Field(
        description="The path for this field of view subgroup.",
        pattern=r"^[A-Za-z0-9]+$",
    )
    # If multiple acquisitions were performed in the plate, it MUST contain an
    # acquisition key whose value MUST be an integer identifying the acquisition which
    # MUST match one of the acquisition JSON objects defined in the plate metadata
    acquisition: int | None = Field(
        default=None,
        description="A unique identifier within the context of the plate.",
    )


class WellDef(_BaseModel):
    images: Annotated[UniqueList[FieldOfView], MinLen(1)] = Field(
        description="The fields of view for this well",
    )


# ------------------------------------------------------------------------------
# Well model (top-level)
# ------------------------------------------------------------------------------


class Well(_BaseModel):
    """A well at the top-level of an ome-zarr file."""

    version: Literal["0.5"] = "0.5"
    well: WellDef
