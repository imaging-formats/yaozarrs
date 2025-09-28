from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import ZarrGroupModel, _BaseModel
from yaozarrs._types import UniqueList


class FieldOfView(_BaseModel):
    path: str = Field(
        description="The path for this field of view subgroup",
        pattern=r"^[A-Za-z0-9]+$",
    )
    acquisition: int | None = Field(
        default=None,
        description="A unique identifier within the context of the plate",
    )


class WellDef(_BaseModel):
    images: Annotated[UniqueList[FieldOfView], MinLen(1)] = Field(
        description="The fields of view for this well",
    )
    version: Literal["0.4"] = "0.4"


# ------------------------------------------------------------------------------
# Well model (top-level)
# ------------------------------------------------------------------------------


class Well(ZarrGroupModel):
    """A well at the top-level of an ome-zarr file."""

    well: WellDef
