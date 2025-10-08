from typing import Annotated, Literal

from annotated_types import Interval, Len, MinLen
from pydantic import Field

from yaozarrs._base import _BaseModel
from yaozarrs._types import UniqueList

from ._image import Image

# ------------------------------------------------------------------------------
# Color model
# ------------------------------------------------------------------------------

Int8bit = Annotated[int, Interval(ge=0, le=255)]


class LabelColor(_BaseModel):
    label_value: float = Field(
        description="The value of the label",
        alias="label-value",
    )
    rgba: Annotated[list[Int8bit], Len(min_length=4, max_length=4)] | None = Field(
        default=None,
        description=(
            "The RGBA color stored as an array of four integers between 0 and 255"
        ),
    )


# ------------------------------------------------------------------------------
# Properties model
# ------------------------------------------------------------------------------


class LabelProperty(_BaseModel):
    label_value: int = Field(
        description="The pixel value for this label",
        alias="label-value",
    )
    # Additionally, an arbitrary number of key-value pairs MAY be present for each label
    # value, denoting arbitrary metadata associated with that label.
    # Label-value objects within the properties array do not need to have the same keys.


# ------------------------------------------------------------------------------
# Source model
# ------------------------------------------------------------------------------


class LabelSource(_BaseModel):
    image: str | None = None


# ------------------------------------------------------------------------------
# ImageLabel model
# ------------------------------------------------------------------------------


class ImageLabel(_BaseModel):
    colors: Annotated[UniqueList[LabelColor], MinLen(1)] | None = Field(
        default=None,
        description="The colors for this label image",
    )
    properties: Annotated[UniqueList[LabelProperty], MinLen(1)] | None = Field(
        default=None,
        description="The properties for this label image",
    )
    source: LabelSource | None = Field(
        default=None,
        description="The source of this label image",
    )

    # NOTE:
    # the WORDING of the spec is
    # image-label object SHOULD contain ... a version key, whose value MUST be a
    # string specifying the version of the OME-Zarr image-label schema.
    # but the EXAMPLE omits it, and the schema doesn't mention it at all.
    version: Literal["0.5"] | None = None


# ------------------------------------------------------------------------------
# Labels group model (contains paths to individual label images)
# ------------------------------------------------------------------------------


# NOTE: this is described in the spec, but doesn't appear in the schema.
class LabelsGroup(_BaseModel):
    """Model for the labels group that contains paths to individual label images."""

    version: Literal["0.5"] = "0.5"
    labels: Annotated[list[str], MinLen(1)] = Field(
        description="Array of paths to labeled multiscale images"
    )


# ------------------------------------------------------------------------------
# Label model (top-level for individual label images)
# ------------------------------------------------------------------------------


class LabelImage(Image):
    """Model for individual label images with multiscales + image-label metadata."""

    image_label: ImageLabel = Field(alias="image-label")
