from typing import Annotated, ClassVar, Literal

from annotated_types import Interval, Len, MinLen
from pydantic import ConfigDict, Field

from yaozarrs._base import _BaseModel
from yaozarrs._types import UniqueList

from ._image import Image

# ------------------------------------------------------------------------------
# Color model
# ------------------------------------------------------------------------------

Int8bit = Annotated[int, Interval(ge=0, le=255)]


class LabelColor(_BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(validate_by_name=True)
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
    version: Literal["0.4"] = "0.4"


# ------------------------------------------------------------------------------
# Labels group model (contains paths to individual label images)
# ------------------------------------------------------------------------------


# NOTE: this is described in the spec, but doesn't appear in the schema.
class LabelsGroup(_BaseModel):
    """Model for the labels group that contains paths to individual label images."""

    labels: Annotated[list[str], MinLen(1)] = Field(
        description="Array of paths to labeled multiscale images"
    )


# ------------------------------------------------------------------------------
# Label model (top-level for individual label images)
# ------------------------------------------------------------------------------


class LabelImage(Image):
    """Model for individual label images with multiscales + image-label metadata."""

    image_label: ImageLabel = Field(alias="image-label")
