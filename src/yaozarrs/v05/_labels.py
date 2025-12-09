from typing import Annotated, Literal

from annotated_types import Interval, Len, MinLen
from pydantic import Field

from yaozarrs._base import _BaseModel
from yaozarrs._types import UniqueList

from ._image import Image

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    # LabelImage and its dependencies
    "LabelImage",
    "ImageLabel",
    "LabelColor",
    "LabelProperty",
    "LabelSource",
    # LabelsGroup (container for label images)
    "LabelsGroup",
]

# ------------------------------------------------------------------------------
# Color model
# ------------------------------------------------------------------------------

Int8bit = Annotated[int, Interval(ge=0, le=255)]


class LabelColor(_BaseModel):
    """Display color mapping for a label value.

    Associates a specific label integer value with an RGBA color for
    visualization purposes.
    """

    label_value: float = Field(
        description="Integer label value from the segmentation image",
        alias="label-value",
    )
    rgba: Annotated[list[Int8bit], Len(min_length=4, max_length=4)] | None = Field(
        default=None,
        description="RGBA color as [red, green, blue, alpha], each 0-255",
    )


# ------------------------------------------------------------------------------
# Properties model
# ------------------------------------------------------------------------------


class LabelProperty(_BaseModel):
    """Custom metadata for a label value.

    Associates arbitrary key-value properties with a specific label integer.
    Different labels can have different sets of properties - there's no requirement
    for consistency across labels.

    !!! example
        ```python
        LabelProperty(label_value=1, cell_type="neuron", area=1250.5)
        LabelProperty(label_value=2, cell_type="glia", perimeter=180.3)
        ```
    """

    label_value: int = Field(
        description="Integer label value from the segmentation image",
        alias="label-value",
    )


# ------------------------------------------------------------------------------
# Source model
# ------------------------------------------------------------------------------


class LabelSource(_BaseModel):
    """Reference to the source image that was segmented.

    Points back to the original intensity image from which this label
    image was derived.
    """

    image: str | None = Field(
        default=None,
        description=(
            "Relative path to the source image group (default: '../../', "
            "pointing to the parent of the labels/ directory)"
        ),
    )


# ------------------------------------------------------------------------------
# ImageLabel model
# ------------------------------------------------------------------------------


class ImageLabel(_BaseModel):
    """Metadata for a segmentation/annotation label image.

    Enhances a multiscale label image with display colors, semantic properties,
    and links back to the source intensity image. Label images are integer-valued
    arrays where each unique value represents a distinct object or region.
    """

    colors: Annotated[UniqueList[LabelColor], MinLen(1)] | None = Field(
        default=None,
        description="Color mappings for label values, used for visualization",
    )
    properties: Annotated[UniqueList[LabelProperty], MinLen(1)] | None = Field(
        default=None,
        description="Arbitrary metadata properties for individual label values",
    )
    source: LabelSource | None = Field(
        default=None,
        description="Reference to the source intensity image that was segmented",
    )

    version: Literal["0.5"] | None = Field(
        default=None,
        description="OME-NGFF image-label specification version (often omitted)",
    )


# ------------------------------------------------------------------------------
# Labels group model (contains paths to individual label images)
# ------------------------------------------------------------------------------


class LabelsGroup(_BaseModel):
    """Top-level labels collection metadata.

    This model corresponds to the `zarr.json` file in a `labels/` directory,
    which acts as a container for multiple segmentation/annotation images.

    !!! example "Typical Structure"
        ```
        my_image/
        ├── zarr.json          # Image metadata
        ├── 0/                 # Image arrays
        └── labels/
            ├── zarr.json      # Contains this metadata
            ├── cells/         # One label image
            └── nuclei/        # Another label image
        ```
    """

    version: Literal["0.5"] = Field(
        default="0.5",
        description="OME-NGFF specification version",
    )
    labels: Annotated[list[str], MinLen(1)] = Field(
        description="Paths to individual label image groups within this collection"
    )


# ------------------------------------------------------------------------------
# Label model (top-level for individual label images)
# ------------------------------------------------------------------------------


class LabelImage(Image):
    """A complete label image with multiscale pyramids and label metadata.

    Combines the standard image structure (multiscale pyramids, axes, etc.)
    with label-specific metadata (colors, properties, source reference).
    Label images must use integer data types.

    !!! note "Relationship to Image"
        This is an [`Image`][yaozarrs.v05.Image] with additional `image-label`
        metadata. The multiscale pyramids follow the same structure as regular images
        but contain integer segmentation masks instead of intensity data.
    """

    image_label: ImageLabel = Field(
        alias="image-label",
        description="Label-specific metadata (colors, properties, source link)",
    )
