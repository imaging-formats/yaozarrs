from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import _BaseModel
from yaozarrs._common._labels import LabelColor, LabelProperty, LabelSource
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
    labels: list[str] = Field(
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
