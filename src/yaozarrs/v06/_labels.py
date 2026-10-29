from typing import Annotated

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import _BaseModel, _OMEDocument
from yaozarrs._common._labels import LabelColor, LabelProperty, LabelSource
from yaozarrs._types import UniqueList

from ._image import Image
from ._version import VersionStr

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

    !!! note
        Unlike v0.5, `image-label` no longer has its own `version` key in v0.6.
    """

    # That `image-label` object SHOULD contain a `colors` key, whose value MUST be a
    # JSON array describing color information for the unique label values.
    colors: Annotated[UniqueList[LabelColor], MinLen(1)] | None = Field(
        default=None,
        description="Color mappings for label values, used for visualization",
    )
    # the value of the `properties` key MUST be an array of JSON objects
    properties: Annotated[UniqueList[LabelProperty], MinLen(1)] | None = Field(
        default=None,
        description="Arbitrary metadata properties for individual label values",
    )
    # The value of the `source` key MUST be a JSON object containing information about
    # the original image from which the label image derives.
    source: LabelSource | None = Field(
        default=None,
        description="Reference to the source intensity image that was segmented",
    )


# ------------------------------------------------------------------------------
# Labels group model (contains paths to individual label images)
# ------------------------------------------------------------------------------


class LabelsGroup(_OMEDocument):
    """Top-level labels collection metadata.

    This model corresponds to the `ome` metadata in the `zarr.json` of a `labels/`
    group, which acts as a container for multiple segmentation/annotation images.

    !!! example "Typical Structure"
        ```
        my_image/
        ├── zarr.json          # Image metadata
        ├── s0/                # Image arrays
        └── labels/
            ├── zarr.json      # Contains this metadata
            ├── cells/         # One label image
            └── nuclei/        # Another label image
        ```
    """

    version: VersionStr = Field(default="0.6", description="OME-NGFF version")
    # [The labels group metadata] MUST contain a JSON object with the key `labels`,
    # whose value is a JSON array of paths to the labeled multiscale image(s).
    labels: list[str] = Field(
        description="Paths to individual label image groups within this collection"
    )


# ------------------------------------------------------------------------------
# Label model (top-level for individual label images)
# ------------------------------------------------------------------------------


class LabelImage(Image):
    """A complete label image with multiscale pyramids and label metadata.

    Combines the standard image structure (multiscale pyramids, coordinate systems,
    etc.) with label-specific metadata (colors, properties, source reference).
    Label images must use integer data types.

    !!! note "Relationship to Image"
        This is an [`Image`][yaozarrs.v06.Image] with additional `image-label`
        metadata. The multiscale pyramids follow the same structure as regular images
        but contain integer segmentation masks instead of intensity data.
    """

    image_label: ImageLabel = Field(
        alias="image-label",
        description="Label-specific metadata (colors, properties, source link)",
    )
