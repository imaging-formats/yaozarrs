"""Label building blocks (identical in v0.5 and v0.6)."""

from typing import Annotated

from annotated_types import Interval, Len
from pydantic import Field

from yaozarrs._base import _BaseModel

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
