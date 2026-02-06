import re
from typing import Annotated, ClassVar, Literal

from pydantic import AfterValidator, ConfigDict, Field

from yaozarrs._base import _BaseModel


class OmeroWindow(_BaseModel):
    """Display intensity window for a channel.

    Defines how pixel intensities map to display brightness. The window
    sets both the current display range (start/end) and the allowed range (min/max).
    """

    start: float = Field(description="Lower bound of the current display window")
    min: float = Field(description="Minimum allowed intensity value for this channel")
    end: float = Field(description="Upper bound of the current display window")
    max: float = Field(description="Maximum allowed intensity value for this channel")


def _valid_hex(value: str) -> str:
    """Ensure a string is a valid 6-character hex color code."""
    value = value.lstrip("#").upper()
    if len(value) == 3:
        value = "".join(2 * c for c in value)
    if len(value) != 6 or not re.fullmatch(r"[0-9A-F]{6}", value):
        raise ValueError(f"Invalid hex color code: {value}")
    return value


class OmeroChannel(_BaseModel):
    """Rendering settings for a single channel.

    Specifies how to display one channel of a multi-channel image, including
    color mapping, intensity windowing, and visibility.
    """

    window: OmeroWindow = Field(
        description="Intensity window for this channel",
    )
    label: str | None = Field(
        default=None,
        description="Human-readable name for this channel (e.g., 'DAPI', 'GFP')",
    )
    family: str | None = Field(
        default=None,
        description="Colormap family (typically 'linear' for fluorescence)",
    )
    color: Annotated[str, AfterValidator(_valid_hex)] = Field(
        description="Display color as hex string without # (e.g., 'FF0000' for red)",
    )
    active: bool | None = Field(
        default=None,
        description="Whether this channel is visible by default",
    )
    inverted: bool | None = Field(
        default=None,
        description="Whether to invert the intensity mapping",
    )
    coefficient: float | None = Field(
        default=None,
        description="Multiplicative coefficient for intensity scaling",
    )


class OmeroRenderingDefs(_BaseModel):
    """Default rendering parameters for multi-dimensional images.

    Specifies which slices to display by default in time-lapse or z-stack images,
    and how to render them (color vs grayscale, projection method).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    model: Literal["color", "greyscale"] | str | None = Field(
        default=None,
        description=(
            "Rendering mode: 'color' for multi-channel composite, "
            "'greyscale' for single channel"
        ),
    )
    defaultT: int | None = Field(
        default=None,
        description="Default time point index to display",
    )
    defaultZ: int | None = Field(
        default=None,
        description="Default z-section index to display",
    )
    projection: str | None = Field(
        default=None,
        description=(
            "Projection method for z-stacks: 'normal', "
            "'intmax' (max intensity), or 'intmean'"
        ),
    )


class Omero(_BaseModel):
    """Optional OMERO rendering metadata for visualization.

    Provides display hints for viewers, including channel colors, intensity windows,
    and default viewing parameters. This metadata is transitional and may be
    replaced in future OME-NGFF versions.

    !!! warning "Transitional Metadata"
        The OMERO metadata is inherited from OME-NGFF 0.4 and maintained for
        backwards compatibility. The spec acknowledges this needs improvement.

    !!! note "Extra Fields Allowed"
        This model permits additional fields beyond those explicitly defined,
        as the full OMERO metadata structure is extensive and evolving.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    channels: list[OmeroChannel] = Field(
        description="Rendering settings for each channel in the image",
    )
    id: int | None = Field(
        default=None,
        description="OMERO image ID (if from an OMERO server)",
    )
    name: str | None = Field(
        default=None,
        description="Image name from OMERO",
    )
    version: str | None = Field(
        default=None,
        description="OMERO metadata version",
    )
    rdefs: OmeroRenderingDefs | None = Field(
        default=None,
        description="Default rendering parameters for multi-dimensional viewing",
    )
