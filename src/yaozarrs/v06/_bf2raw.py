from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import _BaseModel

from ._version import VersionStr

__all__ = ["Bf2Raw", "Series"]


class Bf2Raw(_BaseModel):
    """Model for `zarr.json` in a group representing multi-series image collections.

    The bioformats2raw layout is a transitional specification for filesets that
    already exist in the wild. An upcoming NGFF specification will replace this layout
    with explicit metadata. It is currently the only way to represent multi-position
    image collections, and it has a recommended place for OME-XML metadata.

    !!! example "Typical Structure"
        ```
        root.ome.zarr             # One converted fileset from bioformats2raw
        ├── zarr.json             # Contains "bioformats2raw.layout" metadata
        ├── OME                   # Special group for containing OME metadata
        │   ├── zarr.json         # Contains "series" metadata
        │   └── METADATA.ome.xml  # OME-XML file stored within the Zarr fileset
        ├── 0                     # First image in the collection
        ├── 1                     # Second image in the collection
        └── ...
        ```

    !!! note
        The `OME/zarr.json` file typically contains [`Series`][yaozarrs.v06.Series]
        metadata that explicitly lists image paths. If not present, images MUST be
        numbered as shown above, starting from `0`, e.g., `0/`, `1/`, `2/`, etc.
    """

    version: VersionStr = Field(default="0.6", description="OME-NGFF version")
    # MUST have the value `3` for the `bioformats2raw.layout` key
    bioformats2raw_layout: Literal[3] = Field(
        alias="bioformats2raw.layout",
        description="Layout version marker added by bioformats2raw (currently 3)",
    )


class Series(_BaseModel):
    """Model for `OME/zarr.json` in a [`Bf2Raw`][yaozarrs.v06.Bf2Raw] collection.

    Used when converting files containing multiple distinct images (e.g.,
    multi-series microscopy formats) to OME-NGFF. Each series becomes a
    separate image group, and this metadata lists them all.

    !!! note
        The spec is ambiguous about whether `series` is required.
        This library treats it as required for disambiguation, since otherwise this
        model would simply contain "version".
    """

    version: VersionStr = Field(default="0.6", description="OME-NGFF version")
    # `series` MUST be an array of string objects, each of which is a path to an image
    # group.
    series: Annotated[list[str], MinLen(1)] = Field(
        description=(
            "Ordered list of paths to image groups, matching the order in "
            "the companion OME-XML metadata"
        )
    )
