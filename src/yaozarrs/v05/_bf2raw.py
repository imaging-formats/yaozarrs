from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import _BaseModel


class Bf2Raw(_BaseModel):
    """Model for `zarr.json` in a group representing multi-series image collections.

    The bioformats2raw layout was added in v0.4 as a transitional specification to
    specify filesets that already exist in the wild. An upcoming NGFF specification will
    replace this layout with explicit metadata.

    This model is a transitional spec for representing multi-position image collections
    in OME-NGFF v0.5. It is currently the only way to represent multi-position image
    collections. It also has a recommended place for OME-XML metadata.

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
        The `OME/zarr.json` file typically contains [`Series`][yaozarrs.v05.Series]
        metadata that explicitly lists image paths. If not present, images MUST be
        numbered as shown above, starting from `0`, e.g., `0/`, `1/`, `2/`, etc.
    """

    version: Literal["0.5"] = Field(
        default="0.5",
        description="OME-NGFF specification version",
    )
    bioformats2raw_layout: Literal[3] = Field(
        alias="bioformats2raw.layout",
        description="Layout version marker added by bioformats2raw (currently 3)",
    )


class Series(_BaseModel):
    """Model for `OME/zarr.json` in an [`Bf2Raw`][yaozarrs.v05.Bf2Raw] collection.

    Used when converting files containing multiple distinct images (e.g.,
    multi-series microscopy formats) to OME-NGFF. Each series becomes a
    separate image group, and this metadata lists them all.

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
        The spec is ambiguous about whether `series` is required.
        This library treats it as required for disambiguation. Since otherwise this
        model would simply contain "version".
    """

    version: Literal["0.5"] = Field(
        default="0.5",
        description="OME-NGFF specification version",
    )
    series: Annotated[list[str], MinLen(1)] = Field(
        description=(
            "Ordered list of paths to image groups, matching the order in "
            "the companion OME-XML metadata"
        )
    )
