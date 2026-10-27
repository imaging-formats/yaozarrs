from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field

from yaozarrs._base import _BaseModel
from yaozarrs._common._plate import Acquisition, Column, PlateDef, PlateWell, Row
from yaozarrs._types import UniqueList
from yaozarrs._util import RelaxedFOVPathName

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "Plate",
    "PlateDef",
    # PlateDef
    "Column",
    "Row",
    "PlateWell",
    "Acquisition",
    # Well and its dependencies
    "Well",
    "WellDef",
    "FieldOfView",
]

# ------------------------------------------------------------------------------
# Plate model (top-level)
# ------------------------------------------------------------------------------


class Plate(_BaseModel):
    """Top-level plate metadata for high-content screening.

    This model corresponds to the `zarr.json` file in a plate group, organizing
    a microplate's wells in a grid layout. Each well contains one or more
    fields-of-view, which in turn contain multiscale images.

    !!! example "Typical Structure"
        ```
        my_plate.ome.zarr
        ├── A                       # Col A
        │   ├── 1                   # Row 1
        │   │   ├── 0               # FOV 0 (in A1)
        │   │   │   ├── 0           # FOV 0 - Multiscale level 0
        │   │   │   └── zarr.json   # contains ["ome"]["multiscales"]
        │   │   ├── 1               # FOV 1 (in A1)
        │   │   │   ├── 0           # FOV 1 - Multiscale level 0
        │   │   │   └── zarr.json   # contains ["ome"]["multiscales"]
        │   │   └── zarr.json       # well metadata (contains ['ome']['well'])
        │   ├── 2
        │   │   └── ...
        │   └── 3
        │       └── ...
        ├── B
        │   └── ...
        ├── C
        │   └── ...
        └── zarr.json                # plate metadata (contains ['ome']['plate'])
        ```

    !!! note
        See also:

        - [`Well`][yaozarrs.v05.Well]
        - [`FieldOfView`][yaozarrs.v05.FieldOfView]
    """

    version: Literal["0.5"] = Field(
        default="0.5",
        description="OME-NGFF specification version",
    )
    plate: PlateDef = Field(
        description="Plate layout and well organization",
    )

    bioformats2raw_layout: Literal[3] | None = Field(
        default=None,
        alias="bioformats2raw.layout",
        description=(
            "Marker indicating this plate was created by bioformats2raw version 3"
        ),
    )


class FieldOfView(_BaseModel):
    """A single field-of-view (imaging position) within a well.

    Wells typically contain multiple fields-of-view when the well area is larger
    than a single camera frame. Each field-of-view is a complete multiscale image.

    This class appears within the `images` list of a [`WellDef`][yaozarrs.v05.WellDef].
    """

    path: RelaxedFOVPathName = Field(
        description=(
            "Relative path to this field's image group "
            "(typically a number like '0', '1', etc.)"
        ),
        # pattern=r"^[A-Za-z0-9]+$",
    )
    acquisition: int | None = Field(
        default=None,
        description=(
            "Acquisition ID linking this field to a specific acquisition run. "
            "Required when the parent plate has multiple acquisitions."
        ),
    )


class WellDef(_BaseModel):
    """Organization of fields-of-view within a well.

    This is the core content of the `well` metadata field, listing all
    imaging positions captured for this well.
    """

    images: Annotated[UniqueList[FieldOfView], MinLen(1)] = Field(
        description="List of all fields-of-view imaged in this well",
    )


# ------------------------------------------------------------------------------
# Well model (top-level)
# ------------------------------------------------------------------------------


class Well(_BaseModel):
    """Top-level well metadata within a plate.

    This model corresponds to the `zarr.json` file in a well group. It lists
    all fields-of-view (imaging positions) captured within this well.

    !!! example "Typical Structure"
        ```
        A/1/                   # Well at row A, column 1
        ├── zarr.json          # Contains this metadata
        ├── 0/                 # First field-of-view
        │   ├── zarr.json      # Image metadata
        │   ├── 0/             # Highest resolution
        │   └── 1/             # Next resolution
        └── 1/                 # Second field-of-view
        ```
    """

    version: Literal["0.5"] = Field(
        default="0.5",
        description="OME-NGFF specification version",
    )
    well: WellDef = Field(
        description="Field-of-view organization for this well",
    )
