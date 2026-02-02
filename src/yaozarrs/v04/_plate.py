from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import Self

from yaozarrs._base import ZarrGroupModel, _BaseModel
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
# Acquisition model
# ------------------------------------------------------------------------------


class Acquisition(_BaseModel):
    """An imaging acquisition run within a plate.

    In high-content screening, multiple acquisition runs may be performed on the
    same plate (e.g., at different timepoints or with different settings).
    This class groups related images from a single acquisition session.
    """

    id: NonNegativeInt = Field(
        description="Unique identifier within the plate for this acquisition",
    )
    maximumfieldcount: PositiveInt | None = Field(
        default=None,
        description=(
            "Maximum number of fields-of-view across all wells in this acquisition"
        ),
    )
    name: str | None = Field(
        default=None,
        description="Human-readable name for this acquisition",
    )
    description: str | None = Field(
        default=None,
        description="Detailed description of the acquisition parameters or purpose",
    )
    starttime: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Acquisition start time as Unix epoch timestamp (seconds since 1970-01-01)"
        ),
    )
    endtime: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Acquisition end time as Unix epoch timestamp (seconds since 1970-01-01)"
        ),
    )


# ------------------------------------------------------------------------------
# Column model
# ------------------------------------------------------------------------------


class Column(_BaseModel):
    """A column in the plate grid.

    Columns are typically numbered (1, 2, 3, ...) but can use any
    alphanumeric identifier.
    """

    name: str = Field(
        description="Column identifier (typically numeric, e.g., '1', '2', '3')",
        pattern=r"^[A-Za-z0-9]+$",
    )


# ------------------------------------------------------------------------------
# Row model
# ------------------------------------------------------------------------------


class Row(_BaseModel):
    """A row in the plate grid.

    Rows are typically lettered (A, B, C, ...) but can use any alphanumeric identifier.
    """

    name: str = Field(
        description="Row identifier (typically alphabetic, e.g., 'A', 'B', 'C')",
        pattern=r"^[A-Za-z0-9]+$",
    )


# ------------------------------------------------------------------------------
# Well model
# ------------------------------------------------------------------------------


class PlateWell(_BaseModel):
    """A well location reference within a plate.

    Maps a well's row/column position to its data location. This is a
    lightweight reference used in plate metadata, not the full well group
    (see [`Well`][yaozarrs.v04.Well] for the complete well metadata).
    """

    path: str = Field(
        description=(
            "Relative path to the well's group (format: 'row/column', e.g., 'A/1')"
        ),
        pattern=r"^[A-Za-z0-9]+/[A-Za-z0-9]+$",
    )
    rowIndex: NonNegativeInt = Field(
        description="Zero-based index into the plate's rows list",
    )
    columnIndex: NonNegativeInt = Field(
        description="Zero-based index into the plate's columns list",
    )


# ------------------------------------------------------------------------------
# Plate model
# ------------------------------------------------------------------------------


class PlateDef(_BaseModel):
    """Plate layout and well organization.

    Defines the grid structure of a microplate and maps wells to their data.
    This is the core content of the `plate` metadata field.
    """

    columns: Annotated[UniqueList[Column], MinLen(1)] = Field(
        description="Column definitions for the plate grid"
    )
    rows: Annotated[UniqueList[Row], MinLen(1)] = Field(
        description="Row definitions for the plate grid"
    )
    wells: Annotated[UniqueList[PlateWell], MinLen(1)] = Field(
        description="List of all wells present in this plate with their grid positions"
    )

    acquisitions: list[Acquisition] | None = Field(
        default=None,
        description="Imaging acquisition runs performed on this plate",
    )
    field_count: PositiveInt | None = Field(
        default=None,
        description="Maximum number of fields-of-view per well across the entire plate",
    )
    name: str | None = Field(
        default=None,
        description="Human-readable name for this plate",
    )
    version: Literal["0.4"] = "0.4"

    @model_validator(mode="after")
    def _validate_well_indices(self) -> Self:
        for well in self.wells:
            if well.rowIndex >= len(self.rows):
                raise ValueError(
                    f"Well {well.path} has rowIndex {well.rowIndex} "
                    f"but only {len(self.rows)} rows exist"
                )
            if well.columnIndex >= len(self.columns):
                raise ValueError(
                    f"Well {well.path} has columnIndex {well.columnIndex} "
                    f"but only {len(self.columns)} columns exist"
                )
        return self


class FieldOfView(_BaseModel):
    """A single field-of-view (imaging position) within a well.

    Wells typically contain multiple fields-of-view when the well area is larger
    than a single camera frame. Each field-of-view is a complete multiscale image.

    This class appears within the `images` list of a [`WellDef`][yaozarrs.v04.WellDef].
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
    version: Literal["0.4"] = "0.4"


# ------------------------------------------------------------------------------
# Well model (top-level)
# ------------------------------------------------------------------------------


class Well(ZarrGroupModel):
    """Top-level well metadata within a plate.

    This model corresponds to the `.zattrs` file (or zarr.json attributes)
    in a well group. It lists all fields-of-view (imaging positions)
    captured within this well.

    !!! example "Typical Structure"
        ```
        A/1/                   # Well at row A, column 1
        ├── .zattrs            # Contains this metadata
        ├── 0/                 # First field-of-view
        │   ├── .zattrs        # Image metadata
        │   ├── 0/             # Highest resolution
        │   └── 1/             # Next resolution
        └── 1/                 # Second field-of-view
        ```
    """

    well: WellDef = Field(
        description="Field-of-view organization for this well",
    )


# ------------------------------------------------------------------------------
# Plate model (top-level)
# ------------------------------------------------------------------------------


class Plate(ZarrGroupModel):
    """Top-level plate metadata for high-content screening.

    This model corresponds to the `.zattrs` file (or zarr.json attributes)
    in a plate group, organizing a microplate's wells in a grid layout.
    Each well contains one or more fields-of-view, which in turn contain
    multiscale images.

    !!! example "Typical Structure"
        ```
        my_plate.ome.zarr/
        ├── A/                      # Row A
        │   ├── 1/                  # Well A1
        │   │   ├── 0/              # FOV 0 in A1
        │   │   │   ├── 0/          # Multiscale level 0
        │   │   │   └── .zattrs     # contains {"multiscales": [...]}
        │   │   ├── 1/              # FOV 1 in A1
        │   │   └── .zattrs         # contains {"well": {...}}
        │   └── 2/                  # Well A2
        ├── B/                      # Row B
        └── .zattrs                 # contains {"plate": {...}}
        ```

    !!! note
        See also:

        - [`Well`][yaozarrs.v04.Well]
        - [`FieldOfView`][yaozarrs.v04.FieldOfView]
    """

    plate: PlateDef = Field(
        description="Plate layout and well organization",
    )
