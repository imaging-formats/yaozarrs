"""Plate-grid building blocks shared across OME-NGFF spec versions.

`Acquisition`, `Column`, `Row`, and `PlateWell` are structurally and
semantically identical in the v0.5 and v0.6 plate schemas (only the
`version` string differs at the top-level `Plate`/`Well` models, which stay
version-specific). None of these four are used as `isinstance` discrimination
targets anywhere in the codebase (unlike `Plate`, `Well`, `Bf2Raw`, `Series`,
etc., which dispatch on the top-level OME metadata type), so sharing the
class objects directly -- rather than duplicating them per version -- is safe:
there's no risk of a v0.5 object being mistaken for a v0.6 one via `isinstance`.

Each version's `_plate` module re-exports these under its own namespace so
`yaozarrs.v05.Acquisition` and `yaozarrs.v06.Acquisition` keep working exactly
as before; only the *definition* lives here now.
"""

from __future__ import annotations

from pydantic import Field, NonNegativeInt, PositiveInt

from yaozarrs._base import _BaseModel

__all__ = ["Acquisition", "Column", "PlateWell", "Row"]


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


class Column(_BaseModel):
    """A column in the plate grid.

    Columns are typically numbered (1, 2, 3, ...) but can use any
    alphanumeric identifier.
    """

    name: str = Field(
        description="Column identifier (typically numeric, e.g., '1', '2', '3')",
        pattern=r"^[A-Za-z0-9]+$",
    )


class Row(_BaseModel):
    """A row in the plate grid.

    Rows are typically lettered (A, B, C, ...) but can use any alphanumeric identifier.
    """

    name: str = Field(
        description="Row identifier (typically alphabetic, e.g., 'A', 'B', 'C')",
        pattern=r"^[A-Za-z0-9]+$",
    )


# naming this PlateWell to disambiguate from a top level Well
class PlateWell(_BaseModel):
    """A well location reference within a plate.

    Maps a well's row/column position to its data location. This is a
    lightweight reference used in plate metadata, not the full well group.
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
