"""Plate building blocks (identical in v0.5 and v0.6)."""

import warnings
from collections.abc import Sequence
from typing import Annotated, Any

from annotated_types import MinLen
from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import Self

from yaozarrs._base import _BaseModel
from yaozarrs._errors import ValidationWarning
from yaozarrs._types import UNIQUE_ITEMS, duplicates


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


# (named PlateWell to distinguish it from each version's top-level `Well`)
class PlateWell(_BaseModel):
    """A well location reference within a plate.

    Maps a well's row/column position to its data location. This is a
    lightweight reference used in plate metadata, not the full well group
    (see the top-level `Well` model for the complete well metadata).
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


def check_plate_layout(
    rows: Sequence[Any],
    columns: Sequence[Any],
    wells: Sequence[Any],
    acquisitions: Sequence[Any] | None,
) -> None:
    """The MUSTs relating rows, columns, wells and acquisitions (also used by v04)."""
    for well in wells:
        if well.rowIndex >= len(rows):
            raise ValueError(
                f"Well {well.path} has rowIndex {well.rowIndex} "
                f"but only {len(rows)} rows exist"
            )
        if well.columnIndex >= len(columns):
            raise ValueError(
                f"Well {well.path} has columnIndex {well.columnIndex} "
                f"but only {len(columns)} columns exist"
            )
        # The `rowIndex`, `columnIndex`, and `path` MUST all refer to the same
        # row/column pair.
        row = rows[well.rowIndex].name
        col = columns[well.columnIndex].name
        if well.path != f"{row}/{col}":
            raise ValueError(
                f"Well path {well.path!r} does not match its rowIndex "
                f"({well.rowIndex}) and columnIndex ({well.columnIndex}), "
                f"which refer to '{row}/{col}'"
            )

    # [Each column/row name] MUST NOT be a duplicate of any other `name` in the
    # `columns`/`rows` array.
    for kind, items in (("column", columns), ("row", rows)):
        if dupes := duplicates([x.name for x in items]):
            raise ValueError(f"Plate {kind} names must be unique. Duplicates: {dupes}")
    # (no two wells may be equal (uniqueItems).  Ignoring unknown keys, which the
    # models drop, a well's path and indices must refer to the same row/column pair,
    # so equal paths mean equal wells)
    if dupes := duplicates([well.path for well in wells]):
        raise ValueError(f"Well paths must be unique. Duplicates: {dupes}")

    # Each acquisition object MUST contain an `id` key whose value MUST be an
    # unique integer identifier greater than or equal to 0 within the context
    # of the plate
    if dupes := duplicates([acq.id for acq in acquisitions or ()]):
        raise ValueError(
            f"Acquisition ids must be unique within a plate. Duplicates: {dupes}"
        )


class PlateDef(_BaseModel):
    """Plate layout and well organization.

    Defines the grid structure of a microplate and maps wells to their data.
    This is the core content of the `plate` metadata field.
    """

    # (uniqueItems is implied by the unique names and paths, see `check_plate_layout`)
    columns: Annotated[list[Column], UNIQUE_ITEMS, MinLen(1)] = Field(
        description="Column definitions for the plate grid"
    )
    rows: Annotated[list[Row], UNIQUE_ITEMS, MinLen(1)] = Field(
        description="Row definitions for the plate grid"
    )
    wells: Annotated[list[PlateWell], UNIQUE_ITEMS, MinLen(1)] = Field(
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

    @model_validator(mode="after")
    def _validate_layout(self) -> Self:
        check_plate_layout(self.rows, self.columns, self.wells, self.acquisitions)
        return self

    @model_validator(mode="after")
    def _warn_case_collisions(self) -> Self:
        # Care SHOULD be taken to avoid collisions on case-insensitive filesystems
        # (e.g. avoid using both `Aa` and `aA`).
        for kind, items in (("row", self.rows), ("column", self.columns)):
            lowered = [x.name.lower() for x in items]
            if clashes := [x.name for x in items if lowered.count(x.name.lower()) > 1]:
                warnings.warn(
                    f"Plate {kind} names {clashes} differ only by case, which will "
                    "collide on case-insensitive filesystems.",
                    ValidationWarning,
                    stacklevel=2,
                )
        return self
