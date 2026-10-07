import re
from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import AfterValidator, Field, PositiveInt, model_validator
from typing_extensions import Self

from yaozarrs._base import _BaseModel
from yaozarrs._plate_common import Acquisition, Column, PlateWell, Row
from yaozarrs._types import UniqueList

from ._version import CURRENT_VERSION, OMEV06

# NOTE (v0.6): the plate schema is structurally identical to v0.5 (only the
# `version` string changed), but the well schema changed: field-of-view paths
# now explicitly allow `._-` (with zarr node-name restrictions), see
# `FOVPathName` below. Acquisition/Column/Row/PlateWell are identical between
# v0.5 and v0.6 and are defined once in `yaozarrs._plate_common`.

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

_FOV_PATH_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_fov_path(path: str) -> str:
    # well.schema (v0.6): pattern ^[A-Za-z0-9_.-]+$, and per zarr node-name
    # rules the path must not consist only of periods or start with "__".
    if not _FOV_PATH_PATTERN.fullmatch(path):
        raise ValueError(
            f"Field-of-view path {path!r} may only contain characters in "
            "[A-Za-z0-9_.-]."
        )
    if all(c == "." for c in path):
        raise ValueError(
            f"Field-of-view path {path!r} must not consist only of periods."
        )
    if path.startswith("__"):
        raise ValueError(
            f"Field-of-view path {path!r} must not start with the reserved prefix '__'."
        )
    return path


FOVPathName = Annotated[str, AfterValidator(_validate_fov_path)]


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

    @model_validator(mode="after")
    def _validate_well_indices(self) -> Self:
        # NOTE: index.md is explicit that `path` MUST be "a name in the `rows`
        # array, a file separator, and a name from the `columns` array, in that
        # order" -- i.e. path == f"{row}/{column}". A handful of upstream
        # `ngff-spec` "valid" example fixtures (e.g. plate/minimal_acquisitions)
        # violate this MUST themselves (they use path == f"{column}/{row}");
        # that's an upstream fixture bug, not something to relax here, since the
        # JSON schema's `path` pattern doesn't encode side order and can't catch
        # it -- only the prose does.
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
            # spec: rowIndex, columnIndex, and path MUST all refer to the same
            # row/column pair.
            expected = (
                f"{self.rows[well.rowIndex].name}/{self.columns[well.columnIndex].name}"
            )
            if well.path != expected:
                raise ValueError(
                    f"Well path {well.path!r} does not match the row/column "
                    f"names at rowIndex {well.rowIndex} and columnIndex "
                    f"{well.columnIndex} (expected {expected!r})"
                )
        return self

    @model_validator(mode="after")
    def _validate_unique_acquisition_ids(self) -> Self:
        # spec: each acquisition id MUST be unique within the plate.
        if self.acquisitions:
            ids = [acq.id for acq in self.acquisitions]
            if len(ids) != len(set(ids)):
                dupes = sorted({i for i in ids if ids.count(i) > 1})
                raise ValueError(f"Acquisition ids must be unique. Duplicates: {dupes}")
        return self


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
        ├── A                       # Row A
        │   ├── 1                   # Column 1
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

        - [`Well`][yaozarrs.v06.Well]
        - [`FieldOfView`][yaozarrs.v06.FieldOfView]
    """

    version: OMEV06 = Field(
        default=CURRENT_VERSION,
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

    This class appears within the `images` list of a [`WellDef`][yaozarrs.v06.WellDef].
    """

    path: FOVPathName = Field(
        description=(
            "Relative path to this field's image group "
            "(typically a number like '0', '1', etc.)"
        ),
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

    @model_validator(mode="after")
    def _validate_unique_paths(self) -> Self:
        # spec: a path "MUST NOT be a duplicate of any other path in the images
        # list" (stronger than whole-object uniqueness: same path with different
        # acquisition is still forbidden).
        paths = [img.path for img in self.images]
        if len(paths) != len(set(paths)):
            dupes = sorted({p for p in paths if paths.count(p) > 1})
            raise ValueError(f"Field-of-view paths must be unique. Duplicates: {dupes}")
        return self


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

    version: OMEV06 = Field(
        default=CURRENT_VERSION,
        description="OME-NGFF specification version",
    )
    well: WellDef = Field(
        description="Field-of-view organization for this well",
    )
