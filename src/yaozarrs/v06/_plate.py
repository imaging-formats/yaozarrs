from typing import Annotated, Literal, TypeAlias

from annotated_types import MinLen
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Self

from yaozarrs._base import _BaseModel
from yaozarrs._common._plate import Acquisition, Column, PlateDef, PlateWell, Row
from yaozarrs._types import UniqueList
from yaozarrs._util import BAD_NODE_RE

from ._version import VersionStr

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

    This model corresponds to the `ome` metadata in a plate group's `zarr.json`,
    organizing a microplate's wells in a grid layout. Each well contains one or more
    fields-of-view, which in turn contain multiscale images.

    !!! example "Typical Structure"
        ```
        my_plate.ome.zarr
        ├── A                       # Row A
        │   ├── 1                   # Column 1
        │   │   ├── 0               # FOV 0 (in A1)
        │   │   │   ├── s0          # FOV 0 - Multiscale level 0
        │   │   │   └── zarr.json   # contains ["ome"]["multiscales"]
        │   │   ├── 1               # FOV 1 (in A1)
        │   │   │   └── ...
        │   │   └── zarr.json       # well metadata (contains ['ome']['well'])
        │   └── 2
        │       └── ...
        ├── B
        │   └── ...
        └── zarr.json                # plate metadata (contains ['ome']['plate'])
        ```

    !!! note
        See also:

        - [`Well`][yaozarrs.v06.Well]
        - [`FieldOfView`][yaozarrs.v06.FieldOfView]
    """

    version: VersionStr = Field(default="0.6", description="OME-NGFF version")
    plate: PlateDef = Field(description="Plate layout and well organization")

    # If the top-level group represents a plate, the `bioformats2raw.layout` metadata
    # will be present but the `plate` key MUST also be present
    bioformats2raw_layout: Literal[3] | None = Field(
        default=None,
        alias="bioformats2raw.layout",
        description=(
            "Marker indicating this plate was created by bioformats2raw version 3"
        ),
    )


# ------------------------------------------------------------------------------
# Well model
# ------------------------------------------------------------------------------


def _validate_fov_path(path: str) -> str:
    # The `path` MUST NOT consist only of periods (like `.` or `..`) or start with the
    # reserved prefix `__`; The `path` MUST NOT be an empty string and MUST NOT contain
    # `/` characters
    if BAD_NODE_RE.match(path):
        raise ValueError(
            f"Field of view path {path!r} is not a valid Zarr node name (must not be "
            "empty, only periods, start with '__', or contain '/')."
        )
    return path


# The `path` MUST only use characters in the sets `a-z`, `A-Z`, `0-9`, `-`, `_`, `.`.
# (v0.6 relaxed this from "only alphanumeric characters")
FieldOfViewPath: TypeAlias = Annotated[
    str, Field(pattern=r"^[A-Za-z0-9._-]+$"), AfterValidator(_validate_fov_path)
]


class FieldOfView(_BaseModel):
    """A single field-of-view (imaging position) within a well.

    Wells typically contain multiple fields-of-view when the well area is larger
    than a single camera frame. Each field-of-view is a complete multiscale image.

    This class appears within the `images` list of a [`WellDef`][yaozarrs.v06.WellDef].
    """

    path: FieldOfViewPath = Field(
        description=(
            "Relative path to this field's image group "
            "(typically a number like '0', '1', etc.)"
        )
    )
    # If multiple acquisitions were performed in the plate, it MUST contain an
    # `acquisition` key whose value MUST be an integer identifying the acquisition
    # (that it matches one of the plate's acquisitions needs the plate metadata)
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
        # The `path` MUST be case-sensitive, and MUST NOT be a duplicate of any other
        # `path` in the `images` list.  (UniqueList only catches identical entries.)
        paths = [img.path for img in self.images]
        if dupes := sorted({p for p in paths if paths.count(p) > 1}):
            raise ValueError(f"Field of view paths must be unique. Duplicates: {dupes}")
        return self


class Well(_BaseModel):
    """Top-level well metadata within a plate.

    This model corresponds to the `ome` metadata in a well group's `zarr.json`. It
    lists all fields-of-view (imaging positions) captured within this well.

    !!! example "Typical Structure"
        ```
        A/1/                   # Well at row A, column 1
        ├── zarr.json          # Contains this metadata
        ├── 0/                 # First field-of-view
        │   ├── zarr.json      # Image metadata
        │   ├── s0/            # Highest resolution
        │   └── s1/            # Next resolution
        └── 1/                 # Second field-of-view
        ```
    """

    version: VersionStr = Field(default="0.6", description="OME-NGFF version")
    well: WellDef = Field(description="Field-of-view organization for this well")
