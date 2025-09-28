from typing import Annotated, Literal

from annotated_types import MinLen
from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import Self

from yaozarrs._base import ZarrGroupModel, _BaseModel
from yaozarrs._utils import UniqueList

# ------------------------------------------------------------------------------
# Acquisition model
# ------------------------------------------------------------------------------


class Acquisition(_BaseModel):
    id: NonNegativeInt = Field(
        description="A unique identifier within the context of the plate",
    )
    maximumfieldcount: PositiveInt | None = Field(
        default=None,
        description="The maximum number of fields of view for the acquisition",
    )
    name: str | None = Field(
        default=None,
        description="The name of the acquisition",
    )
    description: str | None = Field(
        default=None,
        description="The description of the acquisition",
    )
    starttime: NonNegativeInt | None = Field(
        default=None,
        description=(
            "The start timestamp of the acquisition, expressed as epoch time "
            "i.e. the number seconds since the Epoch"
        ),
    )
    endtime: NonNegativeInt | None = Field(
        default=None,
        description=(
            "The end timestamp of the acquisition, expressed as epoch time "
            "i.e. the number seconds since the Epoch"
        ),
    )


# ------------------------------------------------------------------------------
# Column model
# ------------------------------------------------------------------------------


class Column(_BaseModel):
    name: str = Field(
        description="The column name",
        pattern=r"^[A-Za-z0-9]+$",
    )


# ------------------------------------------------------------------------------
# Row model
# ------------------------------------------------------------------------------


class Row(_BaseModel):
    name: str = Field(
        description="The row name",
        pattern=r"^[A-Za-z0-9]+$",
    )


# ------------------------------------------------------------------------------
# Well model
# ------------------------------------------------------------------------------


class PlateWell(_BaseModel):
    """Individual well in a plate."""

    path: str = Field(
        description="The path to the well subgroup",
        pattern=r"^[A-Za-z0-9]+/[A-Za-z0-9]+$",
    )
    rowIndex: NonNegativeInt = Field(
        description="The index of the well in the rows list",
    )
    columnIndex: NonNegativeInt = Field(
        description="The index of the well in the columns list",
    )


# ------------------------------------------------------------------------------
# Plate model
# ------------------------------------------------------------------------------


class PlateDef(_BaseModel):
    columns: Annotated[UniqueList[Column], MinLen(1)] = Field(
        description="The columns of the plate"
    )
    rows: Annotated[UniqueList[Row], MinLen(1)] = Field(
        description="The rows of the plate"
    )
    wells: Annotated[UniqueList[PlateWell], MinLen(1)] = Field(
        description="The wells of the plate"
    )

    acquisitions: list[Acquisition] | None = Field(
        default=None,
        description="The acquisitions for this plate",
    )
    field_count: PositiveInt | None = Field(
        default=None,
        description="The maximum number of fields per view across all wells",
    )
    name: str | None = Field(
        default=None,
        description="The name of the plate",
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


# ------------------------------------------------------------------------------
# Plate model (top-level)
# ------------------------------------------------------------------------------


class Plate(ZarrGroupModel):
    plate: PlateDef
