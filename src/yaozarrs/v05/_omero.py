from typing import ClassVar, Literal

from pydantic import ConfigDict

from yaozarrs._base import _BaseModel


class OmeroWindow(_BaseModel):
    start: float
    min: float
    end: float
    max: float


class OmeroChannel(_BaseModel):
    window: OmeroWindow | None = None
    label: str | None = None
    family: str | None = None
    color: str | None = None
    active: bool | None = None
    inverted: bool | None = None
    coefficient: float | None = None


class OmeroRenderingDefs(_BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    model: Literal["color", "greyscale"] | str | None = None
    defaultT: int | None = None
    defaultZ: int | None = None
    projection: str | None = None  # "normal", "intmax", "intmean"


class Omero(_BaseModel):
    """A very rough/incomplete model of ImgData.

    https://omero.readthedocs.io/en/stable/developers/Web/WebGateway.html#imgdata

    Extra fields are allowed to accommodate missing fields.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    channels: list[OmeroChannel]
    id: int | None = None
    name: str | None = None
    version: str | None = None
    rdefs: OmeroRenderingDefs | None = None
