"""v04 ome-zarr model.

https://ngff.openmicroscopy.org/0.4/
https://github.com/ome/ngff/tree/0.4
"""

from __future__ import annotations

from yaozarrs.v04._bf2raw import Bf2Raw
from yaozarrs.v04._ome import OME

from ._image import (
    ChannelAxis,
    CustomAxis,
    Dataset,
    Image,
    Multiscale,
    Omero,
    OmeroChannel,
    OmeroWindow,
    ScaleTransformation,
    SpaceAxis,
    TimeAxis,
    TranslationTransformation,
)
from ._label import (
    ImageLabel,
    LabelColor,
    LabelImage,
    LabelProperty,
    LabelSource,
)
from ._plate import Acquisition, Column, Plate, PlateDef, PlateWell, Row
from ._well import FieldOfView, Well, WellDef
from ._zarr_json import OMEZarrGroupJSON

__all__ = [
    "OME",
    "Acquisition",
    "Bf2Raw",
    "ChannelAxis",
    "Column",
    "CustomAxis",
    "Dataset",
    "FieldOfView",
    "Image",
    "ImageLabel",
    "LabelColor",
    "LabelImage",
    "LabelProperty",
    "LabelSource",
    "Multiscale",
    "OMEZarrGroupJSON",
    "Omero",
    "OmeroChannel",
    "OmeroWindow",
    "Plate",
    "PlateDef",
    "PlateWell",
    "Row",
    "ScaleTransformation",
    "SpaceAxis",
    "TimeAxis",
    "TranslationTransformation",
    "Well",
    "WellDef",
]


# OMENode: TypeAlias = Image | Plate | LabelImage | Well | OME | Bf2Raw
# """Anything that can live in the "ome" key of a v0.4 ome-zarr file."""


# class OMEZarr(BaseModel):
#     ome: OMENode
