"""v04 ome-zarr model.

<https://ngff.openmicroscopy.org/0.4>

<https://github.com/ome/ngff/tree/7ac3430c74a66e5bcf53e41c429143172d68c0a4>
"""

from __future__ import annotations

from yaozarrs._omero import Omero, OmeroChannel, OmeroRenderingDefs, OmeroWindow

from ._bf2raw import Bf2Raw
from ._image import (
    ChannelAxis,
    CustomAxis,
    Dataset,
    Image,
    Multiscale,
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
from ._series import Series
from ._well import FieldOfView, Well, WellDef
from ._zarr_json import OMEZarrGroupJSON

__all__ = [
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
    "OmeroRenderingDefs",
    "OmeroWindow",
    "Plate",
    "PlateDef",
    "PlateWell",
    "Row",
    "ScaleTransformation",
    "Series",
    "SpaceAxis",
    "TimeAxis",
    "TranslationTransformation",
    "Well",
    "WellDef",
]
