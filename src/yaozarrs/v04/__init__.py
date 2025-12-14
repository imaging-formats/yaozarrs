"""OME-NGFF v0.4 metadata models.

Specification: <https://ngff.openmicroscopy.org/0.4>

Schema: <https://github.com/ome/ngff/tree/7ac3430c74a66e5bcf53e41c429143172d68c0a4>
"""

from __future__ import annotations

from yaozarrs._axis import Axis, ChannelAxis, CustomAxis, SpaceAxis, TimeAxis
from yaozarrs._omero import Omero, OmeroChannel, OmeroRenderingDefs, OmeroWindow

from ._bf2raw import Bf2Raw, Series
from ._image import (
    Dataset,
    Image,
    Multiscale,
    ScaleTransformation,
    TranslationTransformation,
)
from ._labels import (
    ImageLabel,
    LabelColor,
    LabelImage,
    LabelProperty,
    LabelSource,
)
from ._plate import (
    Acquisition,
    Column,
    FieldOfView,
    Plate,
    PlateDef,
    PlateWell,
    Row,
    Well,
    WellDef,
)
from ._zarr_json import OMEZarrGroupJSON

__all__ = [
    "Acquisition",
    "Axis",
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
