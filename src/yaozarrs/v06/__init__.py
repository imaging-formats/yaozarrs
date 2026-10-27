"""OME-NGFF v0.6 metadata models.

Specification: <https://ngff.openmicroscopy.org/specifications/0.6/index.html>

Schema: <https://github.com/ome/ngff-spec/tree/89da1da5b0491f83b7cdde7d123aef763a457029/schemas>
"""

from ._axes import Axis, ChannelAxis, CustomAxis, SpaceAxis, TimeAxis
from ._bf2raw import Bf2Raw, Series
from ._coordinate_systems import CoordinateSystem
from ._image import Dataset, Image, Multiscale
from ._labels import (
    ImageLabel,
    LabelColor,
    LabelImage,
    LabelProperty,
    LabelsGroup,
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
from ._scene import Scene, SceneDef
from ._transforms import (
    AffineTransformation,
    BijectionTransformation,
    ByDimensionItem,
    ByDimensionTransformation,
    CoordinatesTransformation,
    CoordinateSystemRef,
    CoordinateTransformation,
    DisplacementsTransformation,
    IdentityTransformation,
    MapAxisTransformation,
    ProjectAxisTransformation,
    RotationTransformation,
    ScaleTransformation,
    SequenceTransformation,
    TranslationTransformation,
)
from ._zarr_json import OMEAttributes, OMEMetadata, OMEZarrGroupJSON

__all__ = [
    "Acquisition",
    "AffineTransformation",
    "Axis",
    "Bf2Raw",
    "BijectionTransformation",
    "ByDimensionItem",
    "ByDimensionTransformation",
    "ChannelAxis",
    "Column",
    "CoordinateSystem",
    "CoordinateSystemRef",
    "CoordinateTransformation",
    "CoordinatesTransformation",
    "CustomAxis",
    "Dataset",
    "DisplacementsTransformation",
    "FieldOfView",
    "IdentityTransformation",
    "Image",
    "ImageLabel",
    "LabelColor",
    "LabelImage",
    "LabelProperty",
    "LabelSource",
    "LabelsGroup",
    "MapAxisTransformation",
    "Multiscale",
    "OMEAttributes",
    "OMEMetadata",
    "OMEZarrGroupJSON",
    "Plate",
    "PlateDef",
    "PlateWell",
    "ProjectAxisTransformation",
    "RotationTransformation",
    "Row",
    "ScaleTransformation",
    "Scene",
    "SceneDef",
    "SequenceTransformation",
    "Series",
    "SpaceAxis",
    "TimeAxis",
    "TranslationTransformation",
    "Well",
    "WellDef",
]
