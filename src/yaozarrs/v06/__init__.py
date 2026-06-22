"""OME-NGFF v0.6 metadata models.

Specification (in development): the `main` branch of
<https://github.com/ome/ngff-spec> *is* the v0.6 (currently `0.6.dev4`) spec.

!!! warning "In-development version"
    v0.6 is still a development version (`0.6.dev4`). Models accept/emit the
    exact version string `"0.6.dev4"`. The headline change from v0.5 is the
    coordinate-systems + coordinate-transformations redesign (RFC-5): the
    multiscale `axes` field is replaced by named `coordinateSystems`, and
    dataset transforms carry `input`/`output`.
"""

from yaozarrs._omero import Omero, OmeroChannel, OmeroRenderingDefs, OmeroWindow

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
from ._scene import ArrayCoordinateSystem, Scene, SceneDef
from ._transforms import (
    AffineTransformation,
    BijectionTransformation,
    ByDimensionTransformation,
    CoordinatesTransformation,
    DisplacementsTransformation,
    IdentityTransformation,
    InputOutput,
    MapAxisTransformation,
    RotationTransformation,
    ScaleTransformation,
    SequenceTransformation,
    Transformation,
    TranslationTransformation,
)
from ._zarr_json import OMEAttributes, OMEMetadata, OMEZarrGroupJSON

__all__ = [
    "Acquisition",
    "AffineTransformation",
    "ArrayCoordinateSystem",
    "Axis",
    "Bf2Raw",
    "BijectionTransformation",
    "ByDimensionTransformation",
    "ChannelAxis",
    "Column",
    "CoordinateSystem",
    "CoordinatesTransformation",
    "CustomAxis",
    "Dataset",
    "DisplacementsTransformation",
    "FieldOfView",
    "IdentityTransformation",
    "Image",
    "ImageLabel",
    "InputOutput",
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
    "Omero",
    "OmeroChannel",
    "OmeroRenderingDefs",
    "OmeroWindow",
    "Plate",
    "PlateDef",
    "PlateWell",
    "RotationTransformation",
    "Row",
    "ScaleTransformation",
    "Scene",
    "SceneDef",
    "SequenceTransformation",
    "Series",
    "SpaceAxis",
    "TimeAxis",
    "Transformation",
    "TranslationTransformation",
    "Well",
    "WellDef",
]
