"""OME-NGFF v0.6 metadata models.

Specification: <https://github.com/ome/ngff-spec>, targeting the `0.6rc0`
release-candidate tag (RFC-5 adopted: transformations, coordinate systems, and
`scene` metadata).

!!! warning "Release candidate"
    v0.6 is a release candidate (`0.6rc0`), not yet final. Models accept the
    whole **0.6 line** -- `"0.6"`, `"0.6.0"`, `"0.6rc*"`, and older `"0.6.dev*"`
    tags (for backwards parsing of pre-rc documents) -- and emit `"0.6rc0"` by
    default, so nothing needs bumping for a later rc tag. (A `0.6.Z` *patch*
    release such as `"0.6.1"` is rejected: it would be different content.) The
    headline change from v0.5 is the coordinate-systems +
    coordinate-transformations redesign (RFC-5): the multiscale `axes` field is
    replaced by named `coordinateSystems`, and dataset transforms carry
    `input`/`output`.
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
from ._scene import Scene, SceneDef
from ._transforms import (
    AffineTransformation,
    BijectionTransformation,
    ByDimensionTransformation,
    CoordinatesTransformation,
    DisplacementsTransformation,
    IdentityTransformation,
    InputOutput,
    MapAxisTransformation,
    ProjectAxisTransformation,
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
    "Transformation",
    "TranslationTransformation",
    "Well",
    "WellDef",
]
