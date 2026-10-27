"""OME-NGFF v0.5 metadata models.

Specification: <https://ngff.openmicroscopy.org/0.5/>

Schema: <https://github.com/ome/ngff/tree/8cbba216e37407bd2d4bd5c7128ab13bd0a6404e>
"""

# FROZEN: v0.5 is a released spec. Models here should only change through
# deliberate bug fixes, committed separately (e.g. `fix(v05): ...`) together with
# the test/snapshot updates they require. Never import from another version package,
# and never modify shared building blocks (`yaozarrs._common`) to suit a newer
# version: fork the class into the newer version instead ("fork, don't flag").
# Guarded by tests/test_frozen_schemas.py and tests/test_architecture.py.

from yaozarrs._axis import Axis, ChannelAxis, CustomAxis, SpaceAxis, TimeAxis
from yaozarrs._common._omero import Omero, OmeroChannel, OmeroRenderingDefs, OmeroWindow

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
from ._zarr_json import OMEAttributes, OMEMetadata, OMEZarrGroupJSON

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
    "LabelsGroup",
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
    "Row",
    "ScaleTransformation",
    "Series",
    "SpaceAxis",
    "TimeAxis",
    "TranslationTransformation",
    "Well",
    "WellDef",
]
