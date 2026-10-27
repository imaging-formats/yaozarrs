"""The `zarr.json` document of an OME-Zarr v0.6 group, and the `ome` metadata union.

As in v0.5, OME metadata lives under `attributes.ome` in a group's `zarr.json`, and
the kind of group (image, label image, labels, plate, well, bioformats2raw, series,
or scene) is determined by which key that object contains.
"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Discriminator, Tag

from yaozarrs._base import ZarrGroupModel, _BaseModel

from ._bf2raw import Bf2Raw, Series
from ._image import Image
from ._labels import LabelImage, LabelsGroup
from ._plate import Plate, Well
from ._scene import Scene

__all__ = ["OMEAttributes", "OMEMetadata", "OMEZarrGroupJSON"]


def _discriminate_ome_v06_metadata(v: Any) -> str | None:
    if isinstance(v, dict):
        if "image-label" in v:
            return "label-image"
        if "multiscales" in v:
            return "image"
        # If the top-level group represents a plate, the `bioformats2raw.layout`
        # metadata will be present but the `plate` key MUST also be present, takes
        # precedence
        if "plate" in v:
            return "plate"
        if "bioformats2raw.layout" in v or "bioformats2raw_layout" in v:
            return "bf2raw"
        if "well" in v:
            return "well"
        if "labels" in v:
            return "labels-group"
        if "series" in v:
            return "series"
        if "scene" in v:
            return "scene"
    elif isinstance(v, BaseModel):
        if isinstance(v, LabelImage):
            return "label-image"
        if isinstance(v, Image):
            return "image"
        if isinstance(v, Plate):
            return "plate"
        if isinstance(v, Bf2Raw):
            return "bf2raw"
        if isinstance(v, Well):
            return "well"
        if isinstance(v, LabelsGroup):
            return "labels-group"
        if isinstance(v, Series):
            return "series"
        if isinstance(v, Scene):
            return "scene"
    return None


OMEMetadata: TypeAlias = Annotated[
    (
        Annotated[LabelImage, Tag("label-image")]
        | Annotated[Image, Tag("image")]
        | Annotated[Plate, Tag("plate")]
        | Annotated[Bf2Raw, Tag("bf2raw")]
        | Annotated[Well, Tag("well")]
        | Annotated[LabelsGroup, Tag("labels-group")]
        | Annotated[Series, Tag("series")]
        | Annotated[Scene, Tag("scene")]
    ),
    Discriminator(_discriminate_ome_v06_metadata),
]
"""Union type for anything that can live in the "ome" key of a v0.6 `zarr.json` file."""


class OMEAttributes(_BaseModel):
    """The attributes field of a `zarr.json` document in an ome-zarr group."""

    model_config = ConfigDict(extra="allow")

    ome: OMEMetadata


class OMEZarrGroupJSON(ZarrGroupModel):
    """A `zarr.json` document found in any ome-zarr group."""

    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"
    attributes: OMEAttributes
