"""A zarr.json document found in any ome-zarr v0.6 group.

https://github.com/ome/ngff-spec (the `main` branch IS the in-development 0.6)

The set of documents you might encounter is the same as v0.5, with two additions
in v0.6:

- a brand-new top-level `scene` object (cross-image coordinate systems +
  transformations), and
- a redesigned image multiscale (named `coordinateSystems` instead of `axes`,
  and dataset transforms carrying `input`/`output`).

See `yaozarrs.v05._zarr_json` for the full annotated catalogue of group types;
the structure is unchanged aside from the per-document differences noted in the
individual v0.6 models.
"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Discriminator, Tag

from yaozarrs._base import ZarrGroupModel, _BaseModel

from ._bf2raw import Bf2Raw, Series
from ._image import Image
from ._labels import LabelImage, LabelsGroup
from ._plate import Plate, Well
from ._scene import Scene


def _discriminate_ome_v06_metadata(v: Any) -> str | None:
    if isinstance(v, dict):
        if "image-label" in v:
            return "label-image"
        if "multiscales" in v:
            return "image"
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
        if isinstance(v, Series):  # pragma: no cover
            return "series"
        if isinstance(v, Scene):  # pragma: no cover
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
