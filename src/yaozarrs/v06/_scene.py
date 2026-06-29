"""Scene model for OME-NGFF v0.6 (new, experimental).

A *scene* combines coordinate systems and coordinate transformations to express
spatial relationships *between* images (e.g. registering several multiscale
datasets into a common world coordinate system). It is a brand-new top-level
object in v0.6 (`scene.schema`) and is still in flux:

- `arrayCoordinateSystem` is slated for removal (ngff-spec PR #151).
- the `input.path`/`output.path` constraints are being reworked
  (ngff-spec PRs #149, #137).

!!! warning "Experimental / unstable"
    This object is modeled for completeness (it appears in the `ome_zarr` root
    union) but the spec around it is actively changing. See `TRICKY_NOTES_v06.md`.
"""

from typing import Annotated, TypeAlias

from annotated_types import MinLen
from pydantic import AfterValidator, Field

from yaozarrs._base import _BaseModel

from ._coordinate_systems import CoordinateSystems
from ._transforms import Transformation
from ._version import OMEV06

__all__ = ["ArrayCoordinateSystem", "Scene", "SceneDef"]


def _validate_scene_io_names(
    transforms: list[Transformation],
) -> list[Transformation]:
    # In a scene, both input and output must reference a coordinate system by name.
    for i, t in enumerate(transforms):
        loc = f"coordinateTransformations[{i}]"
        if t.input is None or t.input.name is None:
            raise ValueError(f"{loc}: 'input' must provide a 'name'.")
        if t.output is None or t.output.name is None:
            raise ValueError(f"{loc}: 'output' must provide a 'name'.")
    return transforms


SceneTransformList: TypeAlias = Annotated[
    list[Transformation],
    MinLen(1),
    AfterValidator(_validate_scene_io_names),
]


class ArrayCoordinateSystem(_BaseModel):
    """A coordinate system whose axes are all of `type="array"`.

    !!! warning "Deprecated"
        Slated for removal from the spec (ngff-spec PR #151); modeled here only
        to round-trip existing documents.
    """

    name: str | None = Field(
        default=None, description="Name of the array coordinate space."
    )
    axes: list = Field(description="Axes, all of type 'array'.")


class SceneDef(_BaseModel):
    """The content of the `scene` metadata field."""

    coordinateTransformations: SceneTransformList = Field(
        description=(
            "Transformations defining spatial relationships between coordinate "
            "systems. Both `input` and `output` reference coordinate systems by "
            "`name`."
        )
    )
    coordinateSystems: CoordinateSystems | None = Field(
        default=None,
        description="Coordinate systems combined with the transforms.",
    )
    arrayCoordinateSystem: ArrayCoordinateSystem | None = Field(
        default=None,
        description="(Deprecated) array coordinate system; being removed from spec.",
    )


class Scene(_BaseModel):
    """Top-level `scene` metadata (combines coordinate systems + transforms).

    !!! note "Version field"
        `scene.schema` does not list `version` under its `ome` object (likely an
        oversight). For consistency with every other v0.6 document, `yaozarrs`
        keeps a `version` field here, defaulting to "0.6.dev4". See
        `TRICKY_NOTES_v06.md`.
    """

    version: OMEV06 = Field(
        default="0.6.dev4",
        description="OME-NGFF specification version",
    )
    scene: SceneDef = Field(description="Coordinate systems and transformations.")
