"""Scene model for OME-NGFF v0.6 (new, experimental).

A *scene* combines coordinate systems and coordinate transformations to express
spatial relationships *between* images (e.g. registering several multiscale
datasets into a common world coordinate system). It is a brand-new top-level
object in v0.6 (`scene.schema`), adopted from RFC-5 in the 0.6rc0 release.

!!! warning "Experimental / unstable"
    This object is modeled for completeness (it appears in the `ome_zarr` root
    union) but the spec around it is still actively changing.

!!! note "Change from 0.6.dev4"
    `arrayCoordinateSystem` was removed in 0.6rc0 (replaced by prose explaining
    how to express dimensionless transforms) and is no longer modeled here --
    it round-trips as a dropped/ignored extra field on older documents. `scene`
    documents now also require a `version` field (it was previously absent from
    `scene.schema`'s `required` list, presumably an oversight).
"""

from typing import Annotated, TypeAlias

from annotated_types import MinLen
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Self

from yaozarrs._base import _BaseModel

from ._coordinate_systems import CoordinateSystems
from ._transforms import Transformation, _validate_unique_transform_names
from ._version import CURRENT_VERSION, OMEV06

__all__ = ["Scene", "SceneDef"]


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
    AfterValidator(_validate_unique_transform_names),
]


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

    @model_validator(mode="after")
    def _validate_local_cs_refs(self) -> Self:
        # A transform endpoint given by `name` alone (no `path`) must resolve
        # to a coordinate system declared in this scene's own
        # `coordinateSystems`. An endpoint that also gives `path` refers to a
        # coordinate system declared in an external image group and can't be
        # checked without touching storage (see `_storage.visit_scene`).
        names = {cs.name for cs in (self.coordinateSystems or [])}
        for i, t in enumerate(self.coordinateTransformations):
            for side, io in (("input", t.input), ("output", t.output)):
                if io is not None and io.path is None and io.name not in names:
                    raise ValueError(
                        f"coordinateTransformations[{i}].{side}: coordinate "
                        f"system {io.name!r} is not declared in this scene's "
                        "'coordinateSystems' (and no 'path' was given)."
                    )
        return self


class Scene(_BaseModel):
    """Top-level `scene` metadata (combines coordinate systems + transforms)."""

    version: OMEV06 = Field(
        default=CURRENT_VERSION,
        description="OME-NGFF specification version",
    )
    scene: SceneDef = Field(description="Coordinate systems and transformations.")
