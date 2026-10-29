from typing import Annotated

from annotated_types import MinLen
from pydantic import Field, model_validator
from typing_extensions import Self

from yaozarrs._base import _BaseModel

from ._coordinate_systems import CoordinateSystemsList
from ._transforms import (
    BijectionTransformation,
    CoordinateSystemRef,
    CoordinateTransformation,
    DisplacementsTransformation,
    IdentityTransformation,
    MapAxisTransformation,
    RotationTransformation,
    ScaleTransformation,
    TranslationTransformation,
    _output_ndim,
    _own_ndim,
    check_transform_ndim,
)
from ._version import VersionStr

__all__ = ["Scene", "SceneDef"]

# transformation types whose input and output always have the same dimensionality
_SAME_NDIM = (
    IdentityTransformation,
    MapAxisTransformation,
    TranslationTransformation,
    ScaleTransformation,
    RotationTransformation,
    BijectionTransformation,
    DisplacementsTransformation,
)


class SceneDef(_BaseModel):
    """Spatial relationships between images (the `scene` object).

    Transformations link coordinate systems of multiscale image subgroups (referenced
    by `path` and `name`) to each other, or to coordinate systems defined in the scene
    itself (referenced by `name` only).
    """

    # The `scene` object MUST contain the field `coordinateTransformations`, whose
    # value MUST be an array of valid transformations.
    coordinateTransformations: Annotated[list[CoordinateTransformation], MinLen(1)] = (
        Field(description="Transformations between coordinate systems of the images.")
    )
    # It MAY contain the field `coordinateSystems`, whose values MUST be an array of
    # valid coordinate systems.
    coordinateSystems: CoordinateSystemsList | None = Field(
        default=None,
        description=(
            "Coordinate systems defined by the scene itself (the first one is "
            "recommended as the default)."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        systems = {cs.name: cs for cs in self.coordinateSystems or ()}
        for i, t in enumerate(self.coordinateTransformations):
            loc = f"coordinateTransformations[{i}] ({t.type})"
            # the `input` and `output` fields of `coordinateTransformations` MUST
            # contain a json object, which MUST contain the field `name` and MAY
            # contain the field `path`.
            if t.input is None or t.output is None:
                raise ValueError(f"{loc}: must have both an 'input' and an 'output'.")
            for side, ref in (("input", t.input), ("output", t.output)):
                if ref.name is None:
                    raise ValueError(f"{loc}: {side} must specify a 'name'.")
                # If `name` refers to a coordinate system in the `scene` object, the
                # `path` value MAY be omitted or null.
                # (so a name without a path must be one of the scene's own coordinate
                # systems; names with a path are resolved in storage validation)
                if ref.path is None and ref.name not in systems:
                    raise ValueError(
                        f"{loc}: {side} {ref.name!r} has no 'path', so it must be one "
                        f"of the scene's coordinateSystems ({list(systems)})."
                    )
            try:
                self._check_ndim(t, systems)
            except ValueError as e:
                raise ValueError(f"{loc}: {e}") from None
        # NOTE: "The graph MUST be connected" applies to the coordinate systems of the
        # multiscales and scene metadata combined, so it needs the referenced images:
        # it's checked in storage validation.
        return self

    def _check_ndim(self, t: CoordinateTransformation, systems: dict) -> None:
        # (dimensionality is known for the scene's own coordinate systems; those of
        # image subgroups are checked in storage validation. Where only one side is
        # known, the other is inferred from the transformation, where possible.)
        def ndim(ref: CoordinateSystemRef | None) -> int | None:
            return systems[ref.name].ndim if ref and ref.path is None else None

        n_in, n_out = ndim(t.input), ndim(t.output)
        inferred = None
        if n_in is None and n_out is not None:
            n_in = n_out if isinstance(t, _SAME_NDIM) else _own_ndim(t)[0]
            inferred = "input"
        elif n_out is None and n_in is not None:
            n_out = _output_ndim(t, n_in)
            inferred = "output"
        if n_in is None or n_out is None:
            return
        try:
            check_transform_ndim(t, n_in, n_out)
        except ValueError as e:
            if inferred is None:
                raise
            raise ValueError(
                f"{e}; the {inferred} size here is inferred, as its coordinate system "
                "is in an image subgroup"
            ) from None


class Scene(_BaseModel):
    """Top-level scene metadata (`ome` in a scene group's `zarr.json`).

    A scene group contains several multiscale images that share a spatial
    relationship, described by transformations between their coordinate systems.

    !!! example "Typical Structure"
        ```
        my_scene.zarr
        ├── zarr.json                  # contains ["ome"]["scene"]
        ├── imageA/                    # a multiscale image
        ├── imageB/                    # another multiscale image
        └── coordinateTransformations/ # (optional) arrays with transform parameters
        ```
    """

    version: VersionStr = Field(default="0.6", description="OME-NGFF version")
    scene: SceneDef = Field(description="Spatial relationships between the images")
