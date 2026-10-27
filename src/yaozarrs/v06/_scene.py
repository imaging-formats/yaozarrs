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
        self._validate_connected(systems)
        return self

    def _check_ndim(self, t: CoordinateTransformation, systems: dict) -> None:
        # (dimensionality is known for the scene's own coordinate systems; those of
        # image subgroups are checked in storage validation)
        def ndim(ref: CoordinateSystemRef | None) -> int | None:
            return systems[ref.name].ndim if ref and ref.path is None else None

        n_in, n_out = ndim(t.input), ndim(t.output)
        if n_in is not None and n_out is not None:
            check_transform_ndim(t, n_in, n_out)
        elif (n := n_in or n_out) is not None and isinstance(t, _SAME_NDIM):
            check_transform_ndim(t, n, n)

    def _validate_connected(self, systems: dict) -> None:
        # The graph MUST be connected (but need not be complete): any two coordinate
        # systems in the metadata MUST be linked by a path of one or more coordinate
        # transformations
        # (nodes are the scene's own coordinate systems and the referenced image
        # coordinate systems; links are followed in both directions)
        def node(ref: CoordinateSystemRef | None) -> tuple[str | None, str | None]:
            # (input/output are known to be set here; see `_validate`)
            return (ref.path, ref.name) if ref else (None, None)

        links: dict[tuple, set[tuple]] = {(None, name): set() for name in systems}
        for t in self.coordinateTransformations:
            a, b = node(t.input), node(t.output)
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
        start = next(iter(links))
        reached, todo = {start}, [start]
        while todo:
            for n in links[todo.pop()] - reached:
                reached.add(n)
                todo.append(n)
        if unreached := [n for n in links if n not in reached]:
            names = [f"{p}:{n}" if p else n for p, n in unreached]
            raise ValueError(
                f"The scene's coordinate systems are not all linked by its "
                f"coordinateTransformations; not reachable from {start[1]!r}: {names}"
            )


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
