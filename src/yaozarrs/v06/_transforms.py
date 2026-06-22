"""Coordinate transformation models for OME-NGFF v0.6.

This is the headline addition of v0.6 (RFC-5). Where v0.5 only had `scale` and
`translation` transforms (each a flat list living under a dataset), v0.6 defines
a whole *coordinate-transformation graph*: transforms map between named
*coordinate systems* and there are many transform types.

`coordinate_transformations.schema` defines these transform types:

- `identity`
- `mapAxis`
- `scale`
- `translation`
- `affine`        (inline matrix OR a `path` to a zarr array)
- `rotation`      (inline matrix OR a `path` to a zarr array)
- `bijection`     (a `forward`/`inverse` pair)
- `sequence`      (an ordered list of transforms)
- `byDimension`   (per-dimension transforms)
- `displacements` (a displacement field stored in a zarr array)
- `coordinates`   (a coordinate field stored in a zarr array)

When a transform appears as a top-level item of a `coordinateTransformations`
array it additionally carries `input` and `output` (see
[`InputOutput`][yaozarrs.v06.InputOutput]) naming the source/target coordinate
systems. When nested (inside `sequence`/`bijection`/`byDimension`) it does not.

!!! warning "Pragmatic validation"
    Following the rest of `yaozarrs`, validation here is deliberately pragmatic.
    See `TRICKY_NOTES.md` in the repo for the (recorded) list of places where we
    do **not** fully enforce the letter of the spec (e.g. affine/rotation matrix
    shapes, `byDimension` axis references, full coordinate-system-graph
    connectivity).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Discriminator, Field, PositiveFloat, model_validator
from typing_extensions import Self

from yaozarrs._base import _BaseModel

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "InputOutput",
    "IdentityTransformation",
    "MapAxisTransformation",
    "ScaleTransformation",
    "TranslationTransformation",
    "AffineTransformation",
    "RotationTransformation",
    "BijectionTransformation",
    "SequenceTransformation",
    "ByDimensionTransformation",
    "DisplacementsTransformation",
    "CoordinatesTransformation",
    "Transformation",
]


class InputOutput(_BaseModel):
    """Reference to the `input`/`output` coordinate system of a transformation.

    A coordinate system is identified either by `name` (a coordinate system
    declared in the same metadata document) and/or by `path` (a relative,
    downward path to an external multiscale dataset). Which of `name`/`path` is
    required depends on *where* the transform is used, so that constraint is
    enforced by the containing model (e.g. a dataset transform requires
    `input.path` and `output.name`).
    """

    name: str | None = Field(
        default=None,
        description="Name of a coordinate system declared in this metadata document.",
    )
    path: str | None = Field(
        default=None,
        description="Relative downward path to an external multiscale dataset.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_bare_string(cls, v: Any) -> Any:
        # Tolerate the bare-string shorthand `"input": "in"` seen in some spec
        # examples (the schema requires the object form `{"name": "in"}`). We
        # accept MORE than the schema here; we always *emit* the object form.
        if isinstance(v, str):
            return {"name": v}
        return v


class _Transform(_BaseModel):
    """Fields common to every coordinate transformation."""

    name: str | None = Field(
        default=None, description="Optional name for this transformation."
    )
    input: InputOutput | None = Field(
        default=None, description="Source coordinate system of the transformation."
    )
    output: InputOutput | None = Field(
        default=None, description="Target coordinate system of the transformation."
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_type_if_missing(cls, v: Any) -> Any:
        # ensure `type` is always present (and counted as "set", so it survives
        # model_dump(exclude_unset=True) — the discriminated union needs it).
        if isinstance(v, dict) and "type" not in v and "type" in cls.model_fields:
            if (default := cls.model_fields["type"].default) is not None:
                v = {**v, "type": default}
        return v


class IdentityTransformation(_Transform):
    """Maps input coordinates directly to output coordinates, unchanged."""

    type: Literal["identity"] = "identity"


class MapAxisTransformation(_Transform):
    """Permute axes by mapping input axes to output axes (by zero-based index)."""

    type: Literal["mapAxis"] = "mapAxis"
    mapAxis: list[Annotated[int, Field(ge=0, le=4)]] = Field(
        description="New axis order as zero-based indices of the input axes.",
        min_length=2,
        max_length=5,
    )


class ScaleTransformation(_Transform):
    """Scales coordinates by a factor along each axis.

    !!! note "Change from v0.5"
        v0.6 requires every scale factor to be strictly positive
        (`exclusiveMinimum: 0`). v0.5 placed no such constraint and required a
        minimum length of 2; v0.6's `scale.schema` drops the length constraint
        (length is instead validated against the axes by the containing
        `Multiscale`).
    """

    type: Literal["scale"] = "scale"
    scale: list[PositiveFloat] = Field(
        description="Scaling factor for each dimension (must be > 0)."
    )

    @property
    def ndim(self) -> int:
        """Number of dimensions in this transformation."""
        return len(self.scale)


class TranslationTransformation(_Transform):
    """Shifts coordinates by an offset along each axis."""

    type: Literal["translation"] = "translation"
    translation: list[float] = Field(
        description="Translation offset for each dimension in physical units."
    )

    @property
    def ndim(self) -> int:
        """Number of dimensions in this transformation."""
        return len(self.translation)


class AffineTransformation(_Transform):
    """Affine transformation, given inline as a matrix or by `path` to a zarr array."""

    type: Literal["affine"] = "affine"
    affine: list[list[float]] | None = Field(
        default=None, description="Affine transformation matrix."
    )
    path: str | None = Field(
        default=None, description="Path to a zarr array containing the affine matrix."
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        if (self.affine is None) == (self.path is None):
            raise ValueError(
                "An affine transformation must provide exactly one of "
                "'affine' (inline matrix) or 'path'."
            )
        return self


class RotationTransformation(_Transform):
    """Rotation, given inline as an NxN matrix or by `path` to a zarr array.

    !!! note "Pragmatic validation"
        The spec restricts the inline matrix to a square NxN matrix with N in
        2..5. We accept any nested list of numbers and do not enforce squareness
        (recorded in `TRICKY_NOTES.md`).
    """

    type: Literal["rotation"] = "rotation"
    rotation: list[list[float]] | None = Field(
        default=None, description="Rotation matrix (NxN, N in 2..5)."
    )
    path: str | None = Field(
        default=None, description="Path to a zarr array containing the rotation matrix."
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        if (self.rotation is None) == (self.path is None):
            raise ValueError(
                "A rotation transformation must provide exactly one of "
                "'rotation' (inline matrix) or 'path'."
            )
        return self


class BijectionTransformation(_Transform):
    """A pair of `forward` and `inverse` coordinate transformations."""

    type: Literal["bijection"] = "bijection"
    forward: Transformation = Field(description="The forward transformation.")
    inverse: Transformation = Field(description="The inverse transformation.")


class SequenceTransformation(_Transform):
    """An ordered sequence of transformations applied in order."""

    type: Literal["sequence"] = "sequence"
    transformations: list[Transformation] = Field(
        description="Transformations applied in order."
    )


class ByDimensionItem(_BaseModel):
    """One entry of a `byDimension` transformation."""

    transformation: Transformation = Field(
        description="The transformation applied to the referenced axes."
    )
    # NOTE (v0.6): the schema types these items as `number` even though the prose
    # describes them as axis names/indices (recorded in TRICKY_NOTES.md). We follow
    # the schema and accept numbers.
    input_axes: list[float] = Field(description="Input axes for this transformation.")
    output_axes: list[float] = Field(description="Output axes for this transformation.")


class ByDimensionTransformation(_Transform):
    """A set of transformations applied independently to subsets of dimensions."""

    type: Literal["byDimension"] = "byDimension"
    transformations: list[ByDimensionItem] = Field(
        description="Per-dimension transformations."
    )


class DisplacementsTransformation(_Transform):
    """Transformation defined by a displacement field stored in a zarr array."""

    type: Literal["displacements"] = "displacements"
    path: str = Field(description="Path to the zarr array with the displacement field.")
    interpolation: Literal["nearest", "linear", "cubic"] = Field(
        default="linear",
        description="Interpolation method used when applying the displacement field.",
    )


class CoordinatesTransformation(_Transform):
    """Transformation defined by a coordinate field stored in a zarr array."""

    type: Literal["coordinates"] = "coordinates"
    path: str = Field(description="Path to the zarr array with the coordinate field.")
    interpolation: Literal["nearest", "linear", "cubic"] = Field(
        default="linear",
        description="Interpolation method used when applying the coordinate field.",
    )


Transformation: TypeAlias = Annotated[
    IdentityTransformation
    | MapAxisTransformation
    | ScaleTransformation
    | TranslationTransformation
    | AffineTransformation
    | RotationTransformation
    | BijectionTransformation
    | SequenceTransformation
    | ByDimensionTransformation
    | DisplacementsTransformation
    | CoordinatesTransformation,
    Discriminator("type"),
]
"""Discriminated union over every v0.6 coordinate transformation `type`."""


# resolve the forward references used by the recursive transforms
BijectionTransformation.model_rebuild()
SequenceTransformation.model_rebuild()
ByDimensionItem.model_rebuild()
ByDimensionTransformation.model_rebuild()
