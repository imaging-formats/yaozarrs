"""Coordinate transformation models for OME-NGFF v0.6.

This is the headline addition of v0.6 (RFC-5). Where v0.5 only had `scale` and
`translation` transforms (each a flat list living under a dataset), v0.6 defines
a whole *coordinate-transformation graph*: transforms map between named
*coordinate systems* and there are many transform types.

`coordinate_transformations.schema` defines these transform types:

- `identity`
- `mapAxis`
- `projectAxis`  (new in 0.6rc0)
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
    Constraints that require resolving the input/output coordinate systems
    (parameter lengths vs. dimensionality, full coordinate-system-graph
    connectivity) are not enforced at this level; only locally-checkable rules
    are.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal, TypeAlias

from annotated_types import Interval, Len
from pydantic import (
    AfterValidator,
    Discriminator,
    Field,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Self

from yaozarrs._base import _BaseModel
from yaozarrs._types import UniqueList  # noqa: TC001
from yaozarrs._validation_warning import ValidationWarning

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "InputOutput",
    "IdentityTransformation",
    "MapAxisTransformation",
    "ProjectAxisTransformation",
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
    mapAxis: UniqueList[Annotated[int, Interval(ge=0, le=4)]] = Field(
        description="New axis order as zero-based indices of the input axes.",
        min_length=2,
        max_length=5,
    )


class ProjectAxisTransformation(_Transform):
    """Add or drop axes from a coordinate vector.

    !!! note "New in v0.6rc0"
        Added alongside the other RFC-5 transforms. `droppedInputs` removes
        (projects out) the given input axis indices; `createdOutputs` inserts
        zero-valued axes at the given output indices. At least one must be given.
    """

    type: Literal["projectAxis"] = "projectAxis"
    droppedInputs: (
        Annotated[UniqueList[Annotated[int, Interval(ge=0, le=4)]], Len(1, 3)] | None
    ) = Field(
        default=None,
        description="Indices of the input axes to drop.",
    )
    createdOutputs: (
        Annotated[UniqueList[Annotated[int, Interval(ge=0, le=4)]], Len(1, 3)] | None
    ) = Field(
        default=None,
        description="Indices where zero-valued axes are inserted in the output.",
    )

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        if self.droppedInputs is None and self.createdOutputs is None:
            raise ValueError(
                "A projectAxis transformation must provide at least one of "
                "'droppedInputs' or 'createdOutputs'."
            )
        return self


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
        # spec: MxN+1 matrix -> non-empty and rectangular (M/N vs. the coordinate
        # systems can only be checked where those are in scope).
        if self.affine is not None:
            if not self.affine or not self.affine[0]:
                raise ValueError("An affine matrix must be non-empty.")
            if len({len(row) for row in self.affine}) != 1:
                raise ValueError(
                    "All rows of an affine matrix must have the same length."
                )
        return self


class RotationTransformation(_Transform):
    """Rotation, given inline as an NxN matrix or by `path` to a zarr array."""

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
        # rotation.schema: the inline matrix must be square NxN with N in 2..5.
        # (determinant/orthonormality are not checked here.)
        if self.rotation is not None:
            n = len(self.rotation)
            if not 2 <= n <= 5 or any(len(row) != n for row in self.rotation):
                raise ValueError(
                    "A rotation matrix must be square (NxN) with N in 2..5, "
                    f"got rows of lengths {[len(r) for r in self.rotation]}."
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
    # NB: when a SequenceTransformation is found inside of a dataset's
    # `coordinateTransformations` list, the transformations may only be of type
    # `scale` or `translation`. This is enforced by the _validate_dataset_transform
    # validator in _image.py, not here.
    transformations: list[Transformation] = Field(
        description="Transformations applied in order.",
        min_length=1,  # spec prose: "A non-empty array of transformations."
    )


class ByDimensionItem(_BaseModel):
    """One entry of a `byDimension` transformation."""

    transformation: Transformation = Field(
        description="The transformation applied to the referenced axes."
    )
    # NOTE (v0.6rc0): the schema loosely types these as `number`, but the prose
    # says "arrays of integers" (axis indices). int coerces 1.0, so schema-valid
    # docs still parse. rc0 renamed these from snake_case (`input_axes`,
    # `output_axes`, as in 0.6.dev4) to camelCase (`inputAxes`/`outputAxes`) for
    # consistency with the rest of the spec; we alias to the new name for
    # serialization but keep accepting the old one on input (validate_by_name).
    input_axes: list[Annotated[int, Field(ge=0)]] = Field(
        alias="inputAxes",
        description="Input axes (indices) for this transformation.",
    )
    output_axes: list[Annotated[int, Field(ge=0)]] = Field(
        alias="outputAxes",
        description="Output axes (indices) for this transformation.",
    )

    @model_validator(mode="after")
    def _axes_match_params(self) -> Self:
        # spec: input_axes/output_axes "MUST have the same length as that
        # transformation's parameter arrays." Only locally checkable for
        # transforms with a same-length-in-and-out inline parameter array.
        t = self.transformation
        if isinstance(t, (ScaleTransformation, TranslationTransformation)):
            params = t.scale if isinstance(t, ScaleTransformation) else t.translation
            for field in ("input_axes", "output_axes"):
                if len(getattr(self, field)) != len(params):
                    raise ValueError(
                        f"{field} (length {len(getattr(self, field))}) must have "
                        f"the same length as the child {t.type!r} transformation's "
                        f"parameter array (length {len(params)})."
                    )
        return self


class ByDimensionTransformation(_Transform):
    """A set of transformations applied independently to subsets of dimensions."""

    type: Literal["byDimension"] = "byDimension"
    transformations: list[ByDimensionItem] = Field(
        description="Per-dimension transformations."
    )

    @model_validator(mode="after")
    def _unique_output_axes(self) -> Self:
        # spec: every output axis index "MUST appear in exactly one child
        # transformation's output_axes array." Full coverage of the output
        # coordinate system requires document-level context; duplicates are
        # checkable locally.
        seen: set[int] = set()
        for item in self.transformations:
            if dupes := seen.intersection(item.output_axes):
                raise ValueError(
                    f"Output axes {sorted(dupes)} appear in more than one "
                    "byDimension child transformation."
                )
            seen.update(item.output_axes)
        return self


def _warn_if_unknown_interpolation(v: str) -> str:
    # spec (index.md): the interpolation method list ("nearest", "linear",
    # "cubic") is explicitly non-exhaustive and non-normative (e.g. prose also
    # mentions "bspline-cubic"), so an unrecognized value is a warning, not a
    # rejection.
    if v not in ("nearest", "linear", "cubic"):
        warnings.warn(
            f"Unrecognized interpolation method {v!r}; the spec's list "
            "('nearest', 'linear', 'cubic') is non-exhaustive, so this is "
            "accepted, but check for typos.",
            ValidationWarning,
            stacklevel=3,
        )
    return v


class DisplacementsTransformation(_Transform):
    """Transformation defined by a displacement field stored in a zarr array."""

    type: Literal["displacements"] = "displacements"
    path: str = Field(description="Path to the zarr array with the displacement field.")
    interpolation: Annotated[
        Literal["nearest", "linear", "cubic"] | str,
        AfterValidator(_warn_if_unknown_interpolation),
    ] = Field(
        default="linear",
        description="Interpolation method used when applying the displacement field.",
    )


class CoordinatesTransformation(_Transform):
    """Transformation defined by a coordinate field stored in a zarr array."""

    type: Literal["coordinates"] = "coordinates"
    path: str = Field(description="Path to the zarr array with the coordinate field.")
    interpolation: Annotated[
        Literal["nearest", "linear", "cubic"] | str,
        AfterValidator(_warn_if_unknown_interpolation),
    ] = Field(
        default="linear",
        description="Interpolation method used when applying the coordinate field.",
    )


def _validate_unique_transform_names(
    transforms: list[Transformation],
) -> list[Transformation]:
    """Warn if transform `name`s repeat within the same list.

    !!! note "Relaxed in v0.6rc0"
        0.6.dev4's prose had a MUST here ("Its value MUST be unique across all
        `name` fields ... in the same list"); rc0 dropped that sentence and now
        only says a `name` is "a unique name for this transformation" with no
        stated scope or MUST. rc0's own `examples/scene/scene.json` (a "valid"
        fixture) repeats a transform name three times, so this can no longer be
        a hard error -- only a warning.
    """
    names = [t.name for t in transforms if t.name is not None]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        warnings.warn(
            f"Coordinate transformation names SHOULD be unique. Duplicates: {dupes}",
            ValidationWarning,
            stacklevel=3,
        )
    return transforms


Transformation: TypeAlias = Annotated[
    IdentityTransformation
    | MapAxisTransformation
    | ProjectAxisTransformation
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
