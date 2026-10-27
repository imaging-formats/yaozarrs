import math
import warnings
from typing import Annotated, Any, Literal, TypeAlias

from annotated_types import MinLen
from pydantic import (
    Discriminator,
    Field,
    NonNegativeInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from yaozarrs._base import _BaseModel
from yaozarrs._validation_warning import ValidationWarning

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "CoordinateTransformation",
    "CoordinateSystemRef",
    "IdentityTransformation",
    "MapAxisTransformation",
    "ProjectAxisTransformation",
    "TranslationTransformation",
    "ScaleTransformation",
    "AffineTransformation",
    "RotationTransformation",
    "SequenceTransformation",
    "ByDimensionTransformation",
    "ByDimensionItem",
    "BijectionTransformation",
    "DisplacementsTransformation",
    "CoordinatesTransformation",
]


class CoordinateSystemRef(_BaseModel):
    """Reference to a coordinate system: the `input` or `output` of a transformation.

    Which of `name` and `path` are required depends on where the transformation is
    stored (dataset, multiscale, or scene); the containing model checks that.
    If `path` is omitted (or null), `name` refers to a coordinate system in the same
    `zarr.json` document.
    """

    name: str | None = Field(default=None, description="Coordinate system name.")
    path: str | None = Field(
        default=None,
        description=(
            "Path to the Zarr group (or array) defining the coordinate system, "
            "if not in the same document."
        ),
    )


class _TransformBase(_BaseModel):
    # (narrowed to a Literal by each subclass, used for discrimination)
    type: str = Field(description="The type of the transformation.")
    name: str | None = Field(
        default=None, description="A unique name for this transformation."
    )
    # input/output are required "unless part of a wrapper transform (sequence,
    # bijection, byDimension)", so that is checked by the containing model.
    input: CoordinateSystemRef | None = Field(
        default=None, description="The input coordinate system."
    )
    output: CoordinateSystemRef | None = Field(
        default=None, description="The output coordinate system."
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_type_if_missing(cls, data: Any) -> Any:
        # allows e.g. `ScaleTransformation(scale=[1, 1])`, while still marking
        # `type` as explicitly set (so that it survives `exclude_unset` dumps).
        # (A dict *without* `type` passed to the CoordinateTransformation union is
        # still rejected, since the union discriminates on `type`.)
        if isinstance(data, dict) and "type" not in data:
            return {**data, "type": cls.model_fields["type"].default}
        return data


class IdentityTransformation(_TransformBase):
    """Maps input coordinates to output coordinates without modification."""

    type: Literal["identity"] = "identity"


class MapAxisTransformation(_TransformBase):
    """Axis permutation: `mapAxis[i]` is the input axis that becomes output axis `i`."""

    type: Literal["mapAxis"] = "mapAxis"
    mapAxis: Annotated[list[NonNegativeInt], MinLen(1)] = Field(
        description="New axis order, as zero-based indices of the input axes."
    )

    @field_validator("mapAxis")
    @classmethod
    def _unique(cls, v: list[int]) -> list[int]:
        # Each index MUST appear exactly once in the array.
        if len(set(v)) != len(v):
            raise ValueError(f"mapAxis indices must be unique, got {v}")
        return v


class ProjectAxisTransformation(_TransformBase):
    """Adds and/or drops dimensions at specified indices of the coordinate vector."""

    type: Literal["projectAxis"] = "projectAxis"
    droppedInputs: Annotated[list[NonNegativeInt], MinLen(1)] | None = Field(
        default=None,
        description="Indices of the input coordinate vector that are dropped.",
    )
    createdOutputs: Annotated[list[NonNegativeInt], MinLen(1)] | None = Field(
        default=None,
        description="Indices of the output coordinate vector that are added (as 0).",
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        # (schema: at least one of the two is required)
        if self.droppedInputs is None and self.createdOutputs is None:
            raise ValueError(
                "projectAxis requires at least one of 'droppedInputs' or "
                "'createdOutputs'"
            )
        # The values in the `createdOutputs` / `droppedInputs` array MUST be unique.
        for field in ("droppedInputs", "createdOutputs"):
            if (v := getattr(self, field)) is not None and len(set(v)) != len(v):
                raise ValueError(f"projectAxis {field} must be unique, got {v}")
        return self


class TranslationTransformation(_TransformBase):
    """Translates coordinates by a fixed offset along each axis."""

    type: Literal["translation"] = "translation"
    # (stored inline only: Zarr parameter storage was removed for scale/translation)
    translation: Annotated[list[float], MinLen(1)] = Field(
        description="Offset for each axis."
    )


class ScaleTransformation(_TransformBase):
    """Scales coordinates by a factor along each axis."""

    type: Literal["scale"] = "scale"
    # (stored inline only: Zarr parameter storage was removed for scale/translation)
    scale: Annotated[list[float], MinLen(1)] = Field(
        description="Scale factor for each axis."
    )

    @field_validator("scale")
    @classmethod
    def _warn_zero(cls, v: list[float]) -> list[float]:
        # Values in the `scale` array SHOULD be non-zero
        # (NOTE: the schema requires values > 0, but the prose allows negative
        # values, i.e. axis flips, so we follow the prose.)
        if 0 in v:
            warnings.warn(
                f"Scale values SHOULD be non-zero (got {v}); a zero scale is not "
                "invertible.",
                ValidationWarning,
                stacklevel=2,
            )
        return v


class AffineTransformation(_TransformBase):
    """Affine transformation, as an (M)x(N+1) matrix inline or in a Zarr array."""

    type: Literal["affine"] = "affine"
    affine: list[Annotated[list[float], MinLen(1)]] | None = Field(
        default=None,
        description="(M)x(N+1) matrix, as a list of M rows (for N input axes).",
    )
    path: str | None = Field(
        default=None, description="Path to a 2D Zarr array holding the matrix."
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        _require_exactly_one(self, "affine")
        # The matrix MUST be stored as 2D nested array ... where the outer array
        # MUST be length `M` and the inner arrays MUST be length `N+1`.
        if self.affine is not None:
            if not self.affine:
                raise ValueError("affine matrix must have at least one row")
            if len({len(row) for row in self.affine}) != 1:
                raise ValueError("affine matrix rows must all have the same length")
        return self


class RotationTransformation(_TransformBase):
    """Rotation, as an NxN matrix inline or in a Zarr array."""

    type: Literal["rotation"] = "rotation"
    rotation: list[Annotated[list[float], MinLen(1)]] | None = Field(
        default=None, description="NxN rotation matrix, as a list of N rows."
    )
    path: str | None = Field(
        default=None, description="Path to a 2D Zarr array holding the matrix."
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        _require_exactly_one(self, "rotation")
        if (m := self.rotation) is not None:
            # the outer array MUST be length `N` and the inner arrays MUST be length N
            if any(len(row) != len(m) for row in m):
                raise ValueError("rotation matrix must be square (NxN)")
            # [Rotations] MUST have determinant equal to one, with orthonormal rows
            # and columns.
            if not _is_rotation_matrix(m):
                raise ValueError(
                    "rotation matrix must be orthonormal with determinant 1"
                )
        return self


class SequenceTransformation(_TransformBase):
    """Applies an ordered list of transformations, first to last."""

    type: Literal["sequence"] = "sequence"
    transformations: Annotated[list["CoordinateTransformation"], MinLen(1)] = Field(
        description="Transformations to apply, in order (non-empty)."
    )


class ByDimensionItem(_BaseModel):
    """A lower-dimensional transformation applied to a subset of axes."""

    transformation: "CoordinateTransformation" = Field(
        description="The transformation to apply to the selected axes."
    )
    inputAxes: list[NonNegativeInt] = Field(
        description="Indices of the parent's input axes this transformation reads."
    )
    outputAxes: list[NonNegativeInt] = Field(
        description="Indices of the parent's output axes this transformation writes."
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        # The `inputAxes` and `outputAxes` arrays of each item MUST have the same
        # length as that transformation's parameter arrays.
        check_transform_ndim(
            self.transformation, len(self.inputAxes), len(self.outputAxes)
        )
        return self


class ByDimensionTransformation(_TransformBase):
    """Builds a transformation from lower-dimensional ones on subsets of axes."""

    type: Literal["byDimension"] = "byDimension"
    transformations: Annotated[list[ByDimensionItem], MinLen(1)] = Field(
        description="Transformations and the axes they apply to (non-empty)."
    )

    @field_validator("transformations")
    @classmethod
    def _unique_outputs(cls, v: list[ByDimensionItem]) -> list[ByDimensionItem]:
        # Every axis index in the parent byDimension's `output` coordinate system
        # MUST appear in exactly one child transformation's `outputAxes` array.
        # (that every axis is covered can only be checked knowing the output ndim)
        out = [ax for item in v for ax in item.outputAxes]
        if dupes := sorted({ax for ax in out if out.count(ax) > 1}):
            raise ValueError(
                f"byDimension output axes {dupes} appear in more than one "
                "transformation's 'outputAxes'"
            )
        return v


class BijectionTransformation(_TransformBase):
    """An invertible transformation with explicit `forward` and `inverse`."""

    type: Literal["bijection"] = "bijection"
    forward: "CoordinateTransformation" = Field(
        description="The forward transformation."
    )
    inverse: "CoordinateTransformation" = Field(
        description="The inverse transformation."
    )

    @model_validator(mode="after")
    def _warn_invertible(self) -> Self:
        # Each direction SHOULD be a transformation type that is not closed-form
        # invertible.
        for direction in ("forward", "inverse"):
            t = getattr(self, direction)
            if _is_closed_form_invertible(t):
                warnings.warn(
                    f"bijection {direction} transformation ({t.type!r}) is "
                    "closed-form invertible; each direction SHOULD be a type that "
                    "is not.",
                    ValidationWarning,
                    stacklevel=2,
                )
        return self


class _FieldTransformBase(_TransformBase):
    path: str = Field(description="Path to the coordinate/displacement field.")
    # The interpolation attributes MAY be provided. (default: "linear")
    # NOTE: the schema enum is ["nearest", "linear", "cubic"], but the prose lists
    # "bspline-cubic" and says the list is not exhaustive; we accept any string.
    interpolation: (
        Literal["nearest", "linear", "bspline-cubic", "cubic"] | str | None
    ) = Field(
        default=None,
        description=(
            "Interpolation method for the field (if omitted, 'linear'). Not "
            "normative: consumers may use a different method."
        ),
    )


class DisplacementsTransformation(_FieldTransformBase):
    """Displacement field: output = input + displacement vector at that point."""

    type: Literal["displacements"] = "displacements"


class CoordinatesTransformation(_FieldTransformBase):
    """Coordinate field: output = coordinate vector stored at the input point."""

    type: Literal["coordinates"] = "coordinates"


CoordinateTransformation: TypeAlias = Annotated[
    IdentityTransformation
    | MapAxisTransformation
    | ProjectAxisTransformation
    | TranslationTransformation
    | ScaleTransformation
    | AffineTransformation
    | RotationTransformation
    | SequenceTransformation
    | ByDimensionTransformation
    | BijectionTransformation
    | DisplacementsTransformation
    | CoordinatesTransformation,
    Discriminator("type"),
]
# Resolve the forward references to CoordinateTransformation.  Older pydantic (2.5)
# needs these explicit rebuilds, but DON'T rebuild BijectionTransformation (which
# has its own model validator) explicitly: on newer pydantic (2.13) that makes its
# validator run twice when validating through the union.  It gets completed anyway.
# (guarded by test_bijection_of_invertible_types_warns in tests/v06)
SequenceTransformation.model_rebuild()
ByDimensionItem.model_rebuild()


# ------------------------------------------------------------------------------
# Dimensionality checks (need the dimensionality of the input/output coordinate
# systems, so they are called by the models that can resolve them)
# ------------------------------------------------------------------------------


def check_transform_ndim(t: CoordinateTransformation, n_in: int, n_out: int) -> None:
    """Check that the parameters of `t` fit `n_in` input and `n_out` output axes.

    Raises
    ------
    ValueError
        If they don't.
    """

    # The parameter values ... MUST be compatible with input and output space
    # dimensionality
    def _fail(msg: str) -> None:
        raise ValueError(f"{t.type}: {msg} (input has {n_in}, output has {n_out} axes)")

    if isinstance(t, (TranslationTransformation, ScaleTransformation)):
        # Input and output dimensionality MUST be identical and MUST equal the
        # length of the `scale`/`translation` array (N).
        params = t.translation if isinstance(t, TranslationTransformation) else t.scale
        if not len(params) == n_in == n_out:
            _fail(f"has {len(params)} values")
    elif isinstance(t, (IdentityTransformation, DisplacementsTransformation)):
        # (`displacements` require M=N)
        if n_in != n_out:
            _fail("input and output must have the same number of axes")
    elif isinstance(t, MapAxisTransformation):
        # The length of the array MUST equal the number of dimensions in both the
        # input and output coordinate systems. Each integer in the array MUST be a
        # valid zero-based index into the input coordinate system's axes
        if not len(t.mapAxis) == n_in == n_out:
            _fail(f"has {len(t.mapAxis)} indices")
        if sorted(t.mapAxis) != list(range(n_in)):
            _fail(f"{t.mapAxis} is not a permutation of 0..{n_in - 1}")
    elif isinstance(t, ProjectAxisTransformation):
        dropped, created = t.droppedInputs or [], t.createdOutputs or []
        if any(i >= n_in for i in dropped):
            _fail(f"droppedInputs {dropped} out of range")
        if any(i >= n_out for i in created):
            _fail(f"createdOutputs {created} out of range")
        if n_out != n_in - len(dropped) + len(created):
            _fail(f"drops {len(dropped)} and creates {len(created)} axes")
    elif isinstance(t, AffineTransformation):
        # the outer array MUST be length `M` and the inner arrays MUST be length N+1
        if t.affine is not None:
            m, n1 = len(t.affine), len(t.affine[0])
            if m != n_out or n1 != n_in + 1:
                _fail(f"matrix is {m}x{n1}, expected {n_out}x{n_in + 1}")
    elif isinstance(t, RotationTransformation):
        # Input and output dimensionality (N) MUST be identical.
        if n_in != n_out:
            _fail("input and output must have the same number of axes")
        if t.rotation is not None and len(t.rotation) != n_in:
            _fail(f"matrix is {len(t.rotation)}x{len(t.rotation)}")
    elif isinstance(t, BijectionTransformation):
        # Its input and output spaces MUST have equal dimension. The input and
        # output dimensions for the both the forward and inverse transformations
        # MUST match bijection's input and output space dimensions.
        if n_in != n_out:
            _fail("input and output must have the same number of axes")
        _check_child(t, "forward", t.forward, n_in, n_out)
        _check_child(t, "inverse", t.inverse, n_out, n_in)
    elif isinstance(t, SequenceTransformation):
        # (intermediate dimensionality is inferred from each step's parameters;
        # checking stops at the first step where it can't be)
        n: int | None = n_in
        for i, step in enumerate(t.transformations):
            if (n_next := _output_ndim(step, n)) is None:
                return
            _check_child(t, f"transformations[{i}]", step, n, n_next)
            n = n_next
        if n != n_out:
            _fail(f"steps produce {n} axes")
    elif isinstance(t, ByDimensionTransformation):
        # The integer values in these arrays correspond to the axis indices in the
        # `byDimension`'s ... `input` and `output` coordinate systems
        for i, item in enumerate(t.transformations):
            if any(ax >= n_in for ax in item.inputAxes):
                _fail(f"transformations[{i}].inputAxes {item.inputAxes} out of range")
            if any(ax >= n_out for ax in item.outputAxes):
                _fail(f"transformations[{i}].outputAxes {item.outputAxes} out of range")
        # Every axis index in the parent byDimension's `output` coordinate system
        # MUST appear in exactly one child transformation's `outputAxes` array.
        covered = {ax for item in t.transformations for ax in item.outputAxes}
        if missing := sorted(set(range(n_out)) - covered):
            _fail(f"output axes {missing} are not covered by any transformation")


def _check_child(
    parent: _TransformBase, loc: str, child: Any, n_in: int, n_out: int
) -> None:
    try:
        check_transform_ndim(child, n_in, n_out)
    except ValueError as e:
        raise ValueError(f"{parent.type}.{loc} -> {e}") from None


def _output_ndim(t: Any, n_in: int | None) -> int | None:
    """Output dimensionality of `t` for `n_in` input axes, if it can be inferred."""
    if n_in is None:
        return None
    if isinstance(t, AffineTransformation):
        return len(t.affine) if t.affine is not None else None
    if isinstance(t, ProjectAxisTransformation):
        return n_in - len(t.droppedInputs or []) + len(t.createdOutputs or [])
    if isinstance(t, SequenceTransformation):
        for step in t.transformations:
            n_in = _output_ndim(step, n_in)
        return n_in
    if isinstance(t, BijectionTransformation):
        return _output_ndim(t.forward, n_in)
    if isinstance(t, (ByDimensionTransformation, CoordinatesTransformation)):
        return None  # can't tell from parameters alone
    return n_in  # all other types preserve dimensionality


def _require_exactly_one(t: Any, matrix_field: str) -> None:
    has_matrix, has_path = getattr(t, matrix_field) is not None, t.path is not None
    if has_matrix == has_path:
        raise ValueError(
            f"{t.type} requires exactly one of '{matrix_field}' (inline matrix) or "
            "'path' (Zarr array)"
        )


# tolerance for floating point checks of rotation matrices
_ROTATION_TOL = 1e-6


def _is_rotation_matrix(m: list[list[float]]) -> bool:
    n = len(m)
    # orthonormal rows (for a square matrix, this implies orthonormal columns)
    for i in range(n):
        for j in range(i, n):
            dot = sum(m[i][k] * m[j][k] for k in range(n))
            if not math.isclose(dot, float(i == j), abs_tol=_ROTATION_TOL):
                return False
    # orthonormal => det is +/-1; a rotation requires +1 (not a reflection)
    return math.isclose(_det(m), 1.0, abs_tol=_ROTATION_TOL)


def _det(m: list[list[float]]) -> float:
    a = [list(map(float, row)) for row in m]
    n, det = len(a), 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if a[p][c] == 0:
            return 0.0
        if p != c:
            a[c], a[p] = a[p], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            for k in range(c, n):
                a[r][k] -= f * a[c][k]
    return det


def _is_closed_form_invertible(t: Any) -> bool:
    if isinstance(t, SequenceTransformation):
        return all(_is_closed_form_invertible(s) for s in t.transformations)
    return isinstance(
        t,
        (
            IdentityTransformation,
            MapAxisTransformation,
            TranslationTransformation,
            ScaleTransformation,
            RotationTransformation,
            AffineTransformation,
        ),
    )
