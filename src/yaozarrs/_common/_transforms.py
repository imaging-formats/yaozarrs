"""`scale` and `translation`, shared by the versions before `coordinateSystems`."""

from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias

from annotated_types import MinLen
from pydantic import (
    AfterValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    WithJsonSchema,
    model_validator,
)

from yaozarrs._base import _BaseModel
from yaozarrs._common._image import require_inline_or_path

__all__ = [
    "CoordinateTransformation",
    "CoordinateTransformsList",
    "ScaleTransformation",
    "TranslationTransformation",
]

# (in the JSON schema, these fields aren't nullable: a transformation has exactly one
# of them, which the models' `oneOf` says)
_VECTOR_SCHEMA = WithJsonSchema(
    {"type": "array", "items": {"type": "number"}, "minItems": 2}
)
_PATH_SCHEMA = WithJsonSchema({"type": "string"})
# a scale or translation vector, inline (or None: stored at `path`)
_Vector: TypeAlias = Annotated[Annotated[list[float], MinLen(2)] | None, _VECTOR_SCHEMA]


class ScaleTransformation(_BaseModel):
    """Maps array indices to physical coordinates via scaling.

    Defines the pixel/voxel size in physical units for each dimension.
    Every dataset must have exactly one scale transformation.

    !!! note
        Scale values represent physical size per pixel. For example, a scale of
        `[0.5, 0.5]` means each pixel is 0.5 units wide in physical space.
    """

    # (in the JSON schema too: exactly one of `scale` or `path`)
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={"oneOf": [{"required": ["scale"]}, {"required": ["path"]}]}
    )

    type: Literal["scale"] = "scale"
    # "scale vector, stored either as a list of floats (`scale`) or as binary
    # data at a location in this container (`path`)"
    scale: _Vector = Field(
        default=None,
        description=(
            "Scaling factor for each dimension in physical units per pixel "
            "(or see `path`)"
        ),
    )
    path: Annotated[str | None, _PATH_SCHEMA] = Field(
        default=None,
        description="Path to a Zarr array in this container holding the scale vector "
        "(instead of `scale`)",
    )

    @model_validator(mode="after")
    def _values_or_path(self) -> Self:
        require_inline_or_path(self, "scale", "vector")
        return self

    @property
    def ndim(self) -> int | None:
        """Number of dimensions in this transformation (None if stored at `path`)."""
        return None if self.scale is None else len(self.scale)


class TranslationTransformation(_BaseModel):
    """Translates the coordinate system origin in physical space.

    Specifies the physical coordinates of the origin (index [0, 0, ...]).
    At most one translation may be present per dataset, and it must appear
    after the scale transformation.
    """

    # (in the JSON schema too: exactly one of `translation` or `path`)
    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={
            "oneOf": [{"required": ["translation"]}, {"required": ["path"]}]
        }
    )

    type: Literal["translation"] = "translation"
    # "translation vector, stored either as a list of floats (`translation`) or as
    # binary data at a location in this container (`path`)"
    translation: _Vector = Field(
        default=None,
        description=(
            "Translation offset for each dimension in physical units (or see `path`)"
        ),
    )
    path: Annotated[str | None, _PATH_SCHEMA] = Field(
        default=None,
        description="Path to a Zarr array in this container holding the translation "
        "vector (instead of `translation`)",
    )

    @model_validator(mode="after")
    def _values_or_path(self) -> Self:
        require_inline_or_path(self, "translation", "vector")
        return self

    @property
    def ndim(self) -> int | None:
        """Number of dimensions in this transformation (None if stored at `path`)."""
        return None if self.translation is None else len(self.translation)


def _transform_tag(v: Any) -> str | None:
    """The union member for a transformation: by its `type`, else by its values."""
    if isinstance(v, dict):
        # (`type` may be omitted, e.g. by `DimSpec`: then the values say which it is)
        kind = v.get("type", "translation" if "translation" in v else "scale")
    else:
        kind = getattr(v, "type", None)
    return _TRANSFORM_TAGS.get(kind, kind) if isinstance(kind, str) else None


_TRANSFORM_TAGS = {
    "scale": "ScaleTransformation",
    "translation": "TranslationTransformation",
}
# (discriminated, so that errors are reported for the one member a transformation is,
# under its class name)
CoordinateTransformation: TypeAlias = Annotated[
    Annotated[ScaleTransformation, Tag("ScaleTransformation")]
    | Annotated[TranslationTransformation, Tag("TranslationTransformation")],
    Discriminator(_transform_tag),
]


def _validate_transforms_list(
    transforms: list[CoordinateTransformation],
) -> list[CoordinateTransformation]:
    # [the list of transforms] MUST contain exactly one scale transformation that
    # specifies the pixel size in physical units or time duration
    scales = [t for t in transforms if isinstance(t, ScaleTransformation)]
    if len(scales) != 1:
        raise ValueError(
            "There must be exactly one scale transformation in the list of transforms. "
            f"Found {len(scales)}.\n\n"
            "TIP:\n"
            "If scaling information is not available or applicable for one of the axes,"
            " the value MUST express the scaling factor between the current resolution "
            "and the first resolution for the given axis, defaulting to 1.0 if there is"
            " no downsampling along the axis"
        )

    # It MAY contain exactly one translation that specifies the offset from the origin
    # in physical units.
    translations = [t for t in transforms if isinstance(t, TranslationTransformation)]
    if len(translations) > 1:
        raise ValueError(
            "There can be at most one translation transformation in the list of "
            f"transforms. Found {len(translations)}."
        )

    # If translation is given it MUST be listed after scale to ensure that it is given
    # in physical coordinates.
    if translations and transforms.index(translations[0]) < transforms.index(scales[0]):
        raise ValueError(
            "If a translation transformation is given, it must be listed after "
            "the scale transformation."
        )

    # The length of the scale and translation array MUST be the same as the length of
    # "axes" (checked by Multiscale), so they must also match each other.
    # (one stored at a `path` has no length to compare)
    if len({t.ndim for t in transforms if t.ndim is not None}) > 1:
        lengths = ", ".join(f"{t.type} has {t.ndim}" for t in transforms)
        raise ValueError(
            "The scale and translation transformations must have the same length "
            f"(one value per axis), but {lengths}."
        )

    return transforms


CoordinateTransformsList: TypeAlias = Annotated[
    list[CoordinateTransformation],
    MinLen(1),
    AfterValidator(_validate_transforms_list),
]
