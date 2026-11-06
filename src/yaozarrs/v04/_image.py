from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias

from annotated_types import MinLen
from pydantic import (
    AfterValidator,
    ConfigDict,
    Discriminator,
    Field,
    SerializerFunctionWrapHandler,
    Tag,
    WithJsonSchema,
    model_serializer,
    model_validator,
)

from yaozarrs._base import ZarrGroupModel, _BaseModel
from yaozarrs._common._axis import AxesList
from yaozarrs._common._image import (
    check_resolution_order,
    check_transform_lengths,
    require_inline_or_path,
)
from yaozarrs._common._node_names import SuggestDatasetPath
from yaozarrs._common._omero import Omero
from yaozarrs._dim_spec import DimSpec, _axes_datasets
from yaozarrs._types import UniqueList

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "Image",
    "Multiscale",
    "Dataset",
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
    # Must contain exactly one scale transformation
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

    # May contain at most one translation
    translations = [t for t in transforms if isinstance(t, TranslationTransformation)]
    if len(translations) > 1:
        raise ValueError(
            "There can be at most one translation transformation. "
            f"Found {len(translations)}."
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


class Dataset(_BaseModel):
    """A single resolution level in a multiscale image pyramid.

    Each dataset points to a Zarr array and defines how its indices map to
    physical coordinates. Together, multiple datasets form a resolution pyramid
    where each level represents the same physical region at different sampling rates.
    """

    path: Annotated[str, SuggestDatasetPath] = Field(
        description=(
            "Path to the Zarr array for this resolution level, "
            "relative to the parent multiscale group. All strings are allowed "
            "according to the spec, but prefer using only alphanumeric characters, "
            "dots (.), underscores (_), or hyphens (-) to avoid issues on some "
            "filesystems or when used in URLs."
        )
    )

    coordinateTransformations: CoordinateTransformsList = Field(
        description=(
            "Transformations mapping array indices to physical coordinates. "
            "Must include exactly one scale transformation, "
            "and optionally one translation."
        )
    )

    @property
    def scale_transform(self) -> ScaleTransformation:
        """Return the scale transformation from the list.

        (CoordinateTransformsList validator ensures there is exactly one.)
        """
        return next(
            t
            for t in self.coordinateTransformations
            if isinstance(t, ScaleTransformation)
        )


def _validate_datasets_list(datasets: list[Dataset]) -> list[Dataset]:
    """Validate a list of Dataset for `Multiscale.datasets`."""
    # Each dataset must have the same number of dimensions
    # (as far as the transformations show it: one stored at a `path` doesn't)
    ndims = {
        dt.path: n
        for dt in datasets
        if (n := next((t.ndim for t in dt.coordinateTransformations if t.ndim), None))
    }
    if not ndims:
        return datasets
    if len(set(ndims.values())) != 1:
        raise ValueError(
            "All datasets must have the same number of dimensions. "
            f"Found differing dimensions: {ndims}"
        )
    # Must not have more than 5 dimensions
    if any(n > 5 for n in ndims.values()):
        raise ValueError("Datasets must not have more than 5 dimensions.")

    return datasets


DatasetsList: TypeAlias = Annotated[
    list[Dataset],
    MinLen(1),
    AfterValidator(_validate_datasets_list),
]


class Multiscale(_BaseModel):
    """Multi-resolution image pyramid (<=5D) with coordinate metadata.

    Defines an image at one or more resolution levels, along with the
    coordinate system that relates array indices to physical space. This is
    the core metadata for any OME-NGFF image.
    """

    name: str | None = Field(
        default=None,
        description="Optional identifier for this multiscale image",
    )
    axes: AxesList = Field(
        description="Ordered list of dimension axes defining the coordinate system"
    )
    datasets: DatasetsList = Field(
        description=("Resolution pyramid levels, from highest to lowest resolution")
    )
    coordinateTransformations: CoordinateTransformsList | None = Field(
        default=None,
        description=(
            "Coordinate transformations that are applied to all resolution levels "
            "in the same manner."
        ),
    )
    version: Literal["0.4"] = "0.4"

    # NOTE: "type", and "metadata" mentioned in the spec, but NOT in image.schema
    type: str | None = Field(  # spec says SHOULD be present, missing in schema
        default=None,
        description=(
            "Type of downscaling method used to generate the multiscale image pyramid."
        ),
    )
    metadata: dict | None = Field(  # spec says SHOULD be present, missing in schema
        default=None,
        description="Unstructured key-value pair with additional "
        "information about the downscaling method.",
    )

    @model_serializer(mode="wrap")
    def serialize_model(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        """Always include `version` (even when dumping with `exclude_unset`)."""
        return {**handler(self), "version": self.version}

    @model_validator(mode="after")
    def _validate(self) -> Self:
        check_transform_lengths(
            self.datasets, self.coordinateTransformations, self.ndim
        )
        # The "path"s MUST be ordered from largest (i.e. highest resolution) to
        # smallest.
        space = [i for i, ax in enumerate(self.axes) if ax.type == "space"]
        # (as far as the scales are inline: one stored at a `path` can't be compared)
        check_resolution_order(
            [
                tuple(scale[i] for i in space)
                for ds in self.datasets
                if (scale := ds.scale_transform.scale) is not None
            ]
        )
        return self

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @classmethod
    def from_dims(
        cls,
        dims: Sequence[DimSpec],
        name: str | None = None,
        n_levels: int = 1,
    ) -> Self:
        """Convenience constructor: Create Multiscale from a sequence of DimSpec.

        Parameters
        ----------
        dims : Sequence[DimSpec]
            A sequence of dimension specifications defining the image dimensions.
            Must follow OME-Zarr axis ordering: `[time,] [channel,] space...`
        name : str | None, optional
            Name for the multiscale. Default is None.
        n_levels : int, optional
            Number of resolution levels in the pyramid. Default is 1.

        Returns
        -------
        Multiscale
            A fully configured Multiscale model.

        Examples
        --------
        >>> from yaozarrs import DimSpec, v04
        >>> dims = [
        ...     DimSpec(name="t", size=512, unit="second"),
        ...     DimSpec(
        ...         name="z", size=50, scale=2.0, unit="micrometer", scale_factor=1.0
        ...     ),
        ...     DimSpec(name="y", size=512, scale=0.5, unit="micrometer"),
        ...     DimSpec(name="x", size=512, scale=0.5, unit="micrometer"),
        ... ]
        >>> ms = v04.Multiscale.from_dims(dims, name="my_multiscale", n_levels=3)
        >>> [ds.path for ds in ms.datasets]
        ['0', '1', '2']
        """
        return cls(name=name, **_axes_datasets(dims, n_levels))  # type: ignore


class Image(ZarrGroupModel):
    """Top-level OME-NGFF v0.4 image metadata.

    This model corresponds to the `.zattrs` file in an image group
    (or `zarr.json` attributes in Zarr v3). It contains one or more multiscale
    pyramids plus optional OMERO rendering hints.

    !!! example "Typical Structure"
        ```
        my_image/
        ├── .zattrs            # contains {"multiscales": [...]}
        ├── 0/                 # Highest resolution array
        ├── 1/                 # Next resolution level
        └── labels/            # Optional segmentation masks
        ```

    !!! note "v0.4 vs v0.5"
        In v0.4, the version field is inside the multiscale objects rather than
        at the top level. See [`Multiscale`][yaozarrs.v04.Multiscale].
    """

    multiscales: Annotated[UniqueList[Multiscale], MinLen(1)] = Field(
        description="One or more multiscale image pyramids in this group"
    )
    omero: Omero | None = Field(
        default=None,
        description="Optional OMERO rendering metadata for visualization",
    )
