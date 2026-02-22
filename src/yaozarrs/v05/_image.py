from collections.abc import Sequence
from typing import Annotated, Literal, TypeAlias

from annotated_types import MinLen
from pydantic import AfterValidator, Field, WrapValidator, model_validator
from typing_extensions import Self

from yaozarrs._axis import AxesList
from yaozarrs._base import _BaseModel
from yaozarrs._dim_spec import DimSpec
from yaozarrs._omero import Omero
from yaozarrs._types import UniqueList
from yaozarrs._util import SuggestDatasetPath

# ------------------------------------------------------------------------------
# Transformations model
# ------------------------------------------------------------------------------

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "Image",
    "Multiscale",
    "Dataset",
    "ScaleTransformation",
    "TranslationTransformation",
]


class ScaleTransformation(_BaseModel):
    """Maps array indices to physical coordinates via scaling.

    Defines the pixel/voxel size in physical units for each dimension.
    Every dataset must have exactly one scale transformation.

    !!! note
        Scale values represent physical size per pixel. For example, a scale of
        `[0.5, 0.5]` means each pixel is 0.5 units wide in physical space.
    """

    type: Literal["scale"] = "scale"
    scale: Annotated[list[float], MinLen(2)] = Field(
        description="Scaling factor for each dimension in physical units per pixel"
    )

    @property
    def ndim(self) -> int:
        """Number of dimensions in this transformation."""
        return len(self.scale)


class TranslationTransformation(_BaseModel):
    """Translates the coordinate system origin in physical space.

    Specifies the physical coordinates of the origin (index [0, 0, ...]).
    At most one translation may be present per dataset, and it must appear
    after the scale transformation.
    """

    type: Literal["translation"] = "translation"
    translation: Annotated[list[float], MinLen(2)] = Field(
        description="Translation offset for each dimension in physical units"
    )

    @property
    def ndim(self) -> int:
        """Number of dimensions in this transformation."""
        return len(self.translation)


CoordinateTransformation = ScaleTransformation | TranslationTransformation


def _validate_transforms_list(
    transforms: list[CoordinateTransformation],
) -> list[CoordinateTransformation]:
    # [the list of transforms] MUST contain exactly one scale transformation that
    # specifies the pixel size in physical units or time duration
    num_scales = len([t for t in transforms if isinstance(t, ScaleTransformation)])
    if num_scales != 1:
        raise ValueError(
            "There must be exactly one scale transformation in the list of transforms. "
            f"Found {num_scales}.\n\n"
            "TIP:\n"
            "If scaling information is not available or applicable for one of the axes,"
            " the value MUST express the scaling factor between the current resolution "
            "and the first resolution for the given axis, defaulting to 1.0 if there is"
            " no downsampling along the axis"
        )

    # It MAY contain exactly one translation that specifies the offset from the origin
    # in physical units.
    num_trans = len([t for t in transforms if isinstance(t, TranslationTransformation)])
    if num_trans > 1:
        raise ValueError(
            "There can be at most one translation transformation in the list of "
            f"transforms. Found {num_trans}."
        )

    # If translation is given it MUST be listed after scale to ensure that it is given
    # in physical coordinates.
    if num_trans:
        translation_idx = next(
            i
            for i, t in enumerate(transforms)
            if isinstance(t, TranslationTransformation)
        )
        scale_idx = next(
            i for i, t in enumerate(transforms) if isinstance(t, ScaleTransformation)
        )
        if translation_idx < scale_idx:
            raise ValueError(
                "If a translation transformation is given, it must be listed after "
                "the scale transformation."
            )

    return transforms


CoordinateTransformsList: TypeAlias = Annotated[
    list[CoordinateTransformation],
    MinLen(1),
    AfterValidator(_validate_transforms_list),
]

# ------------------------------------------------------------------------------
# Dataset model
# ------------------------------------------------------------------------------


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

    @property
    def translation_transform(self) -> TranslationTransformation | None:
        """Return the translation transformation from the list, if present.

        (CoordinateTransformsList validator ensures there is at most one.)
        """
        return next(
            (
                t
                for t in self.coordinateTransformations
                if isinstance(t, TranslationTransformation)
            ),
            None,
        )


def _validate_datasets_list(datasets: list[Dataset]) -> list[Dataset]:
    """Validate a list of Dataset for `Multiscale.datasets`."""
    # Each "datasets" dictionary MUST have the same number of dimensions...
    # NOTE: this wording from the spec is a bit ambiguous,
    # since the "number of dimensions" of a dataset is not explicitly defined.
    # Here we interpret it to mean the dimensionality of the coordinate transformations
    ndims = {dt.path: dt.coordinateTransformations[0].ndim for dt in datasets}
    if len(set(ndims.values())) != 1:
        raise ValueError(
            "All datasets must have the same number of dimensions. "
            f"Found differing dimensions: {ndims}"
        )
    # ... and MUST NOT have more than 5 dimensions.
    if any(n > 5 for n in ndims.values()):
        raise ValueError("Datasets must not have more than 5 dimensions.")

    return datasets


DatasetsList: TypeAlias = Annotated[
    UniqueList[Dataset],
    # NOTE: the MinLen(1) constraint comes from the image.schema,
    # but is not mentioned in the spec.
    MinLen(1),
    # hack to get around ordering of multiple after validators
    WrapValidator(lambda v, h: _validate_datasets_list(h(v))),
]

# ------------------------------------------------------------------------------
# Multiscale model
# ------------------------------------------------------------------------------


class Multiscale(_BaseModel):
    """Multi-resolution image pyramid (<=5D) with coordinate metadata.

    Defines a image at one ore more resolution levels, along with the
    coordinate system that relates array indices to physical space. This is
    the core metadata for any OME-NGFF image.

    !!! note "Resolution Ordering"
        Datasets must be ordered from highest to lowest resolution
        (i.e., finest to coarsest sampling).
    """

    name: str | None = Field(
        default=None,
        description="Optional identifier for this multiscale image",
    )
    axes: AxesList = Field(
        description="Ordered list of dimension axes defining the coordinate system"
    )
    datasets: DatasetsList = Field(
        description=(
            "Resolution pyramid levels, ordered from highest to lowest resolution"
        )
    )
    coordinateTransformations: CoordinateTransformsList | None = Field(
        default=None,
        description=(
            "Coordinate transformations that are applied to all resolution levels "
            "in the same manner."
        ),
    )

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

    @model_validator(mode="after")
    def _post_validate(self) -> Self:
        # The number and order of dimensions in each dataset MUST
        # correspond to number and order of "axes".
        # TODO ... this is ambiguous.  is it the same as the following check?

        # The length of the scale and translation array MUST be the same as the length
        # of "axes".
        for _id, ds in enumerate(self.datasets):
            for _it, transform in enumerate(ds.coordinateTransformations):
                if transform.ndim != self.ndim:
                    raise ValueError(
                        f"at datasets.[{_id}].coordinateTransformations[{_it}]:\n"
                        f"  The length of the transformation ({transform.ndim}) does "
                        f"not match the number of axes ({self.ndim})."
                    )
        if self.coordinateTransformations:
            for _it, transform in enumerate(self.coordinateTransformations):
                if transform.ndim != self.ndim:
                    raise ValueError(
                        f"at coordinateTransformations[{_it}]:\n"
                        f"  The length of the transformation ({transform.ndim}) "
                        f"does not match the number of axes ({self.ndim})."
                    )

        # The "paths" of the datasets MUST be be ordered from the highest resolution to
        # the lowest resolution (i.e. largest to smallest
        spatial_indices: dict[int, str] = {
            i: ax.name for i, ax in enumerate(self.axes) if ax.type == "space"
        }
        spatial_scales = [
            tuple(ds.scale_transform.scale[idx] for idx in spatial_indices)
            for ds in self.datasets
        ]
        if spatial_scales != sorted(spatial_scales):
            raise ValueError(
                "The datasets are not ordered from highest to lowest resolution. "
                f"Found spatial scales: {spatial_scales}"
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
        >>> from yaozarrs import DimSpec, v05
        >>> dims = [
        ...     DimSpec(name="t", size=512, unit="second"),
        ...     DimSpec(
        ...         name="z", size=50, scale=2.0, unit="micrometer", scale_factor=1.0
        ...     ),
        ...     DimSpec(name="y", size=512, scale=0.5, unit="micrometer"),
        ...     DimSpec(name="x", size=512, scale=0.5, unit="micrometer"),
        ... ]
        >>> v05.Multiscale.from_dims(dims, name="my_multiscale", n_levels=3)
        """
        from yaozarrs._dim_spec import _axes_datasets

        return cls(name=name, **_axes_datasets(dims, n_levels))  # type: ignore


# ------------------------------------------------------------------------------
# Image model
# ------------------------------------------------------------------------------


class Image(_BaseModel):
    """Top-level OME-NGFF image metadata.

    This model corresponds to the `zarr.json` file in an image group.
    It contains one or more multiscale pyramids plus optional OMERO rendering hints.

    !!! example "Typical Structure"
        ```
        my_image/
        ├── zarr.json          # contains ["ome"]["multiscales"]
        ├── 0/                 # Highest resolution array
        ├── 1/                 # Next resolution level
        └── labels/            # Optional segmentation masks
            ├── zarr.json      # contains ["ome"]["labels"]
            └── 0              # Multiscale, labeled image.
        ```

    !!! note
        For the optional `labels` group, see [LabelsGroup][yaozarrs.v05.LabelsGroup].
    """

    version: Literal["0.5"] = Field(
        default="0.5",
        description="OME-NGFF specification version",
    )
    multiscales: Annotated[UniqueList[Multiscale], MinLen(1)] = Field(
        description="One or more multiscale image pyramids in this group"
    )
    omero: Omero | None = Field(
        default=None,
        description="Optional OMERO rendering metadata for visualization",
    )
