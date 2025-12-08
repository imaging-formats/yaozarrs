from collections.abc import Sequence
from typing import Annotated, Literal, TypeAlias

from annotated_types import MinLen
from pydantic import AfterValidator, Field, WrapValidator, model_validator
from typing_extensions import Self

from yaozarrs._axis import AxesList
from yaozarrs._base import ZarrGroupModel, _BaseModel
from yaozarrs._dim_spec import DimSpec
from yaozarrs._omero import Omero
from yaozarrs._types import UniqueList

# ------------------------------------------------------------------------------
# Transformations model
# ------------------------------------------------------------------------------


class ScaleTransformation(_BaseModel):
    type: Literal["scale"] = "scale"
    scale: Annotated[list[float], MinLen(2)]

    @property
    def ndim(self) -> int:
        return len(self.scale)


class TranslationTransformation(_BaseModel):
    type: Literal["translation"] = "translation"
    translation: Annotated[list[float], MinLen(2)]

    @property
    def ndim(self) -> int:
        return len(self.translation)


CoordinateTransformation = ScaleTransformation | TranslationTransformation


def _validate_transforms_list(
    transforms: list[CoordinateTransformation],
) -> list[CoordinateTransformation]:
    # Must contain exactly one scale transformation
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

    # May contain at most one translation
    num_trans = len([t for t in transforms if isinstance(t, TranslationTransformation)])
    if num_trans > 1:
        raise ValueError(
            f"There can be at most one translation transformation. Found {num_trans}."
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
    path: str = Field(
        description=(
            "The path to the array for this resolution, "
            "relative to the current zarr group."
        )
    )
    coordinateTransformations: CoordinateTransformsList = Field(
        description=(
            "List of transformations that map the data coordinates to the physical "
            'coordinates (as specified by "axes") for this resolution level.'
        )
    )


def _validate_datasets_list(datasets: list[Dataset]) -> list[Dataset]:
    """Validate a list of Dataset for `Multiscale.datasets`."""
    # Each dataset must have the same number of dimensions
    ndims = {dt.path: dt.coordinateTransformations[0].ndim for dt in datasets}
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
    UniqueList[Dataset],
    MinLen(1),
    WrapValidator(lambda v, h: _validate_datasets_list(h(v))),
]

# ------------------------------------------------------------------------------
# Multiscale model
# ------------------------------------------------------------------------------


class Multiscale(_BaseModel):
    """A multiscale representation of an image."""

    name: str | None = None
    axes: AxesList = Field(description="The axes of the image.")
    datasets: DatasetsList = Field(
        description="The arrays storing the individual resolution levels"
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

    @model_validator(mode="after")
    def _check_ndim(self) -> Self:
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
        >>> v04.Multiscale.from_dims(dims, name="my_multiscale", n_levels=3)
        """
        from yaozarrs._dim_spec import _axes_datasets

        return cls(name=name, **_axes_datasets(dims, n_levels))


# ------------------------------------------------------------------------------
# Image model
# ------------------------------------------------------------------------------


class Image(ZarrGroupModel):
    multiscales: Annotated[UniqueList[Multiscale], MinLen(1)]
    omero: Omero | None = None
