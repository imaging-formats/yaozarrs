from collections.abc import Sequence
from typing import Annotated, Literal, Self, TypeAlias

from annotated_types import MinLen
from pydantic import (
    AfterValidator,
    Field,
    model_validator,
)

from yaozarrs._base import _BaseModel, _OMEDocument
from yaozarrs._common._axis import AxesList
from yaozarrs._common._image import (
    check_resolution_order,
    check_transform_lengths,
)
from yaozarrs._common._node_names import SuggestDatasetPath
from yaozarrs._common._omero import Omero
from yaozarrs._common._transforms import (
    CoordinateTransformsList,
    ScaleTransformation,
    TranslationTransformation,
)
from yaozarrs._dim_spec import DimSpec, _axes_datasets
from yaozarrs._types import UniqueList

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "Image",
    "Multiscale",
    "Dataset",
    "ScaleTransformation",
    "TranslationTransformation",
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
    # ... and MUST NOT have more than 5 dimensions.
    if any(n > 5 for n in ndims.values()):
        raise ValueError("Datasets must not have more than 5 dimensions.")

    return datasets


DatasetsList: TypeAlias = Annotated[
    list[Dataset],
    # NOTE: the MinLen(1) constraint comes from the image.schema,
    # but is not mentioned in the spec.
    MinLen(1),
    AfterValidator(_validate_datasets_list),
]


class Multiscale(_BaseModel):
    """Multi-resolution image pyramid (<=5D) with coordinate metadata.

    Defines an image at one or more resolution levels, along with the
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
    def _validate(self) -> Self:
        # The number and order of dimensions in each dataset MUST
        # correspond to number and order of "axes".
        # TODO ... this is ambiguous.  is it the same as the following check?

        # The length of the scale and translation array MUST be the same as the length
        # of "axes".
        check_transform_lengths(
            self.datasets, self.coordinateTransformations, self.ndim
        )
        # The "paths" of the datasets MUST be ordered from the highest resolution to
        # the lowest resolution (i.e. largest to smallest)
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
        >>> from yaozarrs import DimSpec, v05
        >>> dims = [
        ...     DimSpec(name="t", size=512, unit="second"),
        ...     DimSpec(
        ...         name="z", size=50, scale=2.0, unit="micrometer", scale_factor=1.0
        ...     ),
        ...     DimSpec(name="y", size=512, scale=0.5, unit="micrometer"),
        ...     DimSpec(name="x", size=512, scale=0.5, unit="micrometer"),
        ... ]
        >>> ms = v05.Multiscale.from_dims(dims, name="my_multiscale", n_levels=3)
        >>> [ds.path for ds in ms.datasets]
        ['0', '1', '2']
        """
        return cls(name=name, **_axes_datasets(dims, n_levels))  # type: ignore


class Image(_OMEDocument):
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
