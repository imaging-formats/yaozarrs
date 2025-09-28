from typing import Annotated, Literal, TypeAlias

from annotated_types import Len, MinLen
from pydantic import AfterValidator, Field, WrapValidator, model_validator
from typing_extensions import Self

from yaozarrs._base import _BaseModel
from yaozarrs._types import UniqueList
from yaozarrs._units import SpaceUnits, TimeUnits

# ------------------------------------------------------------------------------
# Axis model
# ------------------------------------------------------------------------------

AxisType: TypeAlias = Literal["space", "time", "channel"]


class _AxisBase(_BaseModel):
    name: str = Field(description="The name of the axis.")


class SpaceAxis(_AxisBase):
    type: Literal["space"] = "space"
    unit: SpaceUnits | None = None  # SHOULD


class TimeAxis(_AxisBase):
    type: Literal["time"] = "time"
    unit: TimeUnits | None = None  # SHOULD


class ChannelAxis(_AxisBase):
    type: Literal["channel"] = "channel"
    unit: str | None = None  # SHOULD


class CustomAxis(_AxisBase):
    type: str | None = None  # SHOULD
    unit: str | None = None  # SHOULD


# this union allows us to restrict units based on type.
# Use CustomAxis for any type/unit.
Axis: TypeAlias = SpaceAxis | TimeAxis | ChannelAxis | CustomAxis


def _validate_axes_list(axes: list[Axis]) -> list[Axis]:
    """Validate a list of Axis for `Multiscale.axes`."""
    # names MUST be unique within the list.
    names = [ax.name for ax in axes]
    if len(names) != len(set(names)):
        raise ValueError(f"Axis names must be unique. Found duplicates in {names}")

    # The "axes" MUST contain 2 or 3 entries of "type:space"
    # and MAY contain one additional entry of "type:time"
    # and MAY contain one additional entry of "type:channel" or a null / custom type.
    n_space_axes = len([ax for ax in axes if ax.type == "space"])
    if n_space_axes < 2 or n_space_axes > 3:
        raise ValueError("There must be 2 or 3 axes of type 'space'.")
    if len([ax for ax in axes if ax.type == "time"]) > 1:
        raise ValueError("There can be at most 1 axis of type 'time'.")
    if len([ax for ax in axes if ax.type == "channel"]) > 1:
        raise ValueError("There can be at most 1 axis of type 'channel'.")

    # The entries MUST be ordered by "type" where the "time" axis must come first (if
    # present), followed by the "channel" or custom axis (if present) and the axes of
    # type "space".
    type_order = {"time": 0, "channel": 1, None: 1, "space": 2}
    sorted_axes = sorted(axes, key=lambda ax: type_order.get(ax.type, 3))
    if axes != sorted_axes:
        raise ValueError(
            "Axes are not in the required order by type. "
            "Order must be [time,] [channel,] space."
        )
    return axes


AxesList: TypeAlias = Annotated[
    UniqueList[Axis],
    Len(min_length=2, max_length=5),
    # hack to get around ordering of multiple after validators
    WrapValidator(lambda v, h: _validate_axes_list(h(v))),
]

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
    path: str = Field(
        description=(
            "The path to the array for this resolution, "
            "relative to the current zarr group."
        )
    )
    coordinateTransformations: CoordinateTransformsList = Field(
        description=(
            "list of transformations that map the data coordinates to the physical "
            'coordinates (as specified by "axes") for this resolution level.'
        )
    )


def _validate_datasets_list(datasets: list[Dataset]) -> list[Dataset]:
    """Validate a list of Dataset for `Multiscale.datasets`."""
    # The "paths" of the datasets MUST be be ordered from the highest resolution to the
    # lowest resolution (i.e. largest to smallest
    ...  # (Cannot validate without I/O)

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
    """A multiscale representation of an image.

    Notes
    -----
    Additional constraints that are not verifiable without I/O:

    - The length of "axes" MUST be equal to the dimensionality of the zarr arrays
      storing the image data (see `datasets.path`).
    - The "dimension_names" attribute MUST be included in the zarr.json of the Zarr
      array of a multiscale level and MUST match the names in the "axes" metadata.
    - The order of axes in "axes" MUST match the order of dimensions in the zarr arrays.
    - The "paths" of the datasets MUST be be ordered from the highest resolution to the
      lowest resolution (i.e. largest to smallest)
    """

    name: str | None = None  # SHOULD be present.
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
        return self

    @property
    def ndim(self) -> int:
        return len(self.axes)


# ------------------------------------------------------------------------------
# Omero model
# ------------------------------------------------------------------------------


class OmeroWindow(_BaseModel):
    start: float
    min: float
    end: float
    max: float


class OmeroChannel(_BaseModel):
    window: OmeroWindow | None = None
    label: str | None = None
    family: str | None = None
    color: str | None = None
    active: bool | None = None
    inverted: bool | None = None
    coefficient: float | None = None


class Omero(_BaseModel):
    channels: list[OmeroChannel]
    id: int | None = None


# ------------------------------------------------------------------------------
# Image model
# ------------------------------------------------------------------------------


class Image(_BaseModel):
    version: Literal["0.5"] = "0.5"
    multiscales: Annotated[UniqueList[Multiscale], MinLen(1)]
    omero: Omero | None = None
