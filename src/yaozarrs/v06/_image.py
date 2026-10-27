import warnings
from typing import Annotated, Any, TypeAlias, cast

from annotated_types import Len, MinLen
from pydantic import Discriminator, Field, model_validator
from typing_extensions import Self

from yaozarrs._base import _BaseModel
from yaozarrs._common._omero import Omero
from yaozarrs._types import UniqueList
from yaozarrs._util import SuggestDatasetPath
from yaozarrs._validation_warning import ValidationWarning

from ._axes import Axis, validate_multiscale_axes
from ._coordinate_systems import CoordinateSystem, CoordinateSystemsList
from ._transforms import (
    CoordinateTransformation,
    IdentityTransformation,
    ScaleTransformation,
    SequenceTransformation,
    TranslationTransformation,
    check_transform_ndim,
)
from ._version import VersionStr

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "Image",
    "Multiscale",
    "Dataset",
]

# ------------------------------------------------------------------------------
# Dataset model
# ------------------------------------------------------------------------------

# The transformation MUST be one of the following:
# - A single scale or identity transformation
# - A sequence transformation containing one scale and one translation transformation.
DatasetTransformation: TypeAlias = Annotated[
    ScaleTransformation | IdentityTransformation | SequenceTransformation,
    Discriminator("type"),
]


class Dataset(_BaseModel):
    """A single resolution level in a multiscale image pyramid.

    Points to a Zarr array, and maps its array coordinates to the multiscale's
    "intrinsic" coordinate system with a single transformation.
    """

    path: Annotated[str, SuggestDatasetPath] = Field(
        description=(
            "Path to the Zarr array for this resolution level, relative to the "
            "multiscale group."
        )
    )
    coordinateTransformations: Annotated[
        list[DatasetTransformation], Len(min_length=1, max_length=1)
    ] = Field(
        description=(
            "Exactly one transformation (scale, identity, or a sequence of a scale "
            "followed by a translation) from the array to the intrinsic coordinate "
            "system."
        )
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        t = self.transform
        # [coordinateTransformations herein MUST] be restricted to a single `scale`,
        # `identity` or `sequence` of a scale followed by a translation transformation.
        if isinstance(t, SequenceTransformation):
            types = [step.type for step in t.transformations]
            if types != ["scale", "translation"]:
                raise ValueError(
                    "A dataset's sequence transformation must be a scale followed by "
                    f"a translation, got {types}."
                )
        # The `input` object MUST specify a `path` field matching the `path` field of
        # the dataset.
        if t.input is None or t.input.path is None:
            raise ValueError(
                f"Dataset {self.path!r}: the transformation's 'input' must specify "
                "the dataset 'path'."
            )
        if t.input.path != self.path:
            raise ValueError(
                f"Dataset {self.path!r}: the transformation's input path "
                f"{t.input.path!r} does not match the dataset path."
            )
        # The `name` field of `output` MUST be the `name` of a coordinate system.
        # (that it exists is checked by the Multiscale)
        if t.output is None or t.output.name is None:
            raise ValueError(
                f"Dataset {self.path!r}: the transformation's 'output' must specify "
                "the 'name' of the intrinsic coordinate system."
            )
        # The `name` field under `input` SHOULD be omitted. ... The `path` field of
        # `output` SHOULD be omitted.
        # (NOTE: another section says "MUST ... omit"; upstream's review intent was
        # "discouraged and ignored", see ngff-spec#117.)
        if t.input.name is not None or t.output.path is not None:
            warnings.warn(
                f"Dataset {self.path!r}: the transformation's input 'name' and output "
                "'path' SHOULD be omitted.",
                ValidationWarning,
                stacklevel=2,
            )
        return self

    @property
    def transform(self) -> DatasetTransformation:
        """The transformation from this dataset's array to the intrinsic system."""
        return self.coordinateTransformations[0]

    @property
    def output_name(self) -> str:
        """Name of the (intrinsic) coordinate system this dataset maps into."""
        # (guaranteed to be set by `_validate`)
        output = self.transform.output
        return cast("str", output.name if output else None)

    @property
    def scale_transform(self) -> ScaleTransformation | None:
        """The scale transformation (alone or in a sequence), if any."""
        t = self.transform
        if isinstance(t, SequenceTransformation):
            t = t.transformations[0]
        return t if isinstance(t, ScaleTransformation) else None

    @property
    def translation_transform(self) -> TranslationTransformation | None:
        """The translation transformation (in a sequence), if any."""
        t = self.transform
        if isinstance(t, SequenceTransformation):
            last = t.transformations[-1]
            return last if isinstance(last, TranslationTransformation) else None
        return None


# ------------------------------------------------------------------------------
# Multiscale model
# ------------------------------------------------------------------------------


# types allowed when transforming to a coordinate system in a child labels group
# (the labels section also allows a sequence of a scale and a translation, which the
# other two sections omit; see ome/ngff#639)
_LABEL_TRANSFORM_TYPES = ("identity", "scale", "translation")


class Multiscale(_BaseModel):
    """A multi-resolution image pyramid, with its coordinate systems.

    All datasets (resolution levels) map into the same "intrinsic" coordinate system,
    which represents the image in its native physical coordinates. Additional
    `coordinateTransformations` may link the intrinsic coordinate system to other
    coordinate systems of this multiscale, or to those of its label images.
    """

    name: str | None = Field(default=None, description="Name of the multiscale image.")
    coordinateSystems: Annotated[CoordinateSystemsList, MinLen(1)] = Field(
        description="Coordinate systems of this multiscale image."
    )
    # (entries are unique as in v05, although neither the schema nor the prose says so)
    datasets: Annotated[UniqueList[Dataset], MinLen(1)] = Field(
        description="Resolution levels, ordered from highest to lowest resolution."
    )
    coordinateTransformations: (
        Annotated[list[CoordinateTransformation], MinLen(1)] | None
    ) = Field(
        default=None,
        description=(
            "Transformations between the intrinsic coordinate system and other "
            "coordinate systems (of this multiscale or of its label images)."
        ),
    )
    type: str | None = Field(
        default=None,
        description="Downsampling method used to generate the multiscale image.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about the downsampling method.",
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        systems = {cs.name: cs for cs in self.coordinateSystems}

        # The following conditions apply to all coordinate systems inside multiscales
        # metadata: ...
        for cs in self.coordinateSystems:
            try:
                validate_multiscale_axes(cs.axes)
            except ValueError as e:
                raise ValueError(f"coordinateSystems[{cs.name!r}]: {e}") from None

        # The `name` field of `output` MUST be the `name` of a coordinate system. It
        # MUST be the same value for every resolution level in a single multiscales
        names = {ds.output_name for ds in self.datasets}
        if len(names) != 1:
            raise ValueError(
                "All datasets must output to the same (intrinsic) coordinate system, "
                f"found {sorted(names)}."
            )
        intrinsic = names.pop()
        if intrinsic not in systems:
            raise ValueError(
                f"The datasets' output coordinate system {intrinsic!r} is not "
                f"declared in coordinateSystems ({list(systems)})."
            )
        ndim = systems[intrinsic].ndim
        for i, ds in enumerate(self.datasets):
            try:
                check_transform_ndim(ds.transform, ndim, ndim)
            except ValueError as e:
                raise ValueError(f"datasets[{i}] ({ds.path!r}): {e}") from None

        self._validate_coordinate_transformations(systems, intrinsic)
        self._validate_connected(systems, intrinsic)
        self._validate_resolution_order(systems[intrinsic])
        return self

    def _validate_coordinate_transformations(
        self, systems: dict[str, CoordinateSystem], intrinsic: str
    ) -> None:
        for i, t in enumerate(self.coordinateTransformations or ()):
            loc = f"coordinateTransformations[{i}] ({t.type})"
            if t.input is None or t.output is None:
                raise ValueError(f"{loc}: must have both an 'input' and an 'output'.")
            if t.input.name is None or t.output.name is None:
                raise ValueError(f"{loc}: input and output must specify a 'name'.")

            # One of `input` or `output` MUST reference the "intrinsic" coordinate
            # system by `name` (`path` MAY be omitted or null).
            in_intrinsic = t.input.path is None and t.input.name == intrinsic
            out_intrinsic = t.output.path is None and t.output.name == intrinsic
            if not (in_intrinsic or out_intrinsic):
                raise ValueError(
                    f"{loc}: one of input or output must reference the intrinsic "
                    f"coordinate system {intrinsic!r}."
                )
            other = t.output if in_intrinsic else t.input

            if other.path is None:
                # The other MUST reference a named coordinate system in the same
                # multiscales group (by `name`, `path` MAY be omitted or null)
                # NOTE: we resolve names within this multiscale entry only. A group
                # may hold several multiscales, whose coordinate system names need not
                # be unique across entries (raised in review of ngff-spec#137, but not
                # resolved there).
                if other.name not in systems:
                    raise ValueError(
                        f"{loc}: coordinate system {other.name!r} is not declared in "
                        f"this multiscale's coordinateSystems ({list(systems)})."
                    )
                n_in = systems[t.input.name].ndim
                n_out = systems[t.output.name].ndim
            else:
                # ... or in a child labels group (by `name` and `path`).
                # When referencing a child labels group, the transformation MUST be
                # `identity`, `scale`, or `translation`.
                # (the labels section also allows a sequence of a scale and a
                # translation; we accept that too, see ome/ngff#639)
                if not _is_label_transform(t):
                    raise ValueError(
                        f"{loc}: a transformation to a child label image must be "
                        "identity, scale, translation, or a sequence of a scale and "
                        "a translation (in either order)."
                    )
                # (these all preserve dimensionality)
                n_in = n_out = systems[intrinsic].ndim
            try:
                check_transform_ndim(t, n_in, n_out)
            except ValueError as e:
                raise ValueError(f"{loc}: {e}") from None

    def _validate_connected(
        self, systems: dict[str, CoordinateSystem], intrinsic: str
    ) -> None:
        # The graph MUST be connected (but need not be complete): any two coordinate
        # systems in the metadata MUST be linked by a path of one or more coordinate
        # transformations
        # (dataset transformations link the arrays to the intrinsic coordinate system;
        # links are followed in both directions)
        links: dict[str, set[str]] = {name: set() for name in systems}
        for t in self.coordinateTransformations or ():
            a = t.input.name if t.input and t.input.path is None else None
            b = t.output.name if t.output and t.output.path is None else None
            if a is not None and b is not None and a in links and b in links:
                links[a].add(b)
                links[b].add(a)
        reached, todo = {intrinsic}, [intrinsic]
        while todo:
            for name in links[todo.pop()] - reached:
                reached.add(name)
                todo.append(name)
        if unreached := [name for name in systems if name not in reached]:
            raise ValueError(
                f"Coordinate systems {unreached} are not linked to the intrinsic "
                f"coordinate system {intrinsic!r} by any coordinateTransformations."
            )

    def _validate_resolution_order(self, intrinsic: CoordinateSystem) -> None:
        # The `path`s MUST be ordered from largest (i.e. highest resolution) to
        # smallest.
        # (Really a storage-level rule; like v05, we also check it on the scales of
        # the spatial axes, when every level has one. Scales may be negative (flips),
        # so we compare magnitudes.)
        scales = [ds.scale_transform for ds in self.datasets]
        if any(s is None for s in scales):
            return
        space = [i for i, ax in enumerate(intrinsic.axes) if ax.type == "space"]
        spatial = [tuple(abs(s.scale[i]) for i in space) for s in scales]
        if spatial != sorted(spatial):
            raise ValueError(
                "The datasets are not ordered from highest to lowest resolution. "
                f"Found spatial scales: {spatial}"
            )

    @property
    def intrinsic_coordinate_system(self) -> CoordinateSystem:
        """The coordinate system that all datasets map into."""
        name = self.datasets[0].output_name
        return next(cs for cs in self.coordinateSystems if cs.name == name)

    @property
    def axes(self) -> list[Axis]:
        """Axes of the intrinsic coordinate system (one per array dimension)."""
        return list(self.intrinsic_coordinate_system.axes)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the image."""
        return self.intrinsic_coordinate_system.ndim


def _is_label_transform(t: CoordinateTransformation) -> bool:
    if isinstance(t, SequenceTransformation):
        # "a sequence of a scale and a translation transformation" (no order is given
        # here, unlike for dataset transformations, so we accept either order)
        return sorted(step.type for step in t.transformations) == [
            "scale",
            "translation",
        ]
    return t.type in _LABEL_TRANSFORM_TYPES


# ------------------------------------------------------------------------------
# Image model
# ------------------------------------------------------------------------------


class Image(_BaseModel):
    """Top-level OME-NGFF image metadata (`ome` in an image group's `zarr.json`).

    Contains one or more multiscale pyramids, plus optional OMERO rendering hints.
    """

    version: VersionStr = Field(default="0.6", description="OME-NGFF version")
    multiscales: Annotated[UniqueList[Multiscale], MinLen(1)] = Field(
        description="One or more multiscale image pyramids in this group"
    )
    omero: Omero | None = Field(
        default=None,
        description="Optional OMERO rendering metadata for visualization",
    )
