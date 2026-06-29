import warnings
from collections.abc import Sequence
from typing import Annotated, TypeAlias

from annotated_types import Len, MinLen
from pydantic import AfterValidator, Field, WrapValidator, model_validator
from typing_extensions import Self

from yaozarrs._base import _BaseModel
from yaozarrs._dim_spec import DimSpec
from yaozarrs._omero import Omero
from yaozarrs._types import UniqueList
from yaozarrs._util import SuggestDatasetPath
from yaozarrs._validation_warning import ValidationWarning

from ._coordinate_systems import CoordinateSystem, CoordinateSystems
from ._transforms import (
    IdentityTransformation,
    ScaleTransformation,
    SequenceTransformation,
    Transformation,
    TranslationTransformation,
)
from ._version import OMEV06

__all__ = [  # noqa: RUF022  (don't resort, this is used for docs ordering)
    "Image",
    "Multiscale",
    "Dataset",
    "CoordinateSystem",
]

# The conventional name of the "intrinsic" coordinate system that dataset
# transformations output to (see the worked example in the v0.6 spec).
DEFAULT_COORDINATE_SYSTEM = "intrinsic"

# ------------------------------------------------------------------------------
# Dataset model
# ------------------------------------------------------------------------------


def _validate_dataset_transform(
    transforms: list[Transformation],
) -> list[Transformation]:
    """Validate the single transform on a dataset.

    Per `image.schema`, a dataset's `coordinateTransformations` array contains
    exactly one transform, which MUST be one of:

    - a single `scale`
    - a single `identity`
    - a `sequence` of exactly one `scale` followed by one `translation`

    and its `input` MUST provide `path` while its `output` MUST provide `name`.
    """
    (t,) = transforms  # Len(1, 1) guarantees exactly one

    # this validation of a SequenceTransformation inside a DatasetTransformList, while
    # verbose and annoying, is caused because the image.schema *specifically* redefines
    # the union of valid Transformations inside of the coordinateTransformations list
    # inside of a Dataset such that SequenceTransformations may only contain
    # exactly one ScaleTransformation followed by exactly one TranslationTransformation.
    if isinstance(t, SequenceTransformation):
        inner = t.transformations
        scales = [x for x in inner if isinstance(x, ScaleTransformation)]
        translations = [x for x in inner if isinstance(x, TranslationTransformation)]
        if len(inner) != 2 or len(scales) != 1 or len(translations) != 1:
            raise ValueError(
                "A dataset 'sequence' transform must contain exactly one scale "
                "and one translation."
            )
        # spec description: "a single scale followed by a single translation"
        if not isinstance(inner[0], ScaleTransformation):
            raise ValueError(
                "In a dataset 'sequence' transform, the scale must come before "
                "the translation."
            )
    elif not isinstance(t, (ScaleTransformation, IdentityTransformation)):
        raise ValueError(
            "A dataset coordinateTransformation must be a 'scale', 'identity', or "
            f"a 'sequence' of scale+translation, not {t.type!r}."
        )

    # input MUST provide `path`; output MUST provide `name`
    if t.input is None or t.input.path is None:
        raise ValueError("A dataset transform's 'input' must provide a 'path'.")
    if t.output is None or t.output.name is None:
        raise ValueError("A dataset transform's 'output' must provide a 'name'.")

    # SHOULD-level: input SHOULD omit `name`, output SHOULD omit `path` (spec).
    if t.input.name is not None:
        warnings.warn(
            "A dataset transform's 'input' SHOULD omit 'name' (only 'path' is used).",
            ValidationWarning,
            stacklevel=2,
        )
    if t.output.path is not None:
        warnings.warn(
            "A dataset transform's 'output' SHOULD omit 'path' (only 'name' is used).",
            ValidationWarning,
            stacklevel=2,
        )
    return transforms


DatasetTransformList: TypeAlias = Annotated[
    UniqueList[Transformation],
    Len(min_length=1, max_length=1),
    AfterValidator(_validate_dataset_transform),
]


class Dataset(_BaseModel):
    """A single resolution level in a multiscale image pyramid.

    Each dataset points to a Zarr array and defines how its indices map into a
    named coordinate system. Together, multiple datasets form a resolution
    pyramid where each level represents the same physical region at different
    sampling rates.

    !!! note "Change from v0.5"
        The single `coordinateTransformations` entry now carries `input`
        (`{"path": ...}`, the dataset's own array) and `output`
        (`{"name": ...}`, the coordinate system it maps into).
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

    coordinateTransformations: DatasetTransformList = Field(
        description=(
            "Exactly one transformation mapping this dataset's array (input.path) "
            "into a coordinate system (output.name). Must be a scale, an identity, "
            "or a sequence of scale+translation."
        )
    )

    @property
    def transform(self) -> Transformation:
        """The single coordinate transformation for this dataset."""
        return self.coordinateTransformations[0]

    @property
    def scale_transform(self) -> ScaleTransformation | None:
        """Return the scale transformation, if this dataset has one.

        For an `identity` dataset transform this is `None`.
        """
        t = self.transform
        if isinstance(t, ScaleTransformation):
            return t
        if isinstance(t, SequenceTransformation):
            return next(
                (x for x in t.transformations if isinstance(x, ScaleTransformation)),
                None,
            )
        return None

    @property
    def translation_transform(self) -> TranslationTransformation | None:
        """Return the translation transformation, if present (sequence only)."""
        t = self.transform
        if isinstance(t, SequenceTransformation):
            return next(
                (
                    x
                    for x in t.transformations
                    if isinstance(x, TranslationTransformation)
                ),
                None,
            )
        return None

    @property
    def output_name(self) -> str | None:
        """Name of the coordinate system this dataset maps into."""
        tf = self.transform
        return None if tf.output is None else tf.output.name

    @property
    def ndim(self) -> int | None:
        """Dimensionality implied by this dataset's scale, or None for identity."""
        st = self.scale_transform
        return None if st is None else st.ndim


def _validate_datasets_list(datasets: list[Dataset]) -> list[Dataset]:
    """Validate a list of Dataset for `Multiscale.datasets`."""
    # Each dataset MUST have the same dimensionality, and MUST NOT have more than
    # 5 dimensions. `identity` datasets carry no explicit dimensionality (ndim is
    # None) and are skipped here.
    ndims = {dt.path: dt.ndim for dt in datasets if dt.ndim is not None}
    if len(set(ndims.values())) > 1:
        raise ValueError(
            "All datasets must have the same number of dimensions. "
            f"Found differing dimensions: {ndims}"
        )
    if any(n > 5 for n in ndims.values()):
        raise ValueError("Datasets must not have more than 5 dimensions.")

    # The datasets MUST be ordered from highest to lowest resolution: each level
    # must be coarser-or-equal to the previous one in *every* dimension (its scale
    # factors element-wise >= the previous level's). The spec
    # restricts dataset transforms to scale/identity/sequence(scale, translation),
    # so the scale is always directly available (never hidden in e.g. an affine).
    # An `identity` level counts as a scale of all ones.
    if ndims:  # all-identity (or empty) lists carry no scale info -> nothing to order
        ndim = next(iter(ndims.values()))
        scales = [
            (dt.scale_transform.scale if dt.scale_transform else [1.0] * ndim)
            for dt in datasets
        ]
        for i in range(1, len(scales)):
            prev, cur = scales[i - 1], scales[i]
            if any(c < p for p, c in zip(prev, cur)):
                raise ValueError(
                    "The datasets are not ordered from highest to lowest "
                    f"resolution: level {i} (scale {cur}) is finer than level "
                    f"{i - 1} (scale {prev}) in at least one dimension."
                )
    return datasets


DatasetsList: TypeAlias = Annotated[
    list[Dataset],
    MinLen(1),
    # hack to get around ordering of multiple after validators
    WrapValidator(lambda v, h: _validate_datasets_list(h(v))),
]

# ------------------------------------------------------------------------------
# Multiscale model
# ------------------------------------------------------------------------------


class Multiscale(_BaseModel):
    """Multi-resolution image pyramid (<=5D) with coordinate metadata.

    Defines an image at one or more resolution levels, along with the coordinate
    system(s) that relate array indices to physical space.

    !!! note "Major change from v0.5"
        The `axes` field is **replaced** by `coordinateSystems`: a list of named
        [`CoordinateSystem`][yaozarrs.v06.CoordinateSystem] objects (each holding
        the `axes`). Datasets map into one of these by name (their `output`).

    !!! note "Resolution Ordering"
        Datasets must be ordered from highest to lowest resolution
        (i.e., finest to coarsest sampling).
    """

    name: str | None = Field(
        default=None,
        description="Optional identifier for this multiscale image",
    )
    datasets: DatasetsList = Field(
        description=(
            "Resolution pyramid levels, ordered from highest to lowest resolution"
        )
    )
    coordinateSystems: Annotated[CoordinateSystems, MinLen(1)] = Field(
        description=(
            "Named coordinate systems for this image. Datasets `output` into one "
            "of these (conventionally named 'intrinsic')."
        )
    )
    coordinateTransformations: Annotated[list[Transformation], MinLen(1)] | None = (
        Field(
            default=None,
            description=(
                "Additional transformations between coordinate systems, applied to "
                "all resolution levels."
            ),
        )
    )

    # NOTE: "type" and "metadata" are mentioned in the spec (SHOULD), but are not
    # in the (non-strict) image.schema.
    type: str | None = Field(
        default=None,
        description=(
            "Type of downscaling method used to generate the multiscale image pyramid."
        ),
    )
    metadata: dict | None = Field(
        default=None,
        description="Unstructured key-value pair with additional "
        "information about the downscaling method.",
    )

    @model_validator(mode="after")
    def _post_validate(self) -> Self:
        cs_names = {cs.name for cs in self.coordinateSystems}

        # All datasets MUST map to the SAME output coordinate system (the spec:
        # the dataset transform's `output.name` "MUST be the same value for every
        # resolution level in a single multiscales").
        out_names = {ds.output_name for ds in self.datasets if ds.output_name}
        if len(out_names) > 1:
            raise ValueError(
                "All datasets in a multiscale must output to the same coordinate "
                f"system, but found multiple: {sorted(out_names)}."
            )

        # every dataset must output to a declared coordinate system, and (when it
        # has an explicit scale) match that system's dimensionality.
        for _id, ds in enumerate(self.datasets):
            out = ds.output_name
            if out is not None and out not in cs_names:
                raise ValueError(
                    f"at datasets.[{_id}]:\n"
                    f"  output coordinate system {out!r} is not declared in "
                    f"coordinateSystems {sorted(cs_names)}."
                )
            if out is not None and ds.ndim is not None:
                target = next(cs for cs in self.coordinateSystems if cs.name == out)
                if ds.ndim != len(target.axes):
                    raise ValueError(
                        f"at datasets.[{_id}]:\n"
                        f"  The dataset transform dimensionality ({ds.ndim}) does "
                        f"not match the number of axes ({len(target.axes)}) of "
                        f"coordinate system {out!r}."
                    )

        # `multiscales > coordinateTransformations` rules (spec): each transform's
        # `input` MUST be the intrinsic coordinate system (referenced by `name`),
        # and `output` MUST reference either a coordinate system declared here (by
        # `name`) or one in a child `labels` group (by `name` + `path`). In the
        # labels case the transform MUST be an identity/scale/translation.
        if self.coordinateTransformations:
            intrinsic_name = next(iter(out_names), None)
            for _id, t in enumerate(self.coordinateTransformations):
                loc = f"coordinateTransformations.[{_id}]"
                if t.input is None or t.input.name is None:
                    raise ValueError(
                        f"at {loc}:\n  'input' must reference the intrinsic "
                        "coordinate system by 'name'."
                    )
                if intrinsic_name is not None and t.input.name != intrinsic_name:
                    raise ValueError(
                        f"at {loc}:\n  'input' must be the intrinsic coordinate "
                        f"system {intrinsic_name!r}, not {t.input.name!r}."
                    )
                if t.input.path is not None:
                    warnings.warn(
                        f"at {loc}: 'input.path' SHOULD be omitted; the input "
                        "refers to the intrinsic coordinate system in the same "
                        "document.",
                        ValidationWarning,
                        stacklevel=2,
                    )
                if t.output is None or t.output.name is None:
                    raise ValueError(
                        f"at {loc}:\n  'output' must reference a coordinate system "
                        "by 'name'."
                    )
                if t.output.path is None:
                    if t.output.name not in cs_names:
                        raise ValueError(
                            f"at {loc}:\n  'output' coordinate system "
                            f"{t.output.name!r} is not declared in coordinateSystems "
                            f"{sorted(cs_names)}."
                        )
                elif not isinstance(
                    t,
                    (
                        IdentityTransformation,
                        ScaleTransformation,
                        TranslationTransformation,
                    ),
                ):
                    raise ValueError(
                        f"at {loc}:\n  a transform whose 'output' links to a child "
                        "labels group (via 'path') must be an identity, scale, or "
                        f"translation, not {t.type!r}."
                    )

        # NOTE: scale ordering of datasets is validated in `_validate_datasets_list`
        return self

    @property
    def intrinsic_coordinate_system(self) -> CoordinateSystem | None:
        """The coordinate system datasets map into (their shared `output.name`).

        Falls back to the sole coordinate system if datasets don't declare an
        output name, or `None` if it cannot be determined.
        """
        out_names = {ds.output_name for ds in self.datasets if ds.output_name}
        if len(out_names) == 1:
            target = next(iter(out_names))
            return next(
                (cs for cs in self.coordinateSystems if cs.name == target), None
            )
        if len(self.coordinateSystems) == 1:
            return self.coordinateSystems[0]
        return None  # pragma: no cover

    @property
    def axes(self) -> list:
        """The axes of the intrinsic coordinate system (convenience accessor).

        !!! note
            In v0.5 `axes` was a real field on the multiscale. In v0.6 it lives
            inside a coordinate system; this read-only property returns the
            intrinsic system's axes for convenience and v0.5 parity.
        """
        cs = self.intrinsic_coordinate_system
        return [] if cs is None else cs.axes

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @classmethod
    def from_dims(
        cls,
        dims: Sequence[DimSpec],
        name: str | None = None,
        n_levels: int = 1,
        coordinate_system: str = DEFAULT_COORDINATE_SYSTEM,
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
        coordinate_system : str, optional
            Name of the (intrinsic) coordinate system the datasets map into.
            Default is "intrinsic".

        Returns
        -------
        Multiscale
            A fully configured Multiscale model.

        Examples
        --------
        >>> from yaozarrs import DimSpec, v06
        >>> dims = [
        ...     DimSpec(name="t", size=512, unit="second"),
        ...     DimSpec(
        ...         name="z", size=50, scale=2.0, unit="micrometer", scale_factor=1.0
        ...     ),
        ...     DimSpec(name="y", size=512, scale=0.5, unit="micrometer"),
        ...     DimSpec(name="x", size=512, scale=0.5, unit="micrometer"),
        ... ]
        >>> v06.Multiscale.from_dims(dims, name="my_multiscale", n_levels=3)
        """
        from yaozarrs._dim_spec import _coordinate_systems_datasets

        kwargs = _coordinate_systems_datasets(dims, n_levels, coordinate_system)
        return cls(name=name, **kwargs)  # type: ignore


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
        For the optional `labels` group, see [LabelsGroup][yaozarrs.v06.LabelsGroup].
    """

    version: OMEV06 = Field(
        default="0.6.dev4",
        description="OME-NGFF specification version",
    )
    multiscales: Annotated[UniqueList[Multiscale], MinLen(1)] = Field(
        description="One or more multiscale image pyramids in this group"
    )
    omero: Omero | None = Field(
        default=None,
        description="Optional OMERO rendering metadata for visualization",
    )
