"""Storage validation for OME-ZARR v0.5 hierarchies.

This module provides functions to validate that OME-ZARR v0.5 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import chain, product
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

import numpy as np
from typing_extensions import NotRequired

from yaozarrs._validate import from_uri, validate_ome_object
from yaozarrs._zarr import ZarrArray, ZarrGroup, open_group
from yaozarrs.v05._bf2raw import Bf2Raw
from yaozarrs.v05._image import Image, Multiscale
from yaozarrs.v05._label import LabelImage, LabelsGroup
from yaozarrs.v05._ome import OME
from yaozarrs.v05._plate import Plate
from yaozarrs.v05._well import Well
from yaozarrs.v05._zarr_json import OMEAttributes, OMEZarrGroupJSON

if TYPE_CHECKING:
    from pathlib import Path

# ----------------------------------------------------------
# ERROR HANDLING
# ----------------------------------------------------------


class ErrorDetails(TypedDict):
    type: str
    """
    The type of error that occurred, this is an identifier designed for
    programmatic use that will change rarely or never.

    `type` is unique for each error message, and can hence be used as an identifier to
    build custom error messages.
    """
    loc: tuple[int | str, ...]
    """Tuple of strings and ints identifying where in the schema the error occurred."""
    msg: str
    """A human readable error message."""
    input: Any
    """The input data at this `loc` that caused the error."""
    ctx: NotRequired[dict[str, Any]]
    """
    Values which are required to render the error message, and could hence be useful in
    rendering custom error messages.
    Also useful for passing custom error data forward.
    """
    url: NotRequired[str]
    """A URL giving information about the error."""


def _create_error(
    error_type: str,
    loc: tuple[int | str, ...],
    msg: str,
    input_val: Any = None,
    ctx: dict[str, Any] | None = None,
    url: str | None = None,
) -> ErrorDetails:
    """Create a standardized error detail dictionary."""
    error: ErrorDetails = {
        "type": error_type,
        "loc": loc,
        "msg": msg,
        "input": input_val,
    }
    if ctx is not None:
        error["ctx"] = ctx
    if url is not None:
        error["url"] = url
    return error


@dataclass(slots=True)
class ValidationResult:
    """Result of a validation operation containing any errors found."""

    errors: list[ErrorDetails] = field(default_factory=list)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge this result with another, combining errors."""
        return ValidationResult(errors=self.errors + other.errors)

    def add_error(
        self,
        error_type: str,
        loc: tuple[int | str, ...],
        msg: str,
        input_val: Any = None,
        ctx: dict[str, Any] | None = None,
        url: str | None = None,
    ) -> ValidationResult:
        """Add an error to this result and return self for chaining."""
        self.errors.append(_create_error(error_type, loc, msg, input_val, ctx, url))
        return self

    @property
    def is_valid(self) -> bool:
        """Return True if no errors were found."""
        return len(self.errors) == 0


class StorageValidationError(ValueError):
    """`StorageValidationError` is raised when validation of zarr storage fails.

    It contains a list of errors which detail why validation failed.
    """

    def __init__(self, errors: list[ErrorDetails]) -> None:
        self._errors = errors
        super().__init__(self._error_message())

    def _error_message(self) -> str:
        """Generate a readable error message from all validation errors."""
        if not self._errors:  # pragma: no cover
            return "No validation errors"

        lines = [f"{len(self._errors)} validation error(s) for storage structure:"]
        for i, error in enumerate(self._errors, 1):
            lines.append(
                f"{i:>2}. {error['msg']} (type={error['type']}, loc={error['loc']})"
            )
        return "\n".join(lines)

    @property
    def title(self) -> str:
        """The title of the error, as used in the heading of `str(validation_error)`."""
        return "StorageValidationError"

    def errors(
        self,
        *,
        include_url: bool = True,
        include_context: bool = True,
        include_input: bool = True,
    ) -> list[ErrorDetails]:
        """
        Details about each error in the validation error.

        Parameters
        ----------
        include_url: bool
            Whether to include a URL to documentation on the error each error.
        include_context: bool
            Whether to include the context of each error.
        include_input: bool
            Whether to include the input value of each error.

        Returns
        -------
            A list of `ErrorDetails` for each error in the validation error.
        """
        filtered_errors: list[ErrorDetails] = []
        for error in self._errors:
            filtered_error = {
                "type": error["type"],
                "loc": error["loc"],
                "msg": error["msg"],
            }
            if include_input and "input" in error:
                filtered_error["input"] = error["input"]
            if include_context and "ctx" in error:
                filtered_error["ctx"] = error["ctx"]
            if include_url and "url" in error:
                filtered_error["url"] = error["url"]
            filtered_errors.append(filtered_error)
        return filtered_errors


# ----------------------------------------------------------
# MAIN VALIDATION FUNCTIONS
# ----------------------------------------------------------


def validate_zarr_store(
    obj: OMEZarrGroupJSON | ZarrGroup | str | Path | Any,
) -> None:
    """Validate an OME-Zarr v0.5 storage structure.

    Parameters
    ----------
    obj : OMEZarrGroupJSON | ZarrGroup | str | Path | Any
        The zarr store to validate. Can be a URI string, a Path, a parsed
        OMEZarrGroupJSON object, a ZarrGroup instance, or a zarr.Group object
        (for backwards compatibility).

    Raises
    ------
    StorageValidationError
        If the storage structure is invalid.
    """
    attrs_model: OMEAttributes | None = None
    if isinstance(obj, ZarrGroup):
        zarr_group = obj
    elif isinstance(obj, OMEZarrGroupJSON):
        if not obj.uri:
            raise ValueError("Cannot validate OMEZarrGroupJSON without a uri")
        attrs_model = obj.attributes
        zarr_group = open_group(obj.uri)
    else:
        zarr_group = open_group(obj)

    # Validate the storage structure using the visitor pattern
    result = StorageValidatorV05.validate_group(zarr_group, attrs_model)

    # Raise error if any validation issues found
    if not result.is_valid:
        raise StorageValidationError(result.errors)


# ----------------------------------------------------------
# VALIDATORS
# ----------------------------------------------------------

Loc: TypeAlias = tuple[int | str, ...]


@dataclass
class LabelsCheckResult:
    """Result of checking for a labels group."""

    result: ValidationResult
    labels_info: tuple[ZarrGroup, LabelsGroup] | None = None


class StorageValidatorV05:
    """Concrete implementation of storage validator. for OME-ZARR v0.5 spec."""

    __slots__ = ()

    @classmethod
    def validate_group(
        cls, zarr_group: ZarrGroup, attrs_model: OMEAttributes | None = None
    ) -> ValidationResult:
        """Entry point that dispatches to appropriate visitor method.

        Parameters
        ----------
        zarr_group : ZarrGroup
            The zarr group to validate.
        attrs_model : OMEAttributes
            The validated OME attributes model.

        Returns
        -------
        ValidationResult
            The validation result containing any errors found.
        """
        if attrs_model is None:
            # extract the model from the zarr attributes
            attrs = zarr_group.attrs.asdict()
            attrs_model = validate_ome_object(attrs, OMEAttributes)

        validator = cls()
        ome_metadata = attrs_model.ome
        loc_prefix = ("ome",)

        # Dispatch to appropriate visitor method based on metadata type
        if isinstance(ome_metadata, LabelImage):
            return validator.visit_label_image(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, Image):
            return validator.visit_image(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, LabelsGroup):
            return validator.visit_labels_group(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, Plate):
            return validator.visit_plate(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, Well):
            return validator.visit_well(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, Multiscale):
            return validator.visit_multiscale(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, Bf2Raw):
            return validator.visit_bioformats2raw(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, OME):
            return validator.visit_series(zarr_group, ome_metadata, loc_prefix)
        else:
            raise NotImplementedError(
                f"Unknown OME metadata type: {type(ome_metadata).__name__}"
            )

    def visit_label_image(
        self, zarr_group: ZarrGroup, label_image_model: LabelImage, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a LabelImage group."""
        result = ValidationResult()

        # The value of the source key MUST be a JSON object containing information
        # about the original image from which the label image derives. This object
        # MAY include a key image, whose value MUST be a string specifying the
        # relative path to a Zarr image group.
        src = label_image_model.image_label.source
        if src is not None and (src_img := src.image) is not None:
            result = result.merge(
                self._validate_labels_image_source(zarr_group, src_img, loc_prefix)
            )

        # For label images, validate integer data types
        result = result.merge(
            self._validate_label_data_types(label_image_model, zarr_group, loc_prefix)
        )

        return result

    def visit_image(
        self, zarr_group: ZarrGroup, image_model: Image, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate an image group with multiscales metadata."""
        result = ValidationResult()

        # Collect all children we'll need to check and prefetch them in one batch
        children_to_prefetch = []
        for multiscale in image_model.multiscales:
            children_to_prefetch.extend(ds.path for ds in multiscale.datasets)
        # Also check for labels group
        children_to_prefetch.append("labels")
        zarr_group.prefetch_children(children_to_prefetch)

        # Validate each multiscale
        for ms_idx, multiscale in enumerate(image_model.multiscales):
            ms_loc = (*loc_prefix, "multiscales", ms_idx)
            # Note: datasets already prefetched above, no need to prefetch again
            result = result.merge(
                self._visit_multiscale_no_prefetch(zarr_group, multiscale, ms_loc)
            )

        # Check whether this image has a labels group, and validate if so
        lbls_check = self._check_for_labels_group(zarr_group, loc_prefix)
        result = result.merge(lbls_check.result)

        if lbls_check.labels_info is not None:
            labels_group, labels_model = lbls_check.labels_info
            result = result.merge(
                self.visit_labels_group(
                    labels_group,
                    labels_model,
                    (*loc_prefix, "labels"),
                    image_model,
                )
            )

        return result

    def visit_labels_group(
        self,
        labels_group: ZarrGroup,
        labels_model: LabelsGroup,
        loc_prefix: Loc,
        parent_image_model: Image | None = None,
    ) -> ValidationResult:
        """Validate a labels group and its referenced label images."""
        result = ValidationResult()

        # Prefetch all label metadata in a single batch request for performance
        labels_group.prefetch_children(labels_model.labels)

        # Validate each label path exists and is valid LabelImage
        for label_idx, label_path in enumerate(labels_model.labels):
            label_loc = (*loc_prefix, "labels", label_idx)

            if label_path not in labels_group:
                result.add_error(
                    "label_path_not_found",
                    label_loc,
                    f"Label path '{label_path}' not found in labels group",
                    label_path,
                )
                continue

            label_group = labels_group[label_path]
            if not isinstance(label_group, ZarrGroup):
                result.add_error(
                    "label_path_not_group",
                    label_loc,
                    f"Label path '{label_path}' is not a zarr group",
                    label_path,
                )
                continue

            # Validate as LabelImage
            label_attrs = label_group.attrs.asdict()
            ome_attrs = validate_ome_object(label_attrs, OMEAttributes)

            if not isinstance((label_image_model := ome_attrs.ome), Image):
                result.add_error(
                    "invalid_label_image",
                    label_loc,
                    f"Label path '{label_path}' does not contain "
                    "valid Image ('multiscales') metadata",
                    {"path": label_path, "type": type(ome_attrs.ome).__name__},
                )
                continue

            # Within the multiscales object, the JSON array associated with the
            # datasets key MUST have the same number of entries (scale levels) as
            # the original unlabeled image.
            if parent_image_model is not None:
                n_lbl_ms = len(label_image_model.multiscales)
                n_img_ms = len(parent_image_model.multiscales)
                if n_lbl_ms != n_img_ms:
                    result.add_error(
                        "label_multiscale_count_mismatch",
                        label_loc,
                        f"Label image '{label_path}' has {n_lbl_ms} "
                        f"multiscales, but parent image has {n_img_ms}",
                        {
                            "label_path": label_path,
                            "label_multiscales": n_lbl_ms,
                            "parent_multiscales": n_img_ms,
                        },
                    )

                for ms_idx, (lbl_ms, img_ms) in enumerate(
                    zip(label_image_model.multiscales, parent_image_model.multiscales)
                ):
                    n_lbl_ds = len(lbl_ms.datasets)
                    n_img_ds = len(img_ms.datasets)
                    if n_lbl_ds < n_img_ds:
                        result.add_error(
                            "label_dataset_count_mismatch",
                            (*label_loc, "multiscales", ms_idx),
                            f"Label image '{label_path}' multiscale index {ms_idx} "
                            f"has {n_lbl_ds} datasets, but parent image multiscale "
                            f"index {ms_idx} has {n_img_ds}",
                            {
                                "label_path": label_path,
                                "multiscale_index": ms_idx,
                                "label_datasets": n_lbl_ds,
                                "parent_datasets": n_img_ds,
                            },
                        )

            if not isinstance(label_image_model, LabelImage):
                # TODO: should it just be a warning?
                result.add_error(
                    "invalid_label_image",
                    label_loc,
                    f"Label path '{label_path}' contains Image metadata, "
                    "but is not a LabelImage (missing 'image-label' metadata?)",
                    {"path": label_path, "type": type(label_image_model).__name__},
                )
                continue

            # Recursively validate the label image
            result = result.merge(
                self.visit_image(label_group, label_image_model, label_loc)
            )

        return result

    def _visit_multiscale_no_prefetch(
        self, zarr_group: ZarrGroup, multiscale: Multiscale, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate multiscale without prefetching (assumes already prefetched)."""
        result = ValidationResult()

        for ds_idx, dataset in enumerate(multiscale.datasets):
            ds_loc = (*loc_prefix, "datasets", ds_idx, "path")

            # Check if path exists as array
            if (arr := zarr_group.get(dataset.path)) is None:
                result.add_error(
                    "dataset_path_not_found",
                    ds_loc,
                    f"Dataset path '{dataset.path}' not found in zarr group",
                    dataset.path,
                )
                continue

            if not isinstance(arr, ZarrArray):
                result.add_error(
                    "dataset_not_array",
                    ds_loc,
                    f"Dataset path '{dataset.path}' exists but is not a zarr array",
                    dataset.path,
                )
                continue

            # Check array dimensionality matches axes
            expected_ndim = len(multiscale.axes)
            if arr.ndim != expected_ndim:
                result.add_error(
                    "dataset_dimension_mismatch",
                    ds_loc,
                    f"Dataset '{dataset.path}' has {arr.ndim} dimensions "
                    f"but axes specify {expected_ndim}",
                    {
                        "actual_ndim": arr.ndim,
                        "expected_ndim": expected_ndim,
                        "path": dataset.path,
                    },
                )

            # Check dimension_names attribute matches axes
            if dim_names := list(arr.attrs.asdict().get("dimension_names", [])):
                expected_names = [ax.name for ax in multiscale.axes]
                if dim_names != expected_names:
                    result.add_error(
                        "dimension_names_mismatch",
                        (*ds_loc, "dimension_names"),
                        f"Array dimension_names {dim_names} don't match "
                        f"axes names {expected_names}",
                        {"actual": dim_names, "expected": expected_names},
                    )

        return result

    def visit_multiscale(
        self, zarr_group: ZarrGroup, multiscale: Multiscale, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate that all dataset paths exist with correct dimensionality."""
        # Prefetch all dataset metadata in a single batch request for performance
        dataset_paths = [ds.path for ds in multiscale.datasets]
        zarr_group.prefetch_children(dataset_paths)
        return self._visit_multiscale_no_prefetch(zarr_group, multiscale, loc_prefix)

    def visit_plate(
        self, zarr_group: ZarrGroup, plate_model: Plate, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a plate group and its wells."""
        result = ValidationResult()

        well_paths = [well.path for well in plate_model.plate.wells]

        # Prefetch all metadata in one go to minimize network round trips
        self._prefetch_plate_hierarchy(zarr_group, well_paths)

        # Validate each well path
        for well_idx, well in enumerate(plate_model.plate.wells):
            well_loc = (*loc_prefix, "plate", "wells", well_idx)

            if (well_group := zarr_group.get(well.path)) is None:
                result.add_error(
                    "well_path_not_found",
                    (*well_loc, "path"),
                    f"Well path '{well.path}' not found in plate group",
                    well.path,
                )
                continue

            if not isinstance(well_group, ZarrGroup):
                result.add_error(
                    "well_path_not_group",
                    (*well_loc, "path"),
                    f"Well path '{well.path}' is not a zarr group",
                    well.path,
                )
                continue

            # Validate well metadata
            well_model = well_group.ome_model().ome
            if not isinstance(well_model, Well):
                result.add_error(
                    "invalid_well",
                    well_loc,
                    f"Well path '{well.path}' does not contain valid Well metadata",
                    {"path": well.path, "type": type(well_model).__name__},
                )
                continue

            result = result.merge(self.visit_well(well_group, well_model, well_loc))

        return result

    def visit_well(
        self, zarr_group: ZarrGroup, well_model: Well, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a well group and its field images."""
        result = ValidationResult()

        # Performance optimization: prefetch all field image metadata in one batch
        # Note: For plates, this is often already prefetched by visit_plate's
        # deep prefetch strategy, but we do it here too for standalone wells
        field_paths = [field.path for field in well_model.well.images]
        zarr_group.prefetch_children(field_paths)

        # Validate each field image path
        for field_idx, field_image in enumerate(well_model.well.images):
            field_loc = (*loc_prefix, "well", "images", field_idx)

            if (field_group := zarr_group.get(field_image.path)) is None:
                result.add_error(
                    "field_path_not_found",
                    (*field_loc, "path"),
                    f"Field path '{field_image.path}' not found in well group",
                    field_image.path,
                )
                continue

            if not isinstance(field_group, ZarrGroup):
                result.add_error(
                    "field_path_not_group",
                    (*field_loc, "path"),
                    f"Field path '{field_image.path}' is not a zarr group",
                    field_image.path,
                )
                continue

            # Validate field as image group
            field_attrs = field_group.attrs.asdict()
            field_attrs_model = validate_ome_object(field_attrs, OMEAttributes)
            if isinstance(field_attrs_model.ome, Image):
                result = result.merge(
                    self.visit_image(field_group, field_attrs_model.ome, field_loc)
                )

        return result

    def visit_bioformats2raw(
        self, zarr_group: ZarrGroup, bf2raw_model: Bf2Raw, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a bioformats2raw layout.

        According to spec:
        1. Check for OME subgroup with optional "series" metadata
        2. If series exists, validate those paths
        3. Otherwise, validate consecutively numbered directories (0/, 1/, 2/, ...)
        """
        result = ValidationResult()

        # First, check if there's an OME subgroup
        zarr_group.prefetch_children(["OME"])

        # Check for OME subgroup with series metadata
        ome_group = zarr_group.get("OME")
        if ome_group is not None and isinstance(ome_group, ZarrGroup):
            ome_attrs = ome_group.attrs.asdict()
            try:
                ome_attrs_model = validate_ome_object(ome_attrs, OMEAttributes)

                # If OME group has series metadata, use that to find images
                if isinstance(ome_attrs_model.ome, OME):
                    # Validate using the series paths
                    result = result.merge(
                        self.visit_series(
                            zarr_group, ome_attrs_model.ome, (*loc_prefix, "OME")
                        )
                    )
                    return result
            except Exception:
                # OME group exists but doesn't have valid OME metadata
                # Fall through to numbered directory validation
                pass

        # No OME group with series, so validate numbered directories
        # Discover consecutively numbered directories (0, 1, 2, etc.)
        # FIXME: ... do better for searching
        numbered_paths = []
        for i in range(1000):  # reasonable upper limit
            if str(i) not in zarr_group:
                break
            numbered_paths.append(str(i))

        if not numbered_paths:
            result.add_error(
                "bf2raw_no_images",
                loc_prefix,
                "Bioformats2raw group contains no numbered image directories",
                None,
            )
            return result

        # Prefetch all numbered path metadata
        zarr_group.prefetch_children(numbered_paths)

        # Validate each numbered directory as image group
        for path in numbered_paths:
            image_loc = (*loc_prefix, path)

            image_group = zarr_group.get(path)
            if image_group is None:
                result.add_error(
                    "bf2raw_path_not_found",
                    image_loc,
                    f"Bioformats2raw path '{path}' not found",
                    path,
                )
                continue

            if not isinstance(image_group, ZarrGroup):
                result.add_error(
                    "bf2raw_path_not_group",
                    image_loc,
                    f"Bioformats2raw path '{path}' is not a zarr group",
                    path,
                )
                continue

            # Validate as image group
            image_attrs = image_group.attrs.asdict()
            image_attrs_model = validate_ome_object(image_attrs, OMEAttributes)

            if isinstance(image_attrs_model.ome, Image):
                result = result.merge(
                    self.visit_image(image_group, image_attrs_model.ome, image_loc)
                )
            else:
                result.add_error(
                    "bf2raw_invalid_image",
                    image_loc,
                    f"Bioformats2raw path '{path}' does not contain "
                    "valid Image metadata",
                    {"path": path, "type": type(image_attrs_model.ome).__name__},
                )

        return result

    def visit_series(
        self, zarr_group: ZarrGroup, ome_model: OME, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate an OME group with series metadata.

        The series attribute is a list of paths to image groups. Each path
        should point to a valid image group with multiscales metadata.
        """
        result = ValidationResult()

        # Prefetch all series path metadata
        zarr_group.prefetch_children(ome_model.series)

        # Validate each series path
        for series_idx, series_path in enumerate(ome_model.series):
            series_loc = (*loc_prefix, "series", series_idx)

            series_group = zarr_group.get(series_path)
            if series_group is None:
                result.add_error(
                    "series_path_not_found",
                    series_loc,
                    f"Series path '{series_path}' not found in series group",
                    series_path,
                )
                continue

            if not isinstance(series_group, ZarrGroup):
                result.add_error(
                    "series_path_not_group",
                    series_loc,
                    f"Series path '{series_path}' is not a zarr group",
                    series_path,
                )
                continue

            # Validate series as image group
            series_attrs = series_group.attrs.asdict()
            series_attrs_model = validate_ome_object(series_attrs, OMEAttributes)

            if isinstance(series_attrs_model.ome, Image):
                result = result.merge(
                    self.visit_image(series_group, series_attrs_model.ome, series_loc)
                )
            else:
                result.add_error(
                    "series_invalid_image",
                    series_loc,
                    f"Series path '{series_path}' does not contain "
                    "valid Image metadata",
                    {
                        "path": series_path,
                        "type": type(series_attrs_model.ome).__name__,
                    },
                )

        return result

    def _prefetch_plate_hierarchy(
        self, zarr_group: ZarrGroup, well_paths: list[str]
    ) -> None:
        """Prefetch entire plate hierarchy to minimize network round trips.

        Strategy: Inspect first well/field to understand structure, then batch
        fetch all metadata across the entire plate (wells, fields, datasets).
        """
        if not well_paths:
            return

        # Step 1: Prefetch all well metadata
        zarr_group.prefetch_children(well_paths)

        # Step 2: Get structure from first well
        field_paths = self._get_field_paths_from_first_well(zarr_group, well_paths[0])
        if not field_paths:
            return

        # Step 3: Prefetch all field images across all wells
        all_field_paths = ("/".join(grp) for grp in product(well_paths, field_paths))
        zarr_group.prefetch_children(all_field_paths)

        # Step 4: Get dataset structure from first field
        first_well = zarr_group.get(well_paths[0])
        if not isinstance(first_well, ZarrGroup):
            return

        dataset_paths = self._get_dataset_paths_from_first_field(
            first_well, field_paths[0]
        )
        if not dataset_paths:
            return

        # Step 5: Prefetch all datasets across entire plate
        # Also prefetch labels metadata (optional, but common)
        ds_paths = (
            "/".join(grp) for grp in product(well_paths, field_paths, dataset_paths)
        )
        labels_paths = (
            "/".join(grp) + "/labels" for grp in product(well_paths, field_paths)
        )
        zarr_group.prefetch_children(chain(ds_paths, labels_paths))

    def _get_field_paths_from_first_well(
        self, zarr_group: ZarrGroup, first_well_path: str
    ) -> list[str]:
        """Extract field image paths from the first well."""
        first_well = zarr_group.get(first_well_path)
        if not isinstance(first_well, ZarrGroup):
            return []

        first_well_attrs = first_well.attrs.asdict()
        first_well_meta = validate_ome_object(first_well_attrs, OMEAttributes)

        if isinstance(first_well_meta.ome, Well):
            return [img.path for img in first_well_meta.ome.well.images]

        return []

    def _get_dataset_paths_from_first_field(
        self, first_well: ZarrGroup, first_field_path: str
    ) -> list[str]:
        """Extract dataset paths from the first field image."""
        first_field = first_well.get(first_field_path)
        if not isinstance(first_field, ZarrGroup):
            return []

        first_field_attrs = first_field.attrs.asdict()
        first_field_meta = validate_ome_object(first_field_attrs, OMEAttributes)

        if not hasattr(first_field_meta.ome, "multiscales"):
            return []

        if not isinstance(first_field_meta.ome, Image):
            return []

        if not first_field_meta.ome.multiscales:
            return []

        return [ds.path for ds in first_field_meta.ome.multiscales[0].datasets]

    def _check_for_labels_group(
        self, zarr_group: ZarrGroup, loc_prefix: Loc
    ) -> LabelsCheckResult:
        """Check for labels group at same level as datasets and return result."""
        result = ValidationResult()

        if (labels_group := zarr_group.get("labels")) is None:
            return LabelsCheckResult(result=result, labels_info=None)

        labels_loc = (*loc_prefix, "labels")

        if not isinstance(labels_group, ZarrGroup):
            result.add_error(
                "labels_not_group",
                labels_loc,
                f"Found 'labels' path but it is a {type(labels_group)}, "
                "not a zarr group",
                "labels",
            )
            return LabelsCheckResult(result=result, labels_info=None)

        attrs = labels_group.attrs.asdict()
        try:
            labels_attrs = validate_ome_object(attrs, OMEAttributes)
            if isinstance(labels_attrs.ome, LabelsGroup):
                # Return the labels info directly
                return LabelsCheckResult(
                    result=result, labels_info=(labels_group, labels_attrs.ome)
                )
        except Exception as e:
            result.add_error(
                "invalid_labels_metadata",
                labels_loc,
                f"Found a 'labels' subg-group inside of ome-zarr group {zarr_group}, "
                f"but metadata not valid LabelsGroup metadata: {e!s}",
                attrs,
            )

        return LabelsCheckResult(result=result, labels_info=None)

    def _validate_labels_image_source(
        self, zarr_group: ZarrGroup, src_img_rel_path: str, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate that label image source exists and is valid."""
        result = ValidationResult()

        # Resolve the source image path relative to the current zarr group
        try:
            image_source = _resolve_source_path(zarr_group, src_img_rel_path)
        except Exception:
            warnings.warn(
                "Unable to resolve source image path", UserWarning, stacklevel=3
            )
            return result

        try:
            img = from_uri(image_source, OMEZarrGroupJSON)
            if not isinstance(img.attributes.ome, Image):
                result.add_error(
                    "invalid_label_image_source",
                    (*loc_prefix, "image_label", "source", "image"),
                    f"Label image source '{image_source}' does not contain "
                    "valid Image ('multiscales') metadata",
                    image_source,
                )
        except Exception as e:
            result.add_error(
                "label_image_source_not_found",
                (*loc_prefix, "image_label", "source", "image"),
                f"Label image source '{image_source}' could not be opened: {e!s}",
                image_source,
            )

        return result

    def _validate_label_data_types(
        self, image_model: LabelImage, zarr_group: ZarrGroup, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate that label arrays contain only integer data types."""
        result = ValidationResult()

        # The "labels" group is not itself an image; it contains images.
        # The pixels of the label images MUST be integer data types, i.e. one of
        # [uint8, int8, uint16, int16, uint32, int32, uint64, int64].
        for ms_idx, multiscale in enumerate(image_model.multiscales):
            ms_loc = (*loc_prefix, "multiscales", ms_idx)

            for ds_idx, dataset in enumerate(multiscale.datasets):
                ds_loc = (*ms_loc, "datasets", ds_idx, "path")
                if (arr := zarr_group.get(dataset.path)) is None:
                    # Path validation will catch this separately
                    continue  # pragma: no cover

                # check if np.integer dtype
                if not (
                    isinstance(arr, ZarrArray)
                    and np.issubdtype((dt := arr.dtype), np.integer)
                ):
                    result.add_error(
                        "label_non_integer_dtype",
                        ds_loc,
                        f"Label array '{dataset.path}' has non-integer dtype "
                        f"'{dt}'. Labels must use integer types.",
                        {"path": dataset.path, "dtype": str(dt)},
                    )

        return result


# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------


def _resolve_source_path(zarr_group: ZarrGroup, src_rel_path: str) -> str:
    """Resolve a relative source path against the zarr group's store location.

    Parameters
    ----------
    zarr_group : ZarrGroup
        The zarr group to resolve relative to
    src_rel_path : str
        The relative path to resolve (e.g., "../other",
        "../../images/source.zarr")

    Returns
    -------
    str
        The resolved absolute path
    """
    import posixpath

    # Get the mapper's root path if available
    mapper = zarr_group._mapper
    path = zarr_group.path

    # Try to get the root path from the mapper
    if hasattr(mapper, "root"):
        root = mapper.root
    elif hasattr(mapper, "fs") and hasattr(mapper.fs, "root"):
        root = mapper.fs.root
    else:
        # Fall back to using the path directly
        root = ""

    # Handle URL paths
    if isinstance(root, str) and root.startswith(("http://", "https://")):
        from urllib.parse import urljoin

        # Ensure root ends with separator for proper urljoin behavior
        if not root.endswith("/"):
            root = root + "/"
        root = urljoin(root, path)
        if not root.endswith("/"):
            root = root + "/"
        return urljoin(root, src_rel_path)
    else:
        # For other filesystems, use posixpath for UNIX-style path joining
        # Most fsspec filesystems use forward slashes as separators
        return posixpath.normpath(posixpath.join(str(root), path, src_rel_path))
