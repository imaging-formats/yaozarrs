"""Storage validation for OME-ZARR v0.4 hierarchies.

This module provides functions to validate that OME-ZARR v0.4 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.
"""

from __future__ import annotations

import posixpath
import warnings
from dataclasses import dataclass
from itertools import chain, product
from typing import TypeAlias

from pydantic import TypeAdapter

from yaozarrs._storage import StorageErrorType, ValidationResult
from yaozarrs._zarr import ZarrArray, ZarrGroup, open_group
from yaozarrs.v04._bf2raw import Bf2Raw, Series
from yaozarrs.v04._image import Image, Multiscale
from yaozarrs.v04._labels import LabelImage, LabelsGroup
from yaozarrs.v04._plate import Plate, Well
from yaozarrs.v04._zarr_json import OMEZarrGroupJSON

# ----------------------------------------------------------
# VALIDATORS
# ----------------------------------------------------------

Loc: TypeAlias = tuple[int | str, ...]

# Reusable TypeAdapter for validating v04 OME-ZARR metadata
_OME_VALIDATOR = TypeAdapter(OMEZarrGroupJSON)


def _build_fs_path(zarr_group: ZarrGroup, relative_path: str = "") -> str:
    """Build a filesystem path for error reporting.

    Parameters
    ----------
    zarr_group : ZarrGroup
        The zarr group to build the path from.
    relative_path : str
        Optional relative path from the group (e.g., "A/1/0").

    Returns
    -------
    str
        A human-readable filesystem path (e.g., "plate.zarr/A/1/0").
    """
    # Get the base path from the group
    base = zarr_group.path or ""

    # Combine with relative path
    if relative_path:
        full_path = f"{base}/{relative_path}" if base else relative_path
    else:
        full_path = base

    # Try to get a readable store name (without protocol)
    store_path = zarr_group.store_path
    # Remove protocol prefix for readability (file://, https://, etc.)
    for prefix in ("file://", "https://", "http://", "s3://"):
        if store_path.startswith(prefix):
            store_path = store_path[len(prefix) :]
            break

    # Get just the store root name
    if full_path:
        # Extract just the zarr store name from the beginning of full store path
        # e.g., "/tmp/foo/plate.zarr/A/1" -> "plate.zarr/A/1"
        parts = store_path.rstrip("/").split("/")
        for _i, part in enumerate(parts):
            if part.endswith(".zarr") or part.endswith(".zarr/"):
                store_name = part.rstrip("/")
                return f"{store_name}/{full_path}"

    # Fall back to just returning the path
    return full_path or store_path.split("/")[-1]


@dataclass
class LabelsCheckResult:
    """Result of checking for a labels group."""

    result: ValidationResult
    labels_info: tuple[ZarrGroup, LabelsGroup] | None = None


class StorageValidatorV04:
    """Concrete implementation of storage validator. for OME-ZARR v0.4 spec."""

    __slots__ = ()

    @classmethod
    def validate_group(
        cls, zarr_group: ZarrGroup, metadata: OMEZarrGroupJSON | None = None
    ) -> ValidationResult:
        """Entry point that dispatches to appropriate visitor method.

        Parameters
        ----------
        zarr_group : ZarrGroup
            The zarr group to validate.
        metadata : OMEZarrGroupJSON
            The validated OME metadata model.

        Returns
        -------
        ValidationResult
            The validation result containing any errors found.
        """
        if metadata is None:
            # extract the model from the zarr attributes
            # In v04, metadata is at the top level (no "ome" wrapper)
            # Convert mappingproxy to dict for discriminator
            metadata = _OME_VALIDATOR.validate_python(dict(zarr_group.attrs))

        validator = cls()
        # In v04, metadata is directly the OME type (no .ome accessor)
        loc_prefix = ()

        # Dispatch to appropriate visitor method based on metadata type
        if isinstance(metadata, LabelImage):
            return validator.visit_label_image(zarr_group, metadata, loc_prefix)
        elif isinstance(metadata, Image):
            return validator.visit_image(zarr_group, metadata, loc_prefix)
        elif isinstance(metadata, LabelsGroup):
            return validator.visit_labels_group(zarr_group, metadata, loc_prefix)
        elif isinstance(metadata, Plate):
            return validator.visit_plate(zarr_group, metadata, loc_prefix)
        elif isinstance(metadata, Well):
            return validator.visit_well(zarr_group, metadata, loc_prefix)
        elif isinstance(metadata, Bf2Raw):
            return validator.visit_bioformats2raw(zarr_group, metadata, loc_prefix)
        elif isinstance(metadata, Series):  # pragma: no cover
            return validator.visit_series(zarr_group, metadata, loc_prefix)
        else:
            raise NotImplementedError(
                f"Unknown OME metadata type: {type(metadata).__name__}"
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
                    StorageErrorType.label_path_not_found,
                    label_loc,
                    f"Label path '{label_path}' not found in labels group",
                    ctx={
                        "fs_path": _build_fs_path(labels_group, label_path),
                        "expected": "zarr group",
                    },
                )
                continue

            label_group = labels_group[label_path]
            if not isinstance(label_group, ZarrGroup):
                result.add_error(
                    StorageErrorType.label_path_not_group,
                    label_loc,
                    f"Label path '{label_path}' is not a zarr group",
                    ctx={
                        "fs_path": _build_fs_path(labels_group, label_path),
                        "expected": "group",
                        "found": "array",
                    },
                )
                continue

            # Validate as LabelImage
            try:
                label_image_model = label_group.ome_metadata(version="0.4")
            except ValueError as e:
                label_image_model = e
            if not isinstance(label_image_model, Image):
                ctx: dict = {"path": label_path}
                if isinstance(label_image_model, Exception):
                    ctx["error"] = label_image_model
                else:
                    ctx["type"] = type(label_image_model).__name__
                result.add_error(
                    StorageErrorType.label_image_invalid,
                    label_loc,
                    f"Label path '{label_path}' does not contain "
                    "valid Image ('multiscales') metadata",
                    ctx=ctx,
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
                        StorageErrorType.label_multiscale_count_mismatch,
                        label_loc,
                        f"Label image '{label_path}' has {n_lbl_ms} "
                        f"multiscales, but parent image has {n_img_ms}",
                        ctx={
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
                            StorageErrorType.label_dataset_count_mismatch,
                            (*label_loc, "multiscales", ms_idx),
                            f"Label image '{label_path}' multiscale index {ms_idx} "
                            f"has {n_lbl_ds} datasets, but parent image multiscale "
                            f"index {ms_idx} has {n_img_ds}",
                            ctx={
                                "label_path": label_path,
                                "multiscale_index": ms_idx,
                                "label_datasets": n_lbl_ds,
                                "parent_datasets": n_img_ds,
                            },
                        )

            if isinstance(label_image_model, LabelImage):
                # Recursively validate the label image
                result = result.merge(
                    self.visit_label_image(label_group, label_image_model, label_loc)
                )
            else:
                result.add_warning(
                    StorageErrorType.label_image_invalid,
                    label_loc,
                    f"Label path '{label_path}' contains Image metadata, "
                    "but is not a LabelImage (SHOULD contain 'image-label' metadata)",
                    ctx={"path": label_path, "type": type(label_image_model).__name__},
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
                    StorageErrorType.dataset_path_not_found,
                    ds_loc,
                    f"Dataset path '{dataset.path}' not found in zarr group",
                    ctx={
                        "fs_path": _build_fs_path(zarr_group, dataset.path),
                        "expected": "zarr array",
                    },
                )
                continue

            if not isinstance(arr, ZarrArray):
                result.add_error(
                    StorageErrorType.dataset_not_array,
                    ds_loc,
                    f"Dataset path '{dataset.path}' exists but is not a zarr array",
                    ctx={
                        "fs_path": _build_fs_path(zarr_group, dataset.path),
                        "expected": "array",
                        "found": "group",
                    },
                )
                continue

            # Check array dimensionality matches axes
            expected_ndim = len(multiscale.axes)
            if arr.ndim != expected_ndim:
                result.add_error(
                    StorageErrorType.dataset_dimension_mismatch,
                    ds_loc,
                    f"Dataset '{dataset.path}' has {arr.ndim} dimensions "
                    f"but axes specify {expected_ndim}",
                    ctx={
                        "fs_path": _build_fs_path(zarr_group, dataset.path),
                        "actual_ndim": arr.ndim,
                        "expected_ndim": expected_ndim,
                        "axes": [ax.name for ax in multiscale.axes],
                    },
                )

            # Check dimension_names attribute matches axes
            if dim_names := list(dict(arr.attrs).get("dimension_names", [])):
                expected_names = [ax.name for ax in multiscale.axes]
                if dim_names != expected_names:
                    result.add_error(
                        StorageErrorType.dimension_names_mismatch,
                        (*ds_loc, "dimension_names"),
                        f"Array dimension_names {dim_names} don't match "
                        f"axes names {expected_names}",
                        ctx={"actual": dim_names, "expected": expected_names},
                    )

        return result

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
                    StorageErrorType.well_path_not_found,
                    (*well_loc, "path"),
                    f"Well path '{well.path}' not found in plate group",
                    ctx={
                        "fs_path": _build_fs_path(zarr_group, well.path),
                        "expected": "zarr group",
                    },
                )
                continue

            if not isinstance(well_group, ZarrGroup):
                result.add_error(
                    StorageErrorType.well_path_not_group,
                    (*well_loc, "path"),
                    f"Well path '{well.path}' is not a zarr group",
                    ctx={
                        "fs_path": _build_fs_path(zarr_group, well.path),
                        "expected": "group",
                        "found": "array",
                    },
                )
                continue

            # Validate well metadata
            try:
                well_model = well_group.ome_metadata(version="0.4")
            except ValueError as e:
                well_model = e
            if isinstance(well_model, Well):
                result = result.merge(self.visit_well(well_group, well_model, well_loc))
            else:
                ctx: dict = {"path": well.path}
                if isinstance(well_model, Exception):
                    ctx["error"] = well_model
                else:
                    ctx["type"] = type(well_model).__name__
                result.add_error(
                    StorageErrorType.well_invalid,
                    well_loc,
                    f"Well path '{well.path}' does not contain valid Well metadata",
                    ctx=ctx,
                )

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
                    StorageErrorType.field_path_not_found,
                    (*field_loc, "path"),
                    f"Field path '{field_image.path}' not found in well group",
                    ctx={
                        "fs_path": _build_fs_path(zarr_group, field_image.path),
                        "expected": "zarr group",
                    },
                )
                continue

            if not isinstance(field_group, ZarrGroup):
                result.add_error(
                    StorageErrorType.field_path_not_group,
                    (*field_loc, "path"),
                    f"Field path '{field_image.path}' is not a zarr group",
                    ctx={
                        "fs_path": _build_fs_path(zarr_group, field_image.path),
                        "expected": "group",
                        "found": "array",
                    },
                )
                continue

            # Validate field as image group
            try:
                field_group_model = field_group.ome_metadata(version="0.4")
            except ValueError as e:
                field_group_model = e
            if isinstance(field_group_model, Image):
                result = result.merge(
                    self.visit_image(field_group, field_group_model, field_loc)
                )
            else:
                ctx: dict = {"fs_path": _build_fs_path(zarr_group, field_image.path)}
                if isinstance(field_group_model, Exception):
                    ctx["error"] = field_group_model
                else:
                    ctx["type"] = type(field_group_model).__name__
                result.add_error(
                    StorageErrorType.field_image_invalid,
                    field_loc,
                    f"Field path '{field_image.path}' does not contain "
                    "valid Image metadata",
                    ctx=ctx,
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
            try:
                # In v04, metadata is at the top level (no "ome" wrapper)
                ome_metadata = _OME_VALIDATOR.validate_python(dict(ome_group.attrs))

                # If OME group has series metadata, use that to find images
                if isinstance(ome_metadata, Series):
                    # Validate using the series paths
                    result = result.merge(
                        self.visit_series(
                            zarr_group, ome_metadata, (*loc_prefix, "OME")
                        )
                    )
                    return result
            except Exception:
                # OME group exists but doesn't have valid OME metadata
                # Fall through to numbered directory validation
                pass  # pragma: no cover

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
                StorageErrorType.bf2raw_no_images,
                loc_prefix,
                "Bioformats2raw group contains no numbered image directories",
            )
            return result

        # Prefetch all numbered path metadata
        zarr_group.prefetch_children(numbered_paths)

        # Validate each numbered directory as image group
        for path in numbered_paths:
            image_loc = (*loc_prefix, path)

            image_group = zarr_group.get(path)
            if not isinstance(image_group, ZarrGroup):
                result.add_error(
                    StorageErrorType.bf2raw_path_not_group,
                    image_loc,
                    f"Bioformats2raw path '{path}' is not a zarr group",
                    ctx={"path": path, "expected": "group", "found": "array"},
                )
                continue

            # Validate as image group
            try:
                image_group_meta = image_group.ome_metadata(version="0.4")
            except ValueError as e:
                image_group_meta = e
            if isinstance(image_group_meta, Image):
                result = result.merge(
                    self.visit_image(image_group, image_group_meta, image_loc)
                )
            else:
                ctx: dict = {"path": path}
                if isinstance(image_group_meta, Exception):
                    ctx["error"] = image_group_meta
                else:
                    ctx["type"] = type(image_group_meta).__name__
                result.add_error(
                    StorageErrorType.bf2raw_invalid_image,
                    image_loc,
                    f"Bioformats2raw path '{path}' does not contain "
                    "valid Image metadata",
                    ctx=ctx,
                )

        return result

    def visit_series(
        self, zarr_group: ZarrGroup, ome_model: Series, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate an OME group with series metadata.

        The series attribute is a list of paths to image groups. Each path
        should point to a valid image group with multiscales metadata.

        IMPORTANT! `zarr_group` here is the *parent* of the OME group, not the
        OME group itself.

        ```
        top_group.zarr      <-- what must be passed to `zarr_group` here
        ├── 0/
        │   ├── zarr.json   <-- contains multiscales metadata
        ├── OME/
        │   ├── zarr.json   <-- contains `ome_model` Series model
        └── zarr.json       <-- contains bioformats2raw metadata
        ```
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
                    StorageErrorType.series_path_not_found,
                    series_loc,
                    f"Series path '{series_path}' not found in series group",
                    ctx={"path": series_path, "expected": "zarr group"},
                )
                continue

            if not isinstance(series_group, ZarrGroup):
                result.add_error(
                    StorageErrorType.series_path_not_group,
                    series_loc,
                    f"Series path '{series_path}' is not a zarr group",
                    ctx={"path": series_path, "expected": "group", "found": "array"},
                )
                continue

            # Validate series as image group
            try:
                series_group_meta = series_group.ome_metadata(version="0.4")
            except ValueError as e:
                series_group_meta = e
            if isinstance(series_group_meta, Image):
                result = result.merge(
                    self.visit_image(series_group, series_group_meta, series_loc)
                )
            else:
                ctx: dict = {"path": series_path}
                if isinstance(series_group_meta, Exception):
                    ctx["error"] = series_group_meta
                else:
                    ctx["type"] = type(series_group_meta).__name__
                result.add_error(
                    StorageErrorType.series_invalid_image,
                    series_loc,
                    f"Series path '{series_path}' does not contain "
                    "valid Image metadata",
                    ctx=ctx,
                )

        return result

    def _prefetch_plate_hierarchy(
        self, zarr_group: ZarrGroup, well_paths: list[str]
    ) -> None:
        """Prefetch entire plate hierarchy to minimize network round trips.

        Strategy: Inspect first well/field to understand structure, then batch
        fetch all metadata across the entire plate (wells, fields, datasets).
        """
        if not well_paths:  # pragma: no cover
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
            return  # pragma: no cover

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

        try:
            # In v04, metadata is at the top level (no "ome" wrapper)
            first_well_meta = _OME_VALIDATOR.validate_python(dict(first_well.attrs))
            if isinstance(first_well_meta, Well):
                return [img.path for img in first_well_meta.well.images]
        except Exception:
            pass
        return []

    def _get_dataset_paths_from_first_field(
        self, first_well: ZarrGroup, first_field_path: str
    ) -> list[str]:
        """Extract dataset paths from the first field image."""
        first_field = first_well.get(first_field_path)
        if not isinstance(first_field, ZarrGroup):
            return []  # pragma: no cover

        try:
            # In v04, metadata is at the top level (no "ome" wrapper)
            first_field_meta = _OME_VALIDATOR.validate_python(dict(first_field.attrs))
        except Exception:
            return []  # pragma: no cover

        if not isinstance(first_field_meta, Image) or not first_field_meta.multiscales:
            return []  # pragma: no cover

        return [ds.path for ds in first_field_meta.multiscales[0].datasets]

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
                StorageErrorType.labels_not_group,
                labels_loc,
                f"Found 'labels' path but it is a {type(labels_group)}, "
                "not a zarr group",
                ctx={"expected": "group", "found": type(labels_group).__name__},
            )
            return LabelsCheckResult(result=result, labels_info=None)

        try:
            # In v04, metadata is at the top level (no "ome" wrapper)
            labels_attrs = _OME_VALIDATOR.validate_python(dict(labels_group.attrs))
            if isinstance(labels_attrs, LabelsGroup):
                # Return the labels info directly
                return LabelsCheckResult(
                    result=result, labels_info=(labels_group, labels_attrs)
                )
        except Exception as e:
            result.add_error(
                StorageErrorType.labels_metadata_invalid,
                labels_loc,
                f"Found a 'labels' subg-group inside of ome-zarr group {zarr_group}, "
                f"but metadata not valid LabelsGroup metadata: {e!s}",
                ctx={"error": str(e)},
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
            # In v04, we open the zarr group and validate the attrs directly
            source_group = open_group(image_source)
            img_metadata = _OME_VALIDATOR.validate_python(dict(source_group.attrs))

            if not isinstance(img_metadata, Image):
                result.add_error(
                    StorageErrorType.label_image_source_invalid,
                    (*loc_prefix, "image_label", "source", "image"),
                    f"Label image source '{image_source}' does not contain "
                    "valid Image ('multiscales') metadata",
                    ctx={"source": image_source, "expected": "Image"},
                )
        except Exception as e:
            result.add_error(
                StorageErrorType.label_image_source_not_found,
                (*loc_prefix, "image_label", "source", "image"),
                f"Label image source '{image_source}' could not be opened: {e!s}",
                ctx={"source": image_source, "error": str(e)},
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

                # check if integer dtype
                if isinstance(arr, ZarrArray):
                    dt = arr.dtype
                    if not _is_integer_dtype(dt):
                        result.add_error(
                            StorageErrorType.label_non_integer_dtype,
                            ds_loc,
                            f"Label array '{dataset.path}' has non-integer dtype "
                            f"'{dt}'. Labels must use integer types.",
                            ctx={
                                "path": dataset.path,
                                "dtype": str(
                                    dt,
                                ),
                            },
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
    # Get the mapper's root path if available
    mapper = zarr_group._store
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


def _is_integer_dtype(dtype_str: str) -> bool:
    """Check if a dtype string represents an integer type.

    Parameters
    ----------
    dtype_str : str
        The dtype string to check (e.g., '<i2', 'uint8', 'int32')

    Returns
    -------
    bool
        True if the dtype represents an integer type, False otherwise
    """
    # Remove endianness markers
    dtype_clean = dtype_str.lstrip("<>=|")
    # Check for integer type indicators
    return dtype_clean.startswith(("int", "uint")) or (
        len(dtype_clean) >= 2
        and dtype_clean[0] in ("i", "u")
        and dtype_clean[1].isdigit()
    )
