"""Storage validation shared by OME-Zarr versions.

`StorageValidator` walks a hierarchy from its root group and checks that the groups
and arrays named in the metadata exist and agree with it (dimensionality, label
dtypes, ...).  Each version's subclass sets the version string and its model
classes, parses metadata with its own models, and may add per-array checks.
"""

from __future__ import annotations

import posixpath
import re
import warnings
from itertools import chain, product
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias
from urllib.parse import urljoin

from pydantic import ValidationError

from yaozarrs._storage import StorageErrorType, ValidationResult
from yaozarrs._zarr import ZarrArray, ZarrGroup, _open_group_at

if TYPE_CHECKING:
    from collections.abc import Mapping

    from yaozarrs._storage import ErrorDetails

Loc: TypeAlias = tuple[int | str, ...]

_PROTOCOL = re.compile(r"^(?:file|https?|s3)://")


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
    full_path = "/".join(p for p in (zarr_group.path, relative_path) if p)
    # Remove protocol prefix for readability (file://, https://, etc.)
    store_path = _PROTOCOL.sub("", zarr_group._display_name(), count=1)
    if full_path:
        # Start from the name of the zarr store in the store's path,
        # e.g., "/tmp/foo/plate.zarr" and "A/1" -> "plate.zarr/A/1"
        for part in store_path.rstrip("/").split("/"):
            if part.endswith(".zarr"):
                return f"{part}/{full_path}"
    return full_path or store_path.split("/")[-1]


def _invalid_ctx(model: Any) -> dict:
    """Error ctx for metadata that isn't the expected model: the error, or its type."""
    if isinstance(model, Exception):
        return {"error": model}
    return {"type": type(model).__name__}


class StorageValidator:
    """Storage validator for one OME-Zarr version (see subclasses)."""

    __slots__ = ("strict",)

    strict: bool | None

    # set by each version's subclass
    VERSION: ClassVar[str]
    # location of the OME metadata within a group's attributes, for error locs
    LOC_PREFIX: ClassVar[Loc] = ("ome",)
    Image: ClassVar[type]
    LabelImage: ClassVar[type]
    LabelsGroup: ClassVar[type]
    Plate: ClassVar[type]
    Well: ClassVar[type]
    Bf2Raw: ClassVar[type]
    Series: ClassVar[type]

    def _parse_ome(self, attrs: Mapping[str, Any]) -> Any:
        """Validate a group's attributes, returning its OME metadata model."""
        raise NotImplementedError

    def _parse_ome_uri(self, uri: str) -> Any:
        """Validate the group at `uri`, returning its OME metadata model."""
        raise NotImplementedError

    def _validate_dataset_array(
        self, arr: ZarrArray, dataset: Any, multiscale: Any, ds_loc: Loc
    ) -> ValidationResult:
        """Version-specific checks of one dataset array (none by default)."""
        return ValidationResult()

    def _child_model(self, group: ZarrGroup) -> Any:
        """Return the OME metadata model of `group`, or the `ValueError` raised."""
        try:
            return group.ome_metadata(version=self.VERSION, strict=self.strict)
        except ValueError as e:
            return e

    @classmethod
    def validate_group(
        cls, zarr_group: ZarrGroup, *, strict: bool | None = None
    ) -> ValidationResult:
        """Entry point that dispatches to appropriate visitor method.

        Parameters
        ----------
        zarr_group : ZarrGroup
            The zarr group to validate.
        strict : bool | None
            If `True`, reject metadata that would otherwise be coerced to the right
            type (e.g. `"1"` for an integer).

        Returns
        -------
        ValidationResult
            The validation result containing any errors found.
        """
        validator = cls()
        validator.strict = strict
        try:
            ome_metadata = validator._parse_ome(zarr_group.attrs)
        except ValidationError as e:
            # report invalid root metadata like the rest of the hierarchy, one entry
            # per pydantic error (the locs already start at the root, e.g. "ome")
            result = ValidationResult()
            for err in e.errors():
                details: ErrorDetails = {
                    "type": err["type"],
                    "loc": err["loc"],
                    "msg": err["msg"],
                }
                if "ctx" in err:
                    details["ctx"] = err["ctx"]
                result.errors.append(details)
            return result
        return validator.visit_root(zarr_group, ome_metadata, cls.LOC_PREFIX)

    def visit_root(
        self, zarr_group: ZarrGroup, ome_metadata: Any, loc_prefix: Loc
    ) -> ValidationResult:
        """Dispatch to the visitor method for the root group's metadata type."""
        if isinstance(ome_metadata, self.LabelImage):
            return self.visit_label_image(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, self.Image):
            return self.visit_image(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, self.LabelsGroup):
            return self.visit_labels_group(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, self.Plate):
            return self.visit_plate(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, self.Well):
            return self.visit_well(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, self.Bf2Raw):
            return self.visit_bioformats2raw(zarr_group, ome_metadata, loc_prefix)
        elif isinstance(ome_metadata, self.Series):
            if (parent := _parent_group(zarr_group)) is None:
                warnings.warn(
                    "Cannot validate series paths: they are relative to the parent of "
                    "the OME group, which is outside of the store",
                    UserWarning,
                    stacklevel=2,
                )
                return ValidationResult()
            return self.visit_series(parent, ome_metadata, loc_prefix)
        else:
            raise NotImplementedError(
                f"Unknown OME metadata type: {type(ome_metadata).__name__}"
            )

    def visit_label_image(
        self, zarr_group: ZarrGroup, label_image_model: Any, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a LabelImage group."""
        # The `zarr.json` file for the label image MUST implement the multiscales
        # specification.
        result = self.visit_image(zarr_group, label_image_model, loc_prefix)

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
        self, zarr_group: ZarrGroup, image_model: Any, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate an image group with multiscales metadata."""
        result = ValidationResult()

        # Prefetch the datasets of every multiscale, and any labels group, in one batch
        zarr_group.prefetch_children(
            [
                *(ds.path for ms in image_model.multiscales for ds in ms.datasets),
                "labels",
            ]
        )
        for ms_idx, multiscale in enumerate(image_model.multiscales):
            ms_loc = (*loc_prefix, "multiscales", ms_idx)
            result = result.merge(
                self._visit_multiscale_no_prefetch(zarr_group, multiscale, ms_loc)
            )

        # Check whether this image has a labels group, and validate if so
        labels_result, labels = self._check_for_labels_group(zarr_group, loc_prefix)
        result = result.merge(labels_result)
        if labels is not None:
            labels_group, labels_model = labels
            result = result.merge(
                self.visit_labels_group(
                    labels_group, labels_model, (*loc_prefix, "labels"), image_model
                )
            )
        return result

    def visit_labels_group(
        self,
        labels_group: ZarrGroup,
        labels_model: Any,
        loc_prefix: Loc,
        parent_image_model: Any = None,
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

            # (validate as an Image here; LabelImage is checked below)
            label_image_model = self._child_model(label_group)
            if not isinstance(label_image_model, self.Image):
                result.add_error(
                    StorageErrorType.label_image_invalid,
                    label_loc,
                    f"Label path '{label_path}' does not contain "
                    "valid Image ('multiscales') metadata",
                    ctx={"path": label_path, **_invalid_ctx(label_image_model)},
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

            if isinstance(label_image_model, self.LabelImage):
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
                result = result.merge(
                    self._visit_unmarked_label_image(
                        label_group, label_image_model, label_loc
                    )
                )

        return result

    def _visit_unmarked_label_image(
        self, zarr_group: ZarrGroup, image_model: Any, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a listed label path that has only Image metadata."""
        return self.visit_image(zarr_group, image_model, loc_prefix)

    def _visit_multiscale_no_prefetch(
        self, zarr_group: ZarrGroup, multiscale: Any, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate multiscale without prefetching (assumes already prefetched)."""
        result = ValidationResult()
        expected_ndim = len(multiscale.axes)
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

            result = result.merge(
                self._validate_dataset_array(arr, dataset, multiscale, ds_loc)
            )

        return result

    def visit_plate(
        self, zarr_group: ZarrGroup, plate_model: Any, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a plate group and its wells."""
        result = ValidationResult()
        # Prefetch all metadata in one go to minimize network round trips
        self._prefetch_plate_hierarchy(
            zarr_group, [well.path for well in plate_model.plate.wells]
        )
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

            well_model = self._child_model(well_group)
            if isinstance(well_model, self.Well):
                result = result.merge(self.visit_well(well_group, well_model, well_loc))
                result = result.merge(
                    self._validate_well_acquisitions(plate_model, well_model, well_loc)
                )
            else:
                result.add_error(
                    StorageErrorType.well_invalid,
                    well_loc,
                    f"Well path '{well.path}' does not contain valid Well metadata",
                    ctx={"path": well.path, **_invalid_ctx(well_model)},
                )

        return result

    def _validate_well_acquisitions(
        self, plate_model: Any, well_model: Any, loc_prefix: Loc
    ) -> ValidationResult:
        """Check the wells' field-of-view acquisitions against the plate's."""
        result = ValidationResult()
        # If multiple acquisitions were performed in the plate, it MUST contain an
        # `acquisition` key whose value MUST be an integer identifying the acquisition
        # which MUST match one of the acquisition JSON objects defined in the plate
        # metadata.
        # (like ome-zarr-models, an id isn't checked when the plate lists none)
        ids = [acq.id for acq in plate_model.plate.acquisitions or ()]
        for idx, field_image in enumerate(well_model.well.images):
            loc = (*loc_prefix, "well", "images", idx, "acquisition")
            if field_image.acquisition is None:
                if len(ids) > 1:
                    result.add_error(
                        StorageErrorType.field_acquisition_missing,
                        loc,
                        f"Field '{field_image.path}' has no 'acquisition', but the "
                        f"plate has {len(ids)} acquisitions",
                        ctx={"path": field_image.path, "acquisitions": ids},
                    )
            elif ids and field_image.acquisition not in ids:
                result.add_error(
                    StorageErrorType.field_acquisition_invalid,
                    loc,
                    f"Field '{field_image.path}' has acquisition "
                    f"{field_image.acquisition}, which is not one of the plate's "
                    f"acquisitions {ids}",
                    ctx={
                        "path": field_image.path,
                        "acquisition": field_image.acquisition,
                        "acquisitions": ids,
                    },
                )
        return result

    def visit_well(
        self, zarr_group: ZarrGroup, well_model: Any, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a well group and its field images."""
        result = ValidationResult()
        # (in a plate, `visit_plate` has usually prefetched these already)
        zarr_group.prefetch_children([field.path for field in well_model.well.images])
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

            field_model = self._child_model(field_group)
            if isinstance(field_model, self.Image):
                result = result.merge(
                    self.visit_image(field_group, field_model, field_loc)
                )
            else:
                fs_path = _build_fs_path(zarr_group, field_image.path)
                result.add_error(
                    StorageErrorType.field_image_invalid,
                    field_loc,
                    f"Field path '{field_image.path}' does not contain "
                    "valid Image metadata",
                    ctx={"fs_path": fs_path, **_invalid_ctx(field_model)},
                )

        return result

    def visit_bioformats2raw(
        self, zarr_group: ZarrGroup, bf2raw_model: Any, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a bioformats2raw layout.

        According to spec:
        1. Check for OME subgroup with optional "series" metadata
        2. If series exists, validate those paths
        3. Otherwise, validate consecutively numbered directories (0/, 1/, 2/, ...)
        """
        result = ValidationResult()

        zarr_group.prefetch_children(["OME"])
        if isinstance(ome_group := zarr_group.get("OME"), ZarrGroup):
            try:
                ome_model = self._parse_ome(ome_group.attrs)
            except ValueError:
                ome_model = None  # (fall back to the numbered directories)
            if isinstance(ome_model, self.Series):
                return self.visit_series(zarr_group, ome_model, (*loc_prefix, "OME"))

        # Without series metadata, find consecutively numbered directories (0, 1, ...)
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

            image_model = self._child_model(image_group)
            if isinstance(image_model, self.Image):
                result = result.merge(
                    self.visit_image(image_group, image_model, image_loc)
                )
            else:
                result.add_error(
                    StorageErrorType.bf2raw_invalid_image,
                    image_loc,
                    f"Bioformats2raw path '{path}' does not contain "
                    "valid Image metadata",
                    ctx={"path": path, **_invalid_ctx(image_model)},
                )

        return result

    def visit_series(
        self, zarr_group: ZarrGroup, ome_model: Any, loc_prefix: Loc
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
        zarr_group.prefetch_children(ome_model.series)
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

            series_model = self._child_model(series_group)
            if isinstance(series_model, self.Image):
                result = result.merge(
                    self.visit_image(series_group, series_model, series_loc)
                )
            else:
                result.add_error(
                    StorageErrorType.series_invalid_image,
                    series_loc,
                    f"Series path '{series_path}' does not contain "
                    "valid Image metadata",
                    ctx={"path": series_path, **_invalid_ctx(series_model)},
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
        zarr_group.prefetch_children(well_paths)

        first_well = zarr_group.get(well_paths[0])
        if not isinstance(first_well, ZarrGroup):
            return
        if not (field_paths := self._field_paths(first_well)):
            return
        zarr_group.prefetch_children(
            "/".join(grp) for grp in product(well_paths, field_paths)
        )

        if not (dataset_paths := self._dataset_paths(first_well.get(field_paths[0]))):
            return
        # (and labels metadata: optional, but common)
        ds_paths = (
            "/".join(grp) for grp in product(well_paths, field_paths, dataset_paths)
        )
        labels_paths = (
            "/".join(grp) + "/labels" for grp in product(well_paths, field_paths)
        )
        zarr_group.prefetch_children(chain(ds_paths, labels_paths))

    def _field_paths(self, well: ZarrGroup) -> list[str]:
        """Field image paths listed by the metadata of `well`, if it is a Well."""
        try:
            well_meta = self._parse_ome(well.attrs)
            if isinstance(well_meta, self.Well):
                return [img.path for img in well_meta.well.images]
        except Exception:
            pass
        return []

    def _dataset_paths(self, field: ZarrGroup | ZarrArray | None) -> list[str]:
        """Dataset paths of the first multiscale of `field`, if it is an Image."""
        if not isinstance(field, ZarrGroup):
            return []  # pragma: no cover

        try:
            field_meta = self._parse_ome(field.attrs)
        except Exception:
            return []  # pragma: no cover

        if not isinstance(field_meta, self.Image) or not field_meta.multiscales:
            return []  # pragma: no cover

        return [ds.path for ds in field_meta.multiscales[0].datasets]

    def _check_for_labels_group(
        self, zarr_group: ZarrGroup, loc_prefix: Loc
    ) -> tuple[ValidationResult, tuple[ZarrGroup, Any] | None]:
        """Check for a labels group at the same level as the datasets.

        Returns the errors found, and the labels group and its model, if valid.
        """
        result = ValidationResult()

        if (labels_group := zarr_group.get("labels")) is None:
            return result, None

        labels_loc = (*loc_prefix, "labels")

        if not isinstance(labels_group, ZarrGroup):
            result.add_error(
                StorageErrorType.labels_not_group,
                labels_loc,
                f"Found 'labels' path but it is a {type(labels_group)}, "
                "not a zarr group",
                ctx={"expected": "group", "found": type(labels_group).__name__},
            )
            return result, None

        try:
            labels_model = self._parse_ome(labels_group.attrs)
            if isinstance(labels_model, self.LabelsGroup):
                return result, (labels_group, labels_model)
            # The OME-Zarr Metadata in the zarr.json file associated with the "labels"
            # group MUST contain a JSON object with the key labels ...
            # (but it's valid OME metadata, and the image is fine: so warn and skip)
            found = type(labels_model).__name__
            result.add_warning(
                StorageErrorType.labels_metadata_invalid,
                labels_loc,
                f"Skipping 'labels' sub-group inside of ome-zarr group {zarr_group}: "
                f"it has {found} metadata, not LabelsGroup metadata",
                ctx={"type": found},
            )
        except Exception as e:
            result.add_error(
                StorageErrorType.labels_metadata_invalid,
                labels_loc,
                f"Found a 'labels' subg-group inside of ome-zarr group {zarr_group}, "
                f"but metadata not valid LabelsGroup metadata: {e!s}",
                ctx={"error": str(e)},
            )

        return result, None

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

        store_key = _source_store_key(zarr_group, src_img_rel_path)
        if store_key is None and not zarr_group._has_uri():
            warnings.warn(
                f"Cannot validate label image source {src_img_rel_path!r}: "
                "it is outside of the store",
                UserWarning,
                stacklevel=3,
            )
            return result

        try:
            if store_key is not None:
                source_group = _open_group_at(zarr_group._store, store_key)
                img = self._parse_ome(source_group.attrs)
            else:
                img = self._parse_ome_uri(image_source)
            if not isinstance(img, self.Image):
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
        self, image_model: Any, zarr_group: ZarrGroup, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate that label arrays contain only integer data types."""
        result = ValidationResult()

        # The pixels of the label images MUST be integer data types, i.e. one of
        # [uint8, int8, uint16, int16, uint32, int32, uint64, int64].
        for ms_idx, multiscale in enumerate(image_model.multiscales):
            ms_loc = (*loc_prefix, "multiscales", ms_idx)
            for ds_idx, dataset in enumerate(multiscale.datasets):
                # (a missing or non-array path is reported by the dataset checks)
                arr = zarr_group.get(dataset.path)
                if isinstance(arr, ZarrArray) and not _is_integer_dtype(
                    dt := arr.dtype
                ):
                    result.add_error(
                        StorageErrorType.label_non_integer_dtype,
                        (*ms_loc, "datasets", ds_idx, "path"),
                        f"Label array '{dataset.path}' has non-integer dtype "
                        f"'{dt}'. Labels must use integer types.",
                        ctx={"path": dataset.path, "dtype": str(dt)},
                    )

        return result


def _parent_group(zarr_group: ZarrGroup) -> ZarrGroup | None:
    """Return the parent of `zarr_group`, or None if it can't be reached.

    The parent of a store's root can only be reached through a filesystem.
    """
    if zarr_group.path:
        return _open_group_at(zarr_group._store, posixpath.dirname(zarr_group.path))
    if not zarr_group._has_uri():
        return None
    mapper, full_path, _ = zarr_group._full_path()
    parent_root = full_path.rstrip("/").rpartition("/")[0]
    return ZarrGroup(mapper.fs.get_mapper(parent_root))


def _source_store_key(zarr_group: ZarrGroup, src_rel_path: str) -> str | None:
    """Return the path of a source within the store, or None if it's outside of it."""
    if "://" in src_rel_path or src_rel_path.startswith("/"):
        return None
    key = posixpath.normpath(posixpath.join(zarr_group.path, src_rel_path))
    if key == ".." or key.startswith("../"):
        return None
    return "" if key == "." else key


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
    mapper = zarr_group._store
    path = zarr_group.path
    if hasattr(mapper, "root"):
        root = mapper.root
    elif hasattr(mapper, "fs") and hasattr(mapper.fs, "root"):
        root = mapper.fs.root
    else:
        root = ""

    if isinstance(root, str) and root.startswith(("http://", "https://")):
        # (urljoin replaces the last segment of a base without a trailing "/")
        if not root.endswith("/"):
            root += "/"
        root = urljoin(root, path)
        if not root.endswith("/"):
            root += "/"
        return urljoin(root, src_rel_path)
    # (most fsspec filesystems use forward slashes as separators)
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
