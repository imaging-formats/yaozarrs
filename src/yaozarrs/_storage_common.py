"""Common storage validation for OME-ZARR hierarchies.

This module provides shared functionality for validating OME-ZARR storage
structures across different versions (v0.4 and v0.5).

Requires zarr to be installed for full validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from yaozarrs._base import ZarrGroupModel


class ValidationMessages:
    """Centralized validation messages."""

    MISSING_OME_METADATA = "Missing 'ome' metadata in attributes"
    ZARR_ACCESS_ERROR = "Cannot access zarr group metadata: {}"
    DATASET_PATH_REQUIRED = "{}.datasets[{}]: path is required"
    AXES_REQUIRED = "{}: axes are required"
    AXES_UNIQUE = "{}: axis names must be unique"
    DATASETS_REQUIRED = "{}: datasets are required"
    DATASET_NOT_ARRAY = "Dataset path is not a zarr array: {}"
    DATASET_NOT_EXIST = "Dataset path does not exist: {}"
    DATASET_ACCESS_ERROR = "Error accessing dataset: {}"
    ARRAY_ACCESS_ERROR = "Zarr array has no accessible dtype"
    ARRAY_VALIDATION_ERROR = "Error validating zarr array: {}"
    SCALE_LENGTH_MISMATCH = (
        "{}.coordinateTransformations[{}]: scale length ({}) must match axes count ({})"
    )
    TRANSLATION_LENGTH_MISMATCH = (
        "{}.coordinateTransformations[{}]: "
        "translation length ({}) must match axes count ({})"
    )
    UNKNOWN_TRANSFORM_TYPE = (
        "{}.coordinateTransformations[{}]: unknown transformation type '{}'"
    )
    SCALE_TRANSFORM_REQUIRED = "{}: at least one 'scale' transformation is required"


class ZarrFormats:
    """Zarr format constants."""

    FORMAT_V2 = 2
    FORMAT_V3 = 3


# Try to import zarr, but make it optional
try:
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    zarr = None  # type: ignore


class ValidationError(NamedTuple):
    """A validation error found during storage validation."""

    path: str
    message: str
    severity: str  # "error" or "warning"


class ValidationResult(NamedTuple):
    """Result of storage validation."""

    valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]


class BaseStorageValidator(ABC):
    """Base validator for OME-ZARR storage."""

    def __init__(self, group_uri: str, zarr_group):
        self.group_uri = group_uri
        self.zarr_group = zarr_group
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []

    @staticmethod
    @abstractmethod
    def _extract_group_uri(uri: str) -> str:
        """Extract the base zarr group URI from a model URI."""
        pass

    @abstractmethod
    def _validate_zarr_format(self, zarr_metadata) -> None:
        """Validate zarr format requirements for this version."""
        pass

    @abstractmethod
    def _validate_version_metadata(self, metadata: dict[str, Any]) -> None:
        """Validate version-specific metadata requirements."""
        pass

    @abstractmethod
    def _get_coordinate_transforms_missing_scale_severity(self) -> str:
        """Get severity level for missing scale transformations."""
        pass

    def validate(self) -> ValidationResult:
        """Run complete validation."""
        try:
            metadata_result = self._extract_and_validate_metadata()
            if not metadata_result:
                return self._result()

            zarr_metadata, ome_metadata = metadata_result
            self._validate_zarr_format(zarr_metadata)
            self._validate_version_metadata(ome_metadata)
            self._validate_ome_structure(ome_metadata)

            return self._result()

        except Exception as e:
            self._add_error(self.group_uri, f"Unexpected error during validation: {e}")
            return self._result()

    def _extract_and_validate_metadata(self) -> tuple[Any, dict[str, Any]] | None:
        """Extract and validate basic zarr group metadata."""
        try:
            zarr_metadata = self.zarr_group.info
            attributes = self.zarr_group.attrs.asdict()
        except Exception as e:
            self._add_error(
                self.group_uri, ValidationMessages.ZARR_ACCESS_ERROR.format(e)
            )
            return None

        # Extract OME metadata using version-specific logic
        ome_metadata = self._extract_ome_metadata(attributes)
        if not ome_metadata:
            return None

        return zarr_metadata, ome_metadata

    def _extract_ome_metadata(
        self, attributes: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract OME metadata from attributes. Override for version-specific logic."""
        ome_metadata = attributes.get("ome", {})
        if not ome_metadata:
            self._add_error(self.group_uri, ValidationMessages.MISSING_OME_METADATA)
            return None
        return ome_metadata

    def _validate_ome_structure(self, ome_metadata: dict[str, Any]) -> None:
        """Validate OME structure based on metadata type."""
        # Determine the type of OME group
        if "multiscales" in ome_metadata:
            if "image-label" in ome_metadata:
                self._validate_label_image(ome_metadata)
            else:
                self._validate_image_group(ome_metadata)
        elif "plate" in ome_metadata:
            self._validate_plate_group(ome_metadata)
        elif "well" in ome_metadata:
            self._validate_well_group(ome_metadata)
        elif "labels" in ome_metadata:
            self._validate_labels_group(ome_metadata)
        elif "series" in ome_metadata:
            self._validate_series_group(ome_metadata)
        elif "bioformats2raw.layout" in ome_metadata:
            self._validate_bioformats2raw_group(ome_metadata)
        else:
            self._add_warning(self.group_uri, "Unknown OME metadata type")

    def _validate_image_group(self, ome_metadata: dict[str, Any]) -> None:
        """Validate image group structure."""
        multiscales = ome_metadata["multiscales"]
        if not isinstance(multiscales, list) or not multiscales:
            self._add_error(self.group_uri, "multiscales must be a non-empty list")
            return

        for i, multiscale in enumerate(multiscales):
            self._validate_multiscale(multiscale, f"multiscales[{i}]")

    def _validate_multiscale(
        self, multiscale: dict[str, Any], path_prefix: str
    ) -> None:
        """Validate a single multiscale definition."""
        axes = self._validate_multiscale_axes(multiscale, path_prefix)
        datasets = self._validate_multiscale_datasets(multiscale, path_prefix)
        if not datasets:
            return

        self._validate_datasets_content(datasets, axes, path_prefix)

    def _validate_multiscale_axes(
        self, multiscale: dict[str, Any], path_prefix: str
    ) -> list[dict]:
        """Validate multiscale axes definition."""
        axes = multiscale.get("axes", [])
        if axes:
            # Check for unique axis names
            axis_names = [ax.get("name") for ax in axes if isinstance(ax, dict)]
            if len(axis_names) != len(set(axis_names)):
                self._add_error(
                    self.group_uri, ValidationMessages.AXES_UNIQUE.format(path_prefix)
                )
        return axes

    def _validate_multiscale_datasets(
        self, multiscale: dict[str, Any], path_prefix: str
    ) -> list[dict]:
        """Validate multiscale datasets definition."""
        datasets = multiscale.get("datasets", [])
        if not datasets:
            self._add_error(
                self.group_uri, ValidationMessages.DATASETS_REQUIRED.format(path_prefix)
            )
        return datasets

    def _validate_datasets_content(
        self, datasets: list[dict], axes: list[dict], path_prefix: str
    ) -> None:
        """Validate the content of each dataset."""
        for j, dataset in enumerate(datasets):
            if not isinstance(dataset, dict):
                continue
            self._validate_single_dataset(dataset, axes, f"{path_prefix}.datasets[{j}]")

    def _validate_single_dataset(
        self, dataset: dict[str, Any], axes: list[dict], dataset_prefix: str
    ) -> None:
        """Validate a single dataset within a multiscale."""
        dataset_path = self._extract_dataset_path(dataset, dataset_prefix)
        if not dataset_path:
            return

        dataset_item = self._get_dataset_item(dataset_path)
        if not dataset_item:
            return

        self._validate_dataset_is_array(dataset_item, dataset_path)
        self._validate_dataset_content(dataset, dataset_item, axes, dataset_prefix)

    def _extract_dataset_path(
        self, dataset: dict[str, Any], dataset_prefix: str
    ) -> str | None:
        """Extract and validate dataset path."""
        dataset_path = dataset.get("path")
        if not dataset_path:
            self._add_error(
                self.group_uri,
                ValidationMessages.DATASET_PATH_REQUIRED.format(
                    dataset_prefix.rsplit(".", 1)[0], dataset_prefix.split("[")[-1][:-1]
                ),
            )
            return None
        return dataset_path

    def _get_dataset_item(self, dataset_path: str):
        """Get dataset item with error handling."""
        return self._safe_zarr_access(dataset_path, "Dataset")

    def _validate_dataset_is_array(self, dataset_item, dataset_path: str) -> None:
        """Validate that dataset item is a zarr array."""
        if not hasattr(dataset_item, "shape"):
            self._add_error(
                f"{self.group_uri}/{dataset_path}",
                ValidationMessages.DATASET_NOT_ARRAY.format(dataset_path),
            )

    def _validate_dataset_content(
        self,
        dataset: dict[str, Any],
        dataset_item,
        axes: list[dict],
        dataset_prefix: str,
    ) -> None:
        """Validate dataset content and transformations."""
        dataset_path = dataset.get("path", "")
        array_path = f"{self.group_uri}/{dataset_path}"
        expected_ndim = len(axes) if axes else None

        self._validate_zarr_array(dataset_item, expected_ndim, array_path)

        # Validate coordinate transformations
        coord_transforms = dataset.get("coordinateTransformations", [])
        if axes:  # Only validate if axes are defined
            self._validate_coordinate_transformations(
                coord_transforms, len(axes), dataset_prefix
            )

    def _safe_zarr_access(self, path: str, operation_name: str) -> Any | None:
        """Safely access zarr group items with consistent error handling."""
        try:
            return self.zarr_group[path]
        except KeyError:
            self._add_error(
                f"{self.group_uri}/{path}",
                ValidationMessages.DATASET_NOT_EXIST.format(path),
            )
            return None
        except Exception as e:
            self._add_error(
                f"{self.group_uri}/{path}",
                ValidationMessages.DATASET_ACCESS_ERROR.format(e),
            )
            return None

    def _validate_zarr_array(
        self, zarr_array, expected_ndim: int | None, array_path: str
    ) -> None:
        """Validate a zarr array."""
        try:
            # Check shape matches expected dimensions if provided
            if expected_ndim is not None:
                shape = zarr_array.shape
                if len(shape) != expected_ndim:
                    self._add_warning(
                        array_path,
                        f"Array dimensions ({len(shape)}) don't match "
                        f"axes count ({expected_ndim})",
                    )

            # Check basic properties
            if hasattr(zarr_array, "info"):
                # The zarr array should be readable
                if not hasattr(zarr_array, "dtype"):
                    self._add_error(array_path, ValidationMessages.ARRAY_ACCESS_ERROR)

        except Exception as e:
            self._add_error(
                array_path, ValidationMessages.ARRAY_VALIDATION_ERROR.format(e)
            )

    def _validate_coordinate_transformations(
        self, transforms: list[Any], axes_count: int, path_prefix: str
    ) -> None:
        """Validate coordinate transformations."""
        has_scale = False

        for i, transform in enumerate(transforms):
            if not isinstance(transform, dict):
                continue

            transform_type = transform.get("type")
            if transform_type == "scale":
                has_scale = True
                scale = transform.get("scale", [])
                if len(scale) != axes_count:
                    self._add_error(
                        self.group_uri,
                        ValidationMessages.SCALE_LENGTH_MISMATCH.format(
                            path_prefix, i, len(scale), axes_count
                        ),
                    )
            elif transform_type == "translation":
                translation = transform.get("translation", [])
                if len(translation) != axes_count:
                    self._add_error(
                        self.group_uri,
                        ValidationMessages.TRANSLATION_LENGTH_MISMATCH.format(
                            path_prefix, i, len(translation), axes_count
                        ),
                    )
            elif transform_type not in ["scale", "translation"]:
                self._add_error(
                    self.group_uri,
                    ValidationMessages.UNKNOWN_TRANSFORM_TYPE.format(
                        path_prefix, i, transform_type
                    ),
                )

        if not has_scale:
            severity = self._get_coordinate_transforms_missing_scale_severity()
            message = ValidationMessages.SCALE_TRANSFORM_REQUIRED.format(path_prefix)
            if severity == "error":
                self._add_error(self.group_uri, message)
            else:
                self._add_warning(self.group_uri, message)

    def _validate_label_image(self, ome_metadata: dict[str, Any]) -> None:
        """Validate label image structure."""
        # First validate as regular image
        self._validate_image_group(ome_metadata)

        # Then validate label-specific requirements
        image_label = ome_metadata.get("image-label", {})
        if not image_label:
            self._add_error(
                self.group_uri, "image-label metadata is required for label images"
            )
            return

        # Check data types are integers
        multiscales = ome_metadata.get("multiscales", [])
        for multiscale in multiscales:
            datasets = multiscale.get("datasets", [])
            for dataset in datasets:
                dataset_path = dataset.get("path")
                if dataset_path:
                    try:
                        dataset_array = self.zarr_group[dataset_path]
                        array_path = f"{self.group_uri}/{dataset_path}"
                        self._validate_label_data_type(dataset_array, array_path)
                    except KeyError:
                        # Already reported by main dataset validation
                        pass
                    except Exception as e:
                        self._add_error(
                            f"{self.group_uri}/{dataset_path}",
                            f"Error accessing label dataset: {e}",
                        )

    def _validate_label_data_type(self, zarr_array, array_path: str) -> None:
        """Validate that label array uses integer data type."""
        try:
            data_type = zarr_array.dtype
            if data_type and data_type.kind not in ("i", "u"):  # integer types
                self._add_error(
                    array_path,
                    f"Label arrays must use integer data types, got '{data_type}'",
                )
        except Exception as e:
            self._add_error(array_path, f"Error checking label data type: {e}")

    def _validate_plate_group(self, ome_metadata: dict[str, Any]) -> None:
        """Validate plate group structure."""
        plate = ome_metadata.get("plate", {})
        if not plate:
            self._add_error(self.group_uri, "plate metadata is required")
            return

        # Validate required fields
        for field in ["columns", "rows", "wells"]:
            if field not in plate:
                self._add_error(self.group_uri, f"plate.{field} is required")

        # Validate columns and rows
        columns = plate.get("columns", [])
        rows = plate.get("rows", [])

        self._validate_plate_names(columns, "columns")
        self._validate_plate_names(rows, "rows")

        # Validate wells
        wells = plate.get("wells", [])
        self._validate_wells(wells, rows, columns)

    def _validate_plate_names(self, names: list[Any], field_name: str) -> None:
        """Validate plate row/column names."""
        seen_names = set()
        for i, name_obj in enumerate(names):
            if not isinstance(name_obj, dict):
                self._add_error(
                    self.group_uri, f"plate.{field_name}[{i}] must be an object"
                )
                continue

            name = name_obj.get("name")
            if not name:
                self._add_error(
                    self.group_uri, f"plate.{field_name}[{i}].name is required"
                )
                continue

            # Check alphanumeric
            if not str(name).replace("_", "").replace("-", "").isalnum():
                self._add_warning(
                    self.group_uri,
                    f"plate.{field_name}[{i}].name should be alphanumeric: '{name}'",
                )

            # Check for duplicates
            if name in seen_names:
                self._add_error(
                    self.group_uri,
                    f"plate.{field_name}[{i}].name is duplicate: '{name}'",
                )
            seen_names.add(name)

    def _validate_wells(
        self, wells: list[Any], rows: list[Any], columns: list[Any]
    ) -> None:
        """Validate wells structure and paths."""
        for i, well in enumerate(wells):
            if not isinstance(well, dict):
                continue
            self._validate_single_well(well, i, rows, columns)

    def _validate_single_well(
        self, well: dict, index: int, rows: list[Any], columns: list[Any]
    ) -> None:
        """Validate a single well configuration."""
        path = well.get("path")
        if path is None:
            self._add_error(self.group_uri, f"plate.wells[{index}].path is required")
            return

        well_group = self._access_well_group(path)
        if well_group is not None:
            self._validate_well_directory(well_group, f"{self.group_uri}/{path}")
            self._validate_well_indices(well, index, rows, columns)

    def _access_well_group(self, path: str):
        """Access a well group with proper error handling."""
        try:
            well_group = self.zarr_group[path]
            if not hasattr(well_group, "attrs"):
                self._add_error(
                    f"{self.group_uri}/{path}",
                    f"Well path is not a zarr group: {path}",
                )
                return None
            return well_group
        except KeyError:
            self._add_error(
                f"{self.group_uri}/{path}", f"Well path does not exist: {path}"
            )
            return None
        except Exception as e:
            self._add_error(f"{self.group_uri}/{path}", f"Error accessing well: {e}")
            return None

    def _validate_well_indices(
        self, well: dict, index: int, rows: list[Any], columns: list[Any]
    ) -> None:
        """Validate well row and column indices."""
        row_index = well.get("rowIndex")
        col_index = well.get("columnIndex")

        if isinstance(row_index, int) and isinstance(col_index, int):
            if not (0 <= row_index < len(rows)):
                self._add_error(
                    self.group_uri,
                    f"plate.wells[{index}].rowIndex out of range: {row_index}",
                )
            if not (0 <= col_index < len(columns)):
                self._add_error(
                    self.group_uri,
                    f"plate.wells[{index}].columnIndex out of range: {col_index}",
                )

    def _validate_well_directory(self, well_group, well_uri: str) -> None:
        """Validate a well group structure."""
        try:
            # Get well attributes
            attributes = well_group.attrs.asdict()
            # Handle both v0.4 (direct) and v0.5 (under 'ome' key) metadata formats
            ome_metadata = attributes.get("ome", attributes)
            well_info = ome_metadata.get("well", {})

            if not well_info:
                self._add_error(well_uri, "well metadata is required in well groups")
                return

            # Validate images
            images = well_info.get("images", [])
            if not images:
                self._add_error(
                    well_uri, "well.images is required and must be non-empty"
                )
                return

            for i, image in enumerate(images):
                if not isinstance(image, dict):
                    continue

                image_path = image.get("path")
                if not image_path:
                    self._add_error(well_uri, f"well.images[{i}].path is required")
                    continue

                # Check that image path exists in well group
                try:
                    image_group = well_group[image_path]
                    if not hasattr(image_group, "attrs"):
                        self._add_error(
                            f"{well_uri}/{image_path}",
                            f"Image path is not a zarr group: {image_path}",
                        )
                except KeyError:
                    self._add_error(
                        f"{well_uri}/{image_path}",
                        f"Image path does not exist: {image_path}",
                    )
                except Exception as e:
                    self._add_error(
                        f"{well_uri}/{image_path}", f"Error accessing image: {e}"
                    )

        except Exception as e:
            self._add_error(well_uri, f"Error validating well group: {e}")

    def _validate_well_group(self, ome_metadata: dict[str, Any]) -> None:
        """Validate well group structure (when validating a well directly)."""
        well = ome_metadata.get("well", {})
        if not well:
            self._add_error(self.group_uri, "well metadata is required")
            return

        # Validate images
        images = well.get("images", [])
        for image in images:
            if not isinstance(image, dict):
                continue

            image_path = image.get("path")
            if image_path:
                # Try to access and recursively validate the image
                try:
                    image_group = self.zarr_group[image_path]
                    if hasattr(image_group, "attrs"):
                        # Recursively validate the image
                        image_uri = f"{self.group_uri}/{image_path}"
                        image_validator = type(self)(image_uri, image_group)
                        image_result = image_validator.validate()
                        self.errors.extend(image_result.errors)
                        self.warnings.extend(image_result.warnings)
                except KeyError:
                    # Already reported in main validation
                    pass
                except Exception as e:
                    self._add_error(
                        f"{self.group_uri}/{image_path}",
                        f"Error recursively validating image: {e}",
                    )

    def _validate_labels_group(self, ome_metadata: dict[str, Any]) -> None:
        """Validate labels group structure."""
        labels = ome_metadata.get("labels", [])
        if not isinstance(labels, list):
            self._add_error(self.group_uri, "labels must be a list")
            return

        # Check that each label path exists
        for i, label_name in enumerate(labels):
            if not isinstance(label_name, str):
                self._add_error(self.group_uri, f"labels[{i}] must be a string")
                continue

            # Check that each label path exists in zarr group
            try:
                label_group = self.zarr_group[label_name]
                if not hasattr(label_group, "attrs"):
                    self._add_error(
                        f"{self.group_uri}/{label_name}",
                        f"Label path is not a zarr group: {label_name}",
                    )
                    continue
            except KeyError:
                self._add_error(
                    f"{self.group_uri}/{label_name}",
                    f"Label path does not exist: {label_name}",
                )
                continue
            except Exception as e:
                self._add_error(
                    f"{self.group_uri}/{label_name}", f"Error accessing label: {e}"
                )
                continue

            # Recursively validate the label image
            label_uri = f"{self.group_uri}/{label_name}"
            label_validator = type(self)(label_uri, label_group)
            label_result = label_validator.validate()
            self.errors.extend(label_result.errors)
            self.warnings.extend(label_result.warnings)

    def _validate_series_group(self, ome_metadata: dict[str, Any]) -> None:
        """Validate series group structure."""
        series = ome_metadata.get("series", [])
        if not isinstance(series, list) or not series:
            self._add_error(self.group_uri, "series must be a non-empty list")
            return

        # Check that each series path exists
        for i, series_path in enumerate(series):
            if not isinstance(series_path, str):
                self._add_error(self.group_uri, f"series[{i}] must be a string")
                continue

            # Check that each series path exists in zarr group
            try:
                series_group = self.zarr_group[series_path]
                if not hasattr(series_group, "attrs"):
                    self._add_error(
                        f"{self.group_uri}/{series_path}",
                        f"Series path is not a zarr group: {series_path}",
                    )
            except KeyError:
                self._add_error(
                    f"{self.group_uri}/{series_path}",
                    f"Series path does not exist: {series_path}",
                )
            except Exception as e:
                self._add_error(
                    f"{self.group_uri}/{series_path}", f"Error accessing series: {e}"
                )

    def _validate_bioformats2raw_group(self, ome_metadata: dict[str, Any]) -> None:
        """Validate bioformats2raw group structure."""
        layout = ome_metadata.get("bioformats2raw.layout")
        if layout is None:
            self._add_error(self.group_uri, "bioformats2raw.layout is required")
            return

        # Check for numbered directories (using zarr group discovery)
        numbered_dirs = []
        try:
            for key in self.zarr_group.keys():
                if key.isdigit() and hasattr(self.zarr_group[key], "attrs"):
                    numbered_dirs.append(int(key))
        except Exception as e:
            self._add_error(
                self.group_uri,
                f"Error exploring zarr group for numbered directories: {e}",
            )
            return

        if not numbered_dirs:
            self._add_warning(
                self.group_uri,
                "No numbered image directories found for bioformats2raw layout",
            )

        # Check for OME metadata directory (v0.4 specific check)
        self._validate_bioformats2raw_ome_directory()

    def _validate_bioformats2raw_ome_directory(self) -> None:
        """Validate OME metadata directory in bioformats2raw layout."""
        try:
            ome_group = self.zarr_group["OME"]
            # Note: zarr groups don't have files like METADATA.ome.xml
            # This would be stored as zarr arrays or attributes
            if "METADATA.ome.xml" not in ome_group.keys():
                self._add_warning(
                    f"{self.group_uri}/OME",
                    "METADATA.ome.xml data not found in OME group",
                )
        except KeyError:
            self._add_warning(f"{self.group_uri}/OME", "OME metadata group not found")
        except Exception as e:
            self._add_error(
                f"{self.group_uri}/OME", f"Error checking OME metadata: {e}"
            )

    def _validate_well_structure(
        self, wells: list[Any], rows: list[Any], columns: list[Any]
    ) -> None:
        """Validate wells structure and paths."""
        for i, well in enumerate(wells):
            if not isinstance(well, dict):
                continue
            self._validate_single_well_structure(well, i, rows, columns)

    def _validate_single_well_structure(
        self, well: dict, index: int, rows: list[Any], columns: list[Any]
    ) -> None:
        """Validate a single well configuration."""
        path = well.get("path")
        if path is None:
            self._add_error(self.group_uri, f"plate.wells[{index}].path is required")
            return

        well_group = self._access_well_group(path)
        if well_group is not None:
            self._validate_well_zarr_group(well_group, f"{self.group_uri}/{path}")
            self._validate_well_indices(well, index, rows, columns)

    def _validate_well_zarr_group(self, well_group, well_uri: str) -> None:
        """Validate a well zarr group structure."""
        try:
            attributes = well_group.attrs.asdict()
            # Handle both v0.4 (direct) and v0.5 (under 'ome' key) metadata formats
            ome_metadata = attributes.get("ome", attributes)
            well_info = ome_metadata.get("well", {})

            if not well_info:
                self._add_error(well_uri, "well metadata is required in well groups")
                return

            # Validate images
            images = well_info.get("images", [])
            if not images:
                self._add_error(
                    well_uri, "well.images is required and must be non-empty"
                )
                return

            for i, image in enumerate(images):
                if not isinstance(image, dict):
                    continue

                image_path = image.get("path")
                if not image_path:
                    self._add_error(well_uri, f"well.images[{i}].path is required")
                    continue

                # Check that image path exists in well group
                try:
                    image_group = well_group[image_path]
                    if not hasattr(image_group, "attrs"):
                        self._add_error(
                            f"{well_uri}/{image_path}",
                            f"Image path is not a zarr group: {image_path}",
                        )
                except KeyError:
                    self._add_error(
                        f"{well_uri}/{image_path}",
                        f"Image path does not exist: {image_path}",
                    )
                except Exception as e:
                    self._add_error(
                        f"{well_uri}/{image_path}", f"Error accessing image: {e}"
                    )

        except Exception as e:
            self._add_error(well_uri, f"Error validating well group: {e}")

    def _add_error(self, path: str, message: str) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(path, message, "error"))

    def _add_warning(self, path: str, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationError(path, message, "warning"))

    def _result(self) -> ValidationResult:
        """Create validation result."""
        return ValidationResult(
            valid=len(self.errors) == 0, errors=self.errors, warnings=self.warnings
        )

    @classmethod
    def validate_group_model(cls, model: ZarrGroupModel) -> ValidationResult:
        """Validate OME-ZARR storage structure using a specific validator.

        Parameters
        ----------
        model : ZarrGroupModel
            A model instance with a uri field pointing to the zarr group to validate.

        Returns
        -------
        ValidationResult
            Validation result with errors and warnings.

        Raises
        ------
        ImportError
            If zarr is not available.
        ValueError
            If the model doesn't have a valid uri.
        """
        if not ZARR_AVAILABLE:
            raise ImportError(
                "zarr package is required for storage validation. "
                "Install with: pip install zarr"
            )

        if not model.uri:
            raise ValueError("Model must have a uri field to validate storage")

        # Extract the base zarr group URI
        group_uri = cls._extract_group_uri(model.uri)

        try:
            # Try to open the zarr group using zarr-python
            # This will work for both local and remote (HTTP, S3, etc.) URLs
            zarr_group = zarr.open_group(group_uri, mode="r")
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[
                    ValidationError(
                        path=group_uri,
                        message=f"Cannot open zarr group: {e}",
                        severity="error",
                    )
                ],
                warnings=[],
            )

        validator = cls(group_uri, zarr_group)
        return validator.validate()
