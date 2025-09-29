"""Storage validation for OME-ZARR v0.5 hierarchies.

This module provides functions to validate that OME-ZARR v0.5 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.

Requires zarr to be installed for full validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from yaozarrs._base import ZarrGroupModel

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


def validate_storage(model: ZarrGroupModel) -> ValidationResult:
    """Validate OME-ZARR v0.5 storage structure.

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

    # Extract the base zarr group URI (remove zarr.json suffix if present)
    group_uri = model.uri
    if group_uri.endswith("/zarr.json"):
        group_uri = group_uri[: -len("/zarr.json")]
    elif group_uri.endswith("\\zarr.json"):  # Windows paths
        group_uri = group_uri[: -len("\\zarr.json")]

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

    validator = _StorageValidator(group_uri, zarr_group)
    return validator.validate()


class _StorageValidator:
    """Internal validator for OME-ZARR v0.5 storage."""

    def __init__(self, group_uri: str, zarr_group):
        self.group_uri = group_uri
        self.zarr_group = zarr_group
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []

    def validate(self) -> ValidationResult:
        """Run complete validation."""
        try:
            # Get zarr group metadata
            try:
                zarr_metadata = self.zarr_group.info
                attributes = self.zarr_group.attrs.asdict()
            except Exception as e:
                self._add_error(
                    self.group_uri, f"Cannot access zarr group metadata: {e}"
                )
                return self._result()

            # Check zarr format - v0.5 prefers zarr v3 but can work with v2
            if hasattr(zarr_metadata, "_zarr_format"):
                zarr_format = zarr_metadata._zarr_format
                if zarr_format not in (2, 3):
                    self._add_error(
                        self.group_uri,
                        f"zarr_format must be 2 or 3 for OME-ZARR v0.5, "
                        f"got {zarr_format}",
                    )
                elif zarr_format == 2:
                    # Import zarr to check version
                    import zarr as zarr_module

                    zarr_major_version = int(zarr_module.__version__.split(".")[0])
                    if zarr_major_version >= 3:
                        self._add_warning(
                            self.group_uri,
                            "OME-ZARR v0.5 should use zarr v3 format "
                            "when zarr v3 is available",
                        )

            # Get OME metadata
            ome_metadata = attributes.get("ome", {})

            if not ome_metadata:
                self._add_error(self.group_uri, "Missing 'ome' metadata in attributes")
                return self._result()

            # Check OME version
            version = ome_metadata.get("version")
            if version != "0.5":
                self._add_error(
                    self.group_uri, f"OME version must be '0.5', got '{version}'"
                )

            # Validate based on OME metadata type
            self._validate_ome_structure(ome_metadata)

            return self._result()

        except Exception as e:
            self._add_error(self.group_uri, f"Unexpected error during validation: {e}")
            return self._result()

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
        # Validate axes
        axes = multiscale.get("axes", [])
        if not axes:
            self._add_error(self.group_uri, f"{path_prefix}: axes are required")
            return

        # Check for unique axis names
        axis_names = [ax.get("name") for ax in axes if isinstance(ax, dict)]
        if len(axis_names) != len(set(axis_names)):
            self._add_error(self.group_uri, f"{path_prefix}: axis names must be unique")

        # Validate datasets
        datasets = multiscale.get("datasets", [])
        if not datasets:
            self._add_error(self.group_uri, f"{path_prefix}: datasets are required")
            return

        # Check that dataset paths exist
        for j, dataset in enumerate(datasets):
            if not isinstance(dataset, dict):
                continue

            dataset_path = dataset.get("path")
            if not dataset_path:
                self._add_error(
                    self.group_uri, f"{path_prefix}.datasets[{j}]: path is required"
                )
                continue

            # Check if dataset exists in zarr group
            try:
                dataset_item = self.zarr_group[dataset_path]
                # Check if it's an array
                if not hasattr(dataset_item, "shape"):
                    self._add_error(
                        f"{self.group_uri}/{dataset_path}",
                        f"Dataset path is not a zarr array: {dataset_path}",
                    )
                    continue
            except KeyError:
                self._add_error(
                    f"{self.group_uri}/{dataset_path}",
                    f"Dataset path does not exist: {dataset_path}",
                )
                continue
            except Exception as e:
                self._add_error(
                    f"{self.group_uri}/{dataset_path}", f"Error accessing dataset: {e}"
                )
                continue

            # Validate that it's a proper zarr array
            array_path = f"{self.group_uri}/{dataset_path}"
            self._validate_zarr_array(dataset_item, len(axes), array_path)

            # Validate coordinate transformations
            coord_transforms = dataset.get("coordinateTransformations", [])
            self._validate_coordinate_transformations(
                coord_transforms, len(axes), f"{path_prefix}.datasets[{j}]"
            )

    def _validate_zarr_array(
        self, zarr_array, expected_ndim: int, array_path: str
    ) -> None:
        """Validate a zarr array."""
        try:
            # Check shape matches expected dimensions
            shape = zarr_array.shape
            if len(shape) != expected_ndim:
                self._add_warning(
                    array_path,
                    f"Array dimensions ({len(shape)}) don't match "
                    f"axes count ({expected_ndim})",
                )

            # For zarr v3, we can check some basic properties
            if hasattr(zarr_array, "info"):
                # The zarr array should be readable
                if not hasattr(zarr_array, "dtype"):
                    self._add_error(array_path, "Zarr array has no accessible dtype")

        except Exception as e:
            self._add_error(array_path, f"Error validating zarr array: {e}")

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
                        f"{path_prefix}.coordinateTransformations[{i}]: "
                        f"scale length ({len(scale)}) must match "
                        f"axes count ({axes_count})",
                    )
            elif transform_type == "translation":
                translation = transform.get("translation", [])
                if len(translation) != axes_count:
                    self._add_error(
                        self.group_uri,
                        f"{path_prefix}.coordinateTransformations[{i}]: "
                        f"translation length ({len(translation)}) must match "
                        f"axes count ({axes_count})",
                    )
            elif transform_type not in ["scale", "translation"]:
                self._add_error(
                    self.group_uri,
                    f"{path_prefix}.coordinateTransformations[{i}]: "
                    f"unknown transformation type '{transform_type}'",
                )

        if not has_scale:
            self._add_error(
                self.group_uri,
                f"{path_prefix}: at least one 'scale' transformation is required",
            )

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
        {r.get("name") for r in rows if isinstance(r, dict)}
        {c.get("name") for c in columns if isinstance(c, dict)}

        for i, well in enumerate(wells):
            if not isinstance(well, dict):
                continue

            path = well.get("path")
            row_index = well.get("rowIndex")
            col_index = well.get("columnIndex")

            if path is None:
                self._add_error(self.group_uri, f"plate.wells[{i}].path is required")
                continue

            # Check that well path exists in zarr group
            try:
                well_group = self.zarr_group[path]
                if not hasattr(well_group, "attrs"):
                    self._add_error(
                        f"{self.group_uri}/{path}",
                        f"Well path is not a zarr group: {path}",
                    )
                    continue
            except KeyError:
                self._add_error(
                    f"{self.group_uri}/{path}", f"Well path does not exist: {path}"
                )
                continue
            except Exception as e:
                self._add_error(
                    f"{self.group_uri}/{path}", f"Error accessing well: {e}"
                )
                continue

            # Validate that it's a proper well group
            self._validate_well_directory(well_group, f"{self.group_uri}/{path}")

            # Validate row/column indices
            if isinstance(row_index, int) and isinstance(col_index, int):
                if not (0 <= row_index < len(rows)):
                    self._add_error(
                        self.group_uri,
                        f"plate.wells[{i}].rowIndex out of range: {row_index}",
                    )
                if not (0 <= col_index < len(columns)):
                    self._add_error(
                        self.group_uri,
                        f"plate.wells[{i}].columnIndex out of range: {col_index}",
                    )

    def _validate_well_directory(self, well_group, well_uri: str) -> None:
        """Validate a well group structure."""
        try:
            # Get well attributes
            attributes = well_group.attrs.asdict()
            ome_metadata = attributes.get("ome", {})
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

        # This is handled by _validate_well_directory
        images = well.get("images", [])
        for _i, image in enumerate(images):
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
                        image_validator = _StorageValidator(image_uri, image_group)
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
            label_validator = _StorageValidator(label_uri, label_group)
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
