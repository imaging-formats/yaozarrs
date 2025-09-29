"""Storage validation for OME-ZARR v0.4 hierarchies.

This module provides functions to validate that OME-ZARR v0.4 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.

Requires zarr to be installed for full validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yaozarrs._storage_common import BaseStorageValidator, ValidationResult

if TYPE_CHECKING:
    from yaozarrs._base import ZarrGroupModel


def validate_storage(model: ZarrGroupModel) -> ValidationResult:
    """Validate OME-ZARR v0.4 storage structure.

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
    return _StorageValidator.validate_group_model(model)


class _StorageValidator(BaseStorageValidator):
    """Internal validator for OME-ZARR v0.4 storage."""

    @staticmethod
    def _extract_group_uri(uri: str) -> str:
        """Extract the base zarr group URI (remove .zattrs suffix if present)."""
        group_uri = uri
        if group_uri.endswith("/.zattrs"):
            group_uri = group_uri[: -len("/.zattrs")]
        elif group_uri.endswith("\\.zattrs"):  # Windows paths
            group_uri = group_uri[: -len("\\.zattrs")]
        return group_uri

    def _validate_zarr_format(self, zarr_metadata) -> None:
        """Validate zarr format requirements for v0.4 (must be format 2)."""
        if hasattr(zarr_metadata, "_zarr_format") and zarr_metadata._zarr_format != 2:
            self._add_error(
                self.group_uri,
                f"zarr_format must be 2 for OME-ZARR v0.4, "
                f"got {zarr_metadata._zarr_format}",
            )

    def _extract_ome_metadata(
        self, attributes: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract OME metadata from attributes for v0.4 (direct in attributes)."""
        # v0.4 stores OME metadata directly in the attributes, not under 'ome' key
        if not attributes:
            self._add_error(
                self.group_uri, "Empty OME metadata in zarr group attributes"
            )
            return None
        return attributes

    def _validate_version_metadata(self, metadata: dict[str, Any]) -> None:
        """Validate version-specific metadata requirements for v0.4."""
        # v0.4 expects OME metadata directly in attributes, not under 'ome' key
        # This validation handles the direct metadata format
        if not metadata:
            self._add_error(
                self.group_uri, "Empty OME metadata in zarr group attributes"
            )

    def _get_coordinate_transforms_missing_scale_severity(self) -> str:
        """Get severity level for missing scale transformations (warning for v0.4)."""
        return "warning"

    def _validate_multiscale(
        self, multiscale: dict[str, Any], path_prefix: str
    ) -> None:
        """Validate a single multiscale definition with v0.4 specific logic."""
        # Check version
        version = multiscale.get("version")
        if version and version != "0.4":
            self._add_warning(
                self.group_uri,
                f"{path_prefix}: version should be '0.4', got '{version}'",
            )

        # Call parent implementation for common validation
        super()._validate_multiscale(multiscale, path_prefix)

    def _validate_label_image(self, ome_metadata: dict[str, Any]) -> None:
        """Validate label image structure with v0.4 specific logic."""
        # First validate as regular image
        self._validate_image_group(ome_metadata)

        # Then validate label-specific requirements
        image_label = ome_metadata.get("image-label", {})
        if not image_label:
            self._add_error(
                self.group_uri, "image-label metadata is required for label images"
            )
            return

        # Check version
        version = image_label.get("version")
        if version and version != "0.4":
            self._add_warning(
                self.group_uri, f"image-label version should be '0.4', got '{version}'"
            )

        # Check data types are integers
        multiscales = ome_metadata.get("multiscales", [])
        for multiscale in multiscales:
            datasets = multiscale.get("datasets", [])
            for dataset in datasets:
                dataset_path = dataset.get("path")
                if dataset_path:
                    try:
                        dataset_item = self.zarr_group[dataset_path]
                        array_path = f"{self.group_uri}/{dataset_path}"
                        self._validate_label_data_type_v04(dataset_item, array_path)
                    except KeyError:
                        # Already reported by main dataset validation
                        pass
                    except Exception as e:
                        self._add_error(
                            f"{self.group_uri}/{dataset_path}",
                            f"Error accessing label dataset: {e}",
                        )

    def _validate_label_data_type_v04(self, dataset_item, array_path: str) -> None:
        """Validate that label array uses integer data type (v0.4 specific check)."""
        try:
            # Check if it's an array and get its dtype
            if hasattr(dataset_item, "dtype"):
                dtype_str = str(dataset_item.dtype)
                # Check if it's an integer type
                if not any(dtype_str.startswith(prefix) for prefix in ["int", "uint"]):
                    self._add_error(
                        array_path,
                        f"Label arrays must use integer data types, got '{dtype_str}'",
                    )
        except Exception as e:
            self._add_error(array_path, f"Error checking label data type: {e}")

    def _validate_plate_group(self, ome_metadata: dict[str, Any]) -> None:
        """Validate plate group structure with v0.4 specific logic."""
        plate = ome_metadata.get("plate", {})
        if not plate:
            self._add_error(self.group_uri, "plate metadata is required")
            return

        # Check version
        version = plate.get("version")
        if version and version != "0.4":
            self._add_warning(
                self.group_uri, f"plate version should be '0.4', got '{version}'"
            )

        # Call parent implementation for common validation
        super()._validate_plate_group(ome_metadata)

    def _validate_well_group(self, ome_metadata: dict[str, Any]) -> None:
        """Validate well group structure with v0.4 specific logic."""
        well = ome_metadata.get("well", {})
        if not well:
            self._add_error(self.group_uri, "well metadata is required")
            return

        # Check version
        version = well.get("version")
        if version and version != "0.4":
            self._add_warning(
                self.group_uri, f"well version should be '0.4', got '{version}'"
            )

        # Call parent implementation for common validation
        super()._validate_well_group(ome_metadata)

    def _validate_well_zarr_group(self, well_group, well_path: str) -> None:
        """Validate a well zarr group structure with v0.4 specific logic."""
        well_uri = f"{self.group_uri}/{well_path}"

        # Use parent validation for common structure
        super()._validate_well_zarr_group(well_group, well_uri)

        # Add v0.4 specific version validation
        try:
            well_metadata = well_group.attrs.asdict()
            ome_metadata = well_metadata.get("ome", {})
            well_info = ome_metadata.get("well", {})

            if well_info:
                version = well_info.get("version")
                if version and version != "0.4":
                    self._add_warning(
                        well_uri, f"well version should be '0.4', got '{version}'"
                    )
        except Exception:
            # Error already handled by parent method
            pass
