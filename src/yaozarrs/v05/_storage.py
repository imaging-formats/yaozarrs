"""Storage validation for OME-ZARR v0.5 hierarchies.

This module provides functions to validate that OME-ZARR v0.5 storage structures
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
    return _StorageValidator.validate_group_model(model)


class _StorageValidator(BaseStorageValidator):
    """Internal validator for OME-ZARR v0.5 storage."""

    @staticmethod
    def _extract_group_uri(uri: str) -> str:
        """Extract the base zarr group URI (remove zarr.json suffix if present)."""
        group_uri = uri
        if group_uri.endswith("/zarr.json"):
            group_uri = group_uri[: -len("/zarr.json")]
        elif group_uri.endswith("\\zarr.json"):  # Windows paths
            group_uri = group_uri[: -len("\\zarr.json")]
        return group_uri

    def _validate_zarr_format(self, zarr_metadata) -> None:
        """Validate zarr format requirements for v0.5 (allows format 2 or 3)."""
        if hasattr(zarr_metadata, "_zarr_format"):
            zarr_format = zarr_metadata._zarr_format
            if zarr_format not in (2, 3):
                self._add_error(
                    self.group_uri,
                    f"zarr_format must be 2 or 3 for OME-ZARR v0.5, got {zarr_format}",
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

    def _validate_version_metadata(self, metadata: dict[str, Any]) -> None:
        """Validate version-specific metadata requirements for v0.5."""
        # v0.5 expects OME metadata under 'ome' key
        ome_metadata = metadata.get("ome", {})
        if not ome_metadata:
            self._add_error(self.group_uri, "Missing 'ome' metadata in attributes")
            return

        # Check OME version
        version = ome_metadata.get("version")
        if version != "0.5":
            self._add_error(
                self.group_uri, f"OME version must be '0.5', got '{version}'"
            )

    def _get_coordinate_transforms_missing_scale_severity(self) -> str:
        """Get severity level for missing scale transformations (error for v0.5)."""
        return "error"

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

            # Check zarr format
            self._validate_zarr_format(zarr_metadata)

            # Validate version-specific metadata
            self._validate_version_metadata(attributes)

            # Get OME metadata for structure validation
            ome_metadata = attributes.get("ome", {})
            if ome_metadata:
                # Validate based on OME metadata type
                self._validate_ome_structure(ome_metadata)

            return self._result()

        except Exception as e:
            self._add_error(self.group_uri, f"Unexpected error during validation: {e}")
            return self._result()

    def _validate_multiscale(
        self, multiscale: dict[str, Any], path_prefix: str
    ) -> None:
        """Validate a single multiscale definition with v0.5 specific logic."""
        # Validate axes (required in v0.5)
        axes = multiscale.get("axes", [])
        if not axes:
            self._add_error(self.group_uri, f"{path_prefix}: axes are required")
            return

        # Call parent implementation for common validation
        super()._validate_multiscale(multiscale, path_prefix)
