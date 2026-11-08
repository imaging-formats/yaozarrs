"""Storage validation for OME-ZARR v0.5 hierarchies.

This module provides functions to validate that OME-ZARR v0.5 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.  The traversal itself is shared
(`yaozarrs._common._storage.StorageValidator`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yaozarrs._common._storage import VectorsAtPathMixin
from yaozarrs._errors import StorageErrorType, ValidationResult
from yaozarrs._validate import validate_ome_object, validate_ome_uri
from yaozarrs.v05._bf2raw import Bf2Raw, Series
from yaozarrs.v05._image import Dataset, Image, Multiscale
from yaozarrs.v05._labels import LabelImage, LabelsGroup
from yaozarrs.v05._plate import Plate, Well
from yaozarrs.v05._zarr_json import OMEAttributes, OMEZarrGroupJSON

if TYPE_CHECKING:
    from collections.abc import Mapping

    from yaozarrs._common._storage import Loc
    from yaozarrs._zarr import ZarrArray, ZarrGroup


class StorageValidatorV05(VectorsAtPathMixin):
    """Storage validator for the OME-Zarr v0.5 spec."""

    __slots__ = ()

    VERSION = "0.5"
    Image = Image
    LabelImage = LabelImage
    LabelsGroup = LabelsGroup
    Plate = Plate
    Well = Well
    Bf2Raw = Bf2Raw
    Series = Series

    def _visit_unmarked_label_image(
        self, zarr_group: ZarrGroup, image_model: Any, loc_prefix: Loc
    ) -> ValidationResult:
        # The `labels` value is "a JSON array of paths to the labeled multiscale
        # image(s)", and "the pixels of the label images MUST be integer data types",
        # whether or not they have the (SHOULD) `image-label` metadata.
        result = self.visit_image(zarr_group, image_model, loc_prefix)
        return result.merge(
            self._validate_label_data_types(image_model, zarr_group, loc_prefix)
        )

    def _parse_ome(self, attrs: Mapping[str, Any]) -> Any:
        return validate_ome_object(dict(attrs), OMEAttributes, strict=self.strict).ome

    def _parse_ome_uri(self, uri: str) -> Any:
        return validate_ome_uri(
            uri, OMEZarrGroupJSON, strict=self.strict
        ).attributes.ome

    def _validate_dataset_array(
        self, arr: ZarrArray, dataset: Dataset, multiscale: Multiscale, ds_loc: Loc
    ) -> ValidationResult:
        result = ValidationResult()
        # dimension_names (top-level in zarr.json) must be present and match axes
        expected_names = [ax.name for ax in multiscale.axes]
        dim_names = arr.dimension_names
        if dim_names is None:
            # spec says MUST, but zarr-python omits it by default: warn only
            result.add_warning(
                StorageErrorType.dimension_names_missing,
                (*ds_loc, "dimension_names"),
                f"Array '{dataset.path}' has no dimension_names "
                f"(expected {expected_names})",
                ctx={"expected": expected_names},
            )
        elif dim_names != expected_names:
            result.add_error(
                StorageErrorType.dimension_names_mismatch,
                (*ds_loc, "dimension_names"),
                f"Array dimension_names {dim_names} don't match "
                f"axes names {expected_names}",
                ctx={"actual": dim_names, "expected": expected_names},
            )
        return result
