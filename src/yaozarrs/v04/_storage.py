"""Storage validation for OME-ZARR v0.4 hierarchies.

This module provides functions to validate that OME-ZARR v0.4 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.  The traversal itself is shared
(`yaozarrs._common._storage.StorageValidator`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from yaozarrs._common._storage import StorageValidator
from yaozarrs._storage import StorageErrorType, ValidationResult
from yaozarrs._zarr import open_group
from yaozarrs.v04._bf2raw import Bf2Raw, Series
from yaozarrs.v04._image import Dataset, Image, Multiscale
from yaozarrs.v04._labels import LabelImage, LabelsGroup
from yaozarrs.v04._plate import Plate, Well
from yaozarrs.v04._zarr_json import OMEZarrGroupJSON

if TYPE_CHECKING:
    from collections.abc import Mapping

    from yaozarrs._common._storage import Loc
    from yaozarrs._zarr import ZarrArray

# Reusable TypeAdapter for validating v04 OME-ZARR metadata
_OME_VALIDATOR = TypeAdapter(OMEZarrGroupJSON)


class StorageValidatorV04(StorageValidator):
    """Concrete implementation of storage validator. for OME-ZARR v0.4 spec."""

    __slots__ = ()

    VERSION = "0.4"
    # In v04, metadata is at the top level (no "ome" wrapper)
    LOC_PREFIX = ()
    Image = Image
    LabelImage = LabelImage
    LabelsGroup = LabelsGroup
    Plate = Plate
    Well = Well
    Bf2Raw = Bf2Raw
    Series = Series

    def _parse_ome(self, attrs: Mapping[str, Any]) -> Any:
        # Convert mappingproxy to dict for discriminator
        return _OME_VALIDATOR.validate_python(dict(attrs))

    def _parse_ome_uri(self, uri: str) -> Any:
        # In v04, we open the zarr group and validate the attrs directly
        return self._parse_ome(open_group(uri).attrs)

    def _validate_dataset_array(
        self, arr: ZarrArray, dataset: Dataset, multiscale: Multiscale, ds_loc: Loc
    ) -> ValidationResult:
        result = ValidationResult()
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
