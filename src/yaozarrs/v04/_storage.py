"""Storage validation for OME-ZARR v0.4 hierarchies.

This module provides functions to validate that OME-ZARR v0.4 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.

The shared logic lives in [`yaozarrs._storage_base.BaseStorageValidator`][]. The
only v0.4 differences are *where* the metadata lives: v0.4 (Zarr v2) stores the
OME metadata directly at the top level of the group attributes (no `"ome"` key).
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from yaozarrs._storage_base import BaseStorageValidator
from yaozarrs._zarr import ZarrGroup, open_group
from yaozarrs.v04._bf2raw import Bf2Raw, Series
from yaozarrs.v04._image import Image
from yaozarrs.v04._labels import LabelImage, LabelsGroup
from yaozarrs.v04._plate import Plate, Well
from yaozarrs.v04._zarr_json import OMEZarrGroupJSON

__all__ = ["StorageValidatorV04"]

# Reusable TypeAdapter for validating v04 OME-ZARR metadata
_OME_VALIDATOR = TypeAdapter(OMEZarrGroupJSON)


class StorageValidatorV04(BaseStorageValidator):
    """Concrete implementation of storage validator for OME-ZARR v0.4 spec."""

    __slots__ = ()

    OME_VERSION = "0.4"
    # In v04, metadata is at the top level (no "ome" wrapper)
    ROOT_LOC = ()
    Image = Image
    LabelImage = LabelImage
    LabelsGroup = LabelsGroup
    Plate = Plate
    Well = Well
    Bf2Raw = Bf2Raw
    Series = Series
    OMEZarrGroupJSON = OMEZarrGroupJSON

    @classmethod
    def _root_metadata(cls, zarr_group: ZarrGroup, given: Any | None) -> Any:
        # In v04 the (optional) given metadata IS the OME object (no .ome accessor)
        if given is None:
            # Convert mappingproxy to dict for discriminator
            given = _OME_VALIDATOR.validate_python(dict(zarr_group.attrs))
        return given

    def _parse_attrs(self, attrs: Any) -> Any:
        return _OME_VALIDATOR.validate_python(dict(attrs))

    def _load_source_image(self, uri: str) -> Any:
        # In v04, we open the zarr group and validate the attrs directly
        return _OME_VALIDATOR.validate_python(dict(open_group(uri).attrs))
