"""Storage validation for OME-ZARR v0.5 hierarchies.

This module provides functions to validate that OME-ZARR v0.5 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.

The shared logic lives in [`yaozarrs._storage_base.BaseStorageValidator`][]; v0.5
uses its default behavior, so this subclass only supplies the v0.5 models.
"""

from __future__ import annotations

from yaozarrs._storage_base import BaseStorageValidator
from yaozarrs.v05._bf2raw import Bf2Raw, Series
from yaozarrs.v05._image import Image
from yaozarrs.v05._labels import LabelImage, LabelsGroup
from yaozarrs.v05._plate import Plate, Well
from yaozarrs.v05._zarr_json import OMEAttributes, OMEZarrGroupJSON

__all__ = ["StorageValidatorV05"]


class StorageValidatorV05(BaseStorageValidator):
    """Concrete implementation of storage validator for OME-ZARR v0.5 spec."""

    __slots__ = ()

    OME_VERSION = "0.5"
    Image = Image
    LabelImage = LabelImage
    LabelsGroup = LabelsGroup
    Plate = Plate
    Well = Well
    Bf2Raw = Bf2Raw
    Series = Series
    OMEAttributes = OMEAttributes
    OMEZarrGroupJSON = OMEZarrGroupJSON
