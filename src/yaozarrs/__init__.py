"""Yet another ome-zarr model."""

from typing import TYPE_CHECKING, Any

from . import v04, v05, v06
from ._convert import ConversionWarning, convert_version
from ._dim_spec import DimSpec
from ._errors import (
    StorageValidationError,
    StorageValidationWarning,
    ValidationWarning,
)
from ._validate import (
    validate_ome_json,
    validate_ome_object,
    validate_ome_uri,
    validate_zarr_store,
)
from ._zarr import ZarrArray, ZarrGroup, open_group

__all__ = [
    "ConversionWarning",
    "DimSpec",
    "StorageValidationError",
    "StorageValidationWarning",
    "ValidationWarning",
    "ZarrArray",
    "ZarrGroup",
    "convert_version",
    "open_group",
    "v04",
    "v05",
    "v06",
    "validate_ome_json",
    "validate_ome_object",
    "validate_ome_uri",
    "validate_zarr_store",
]

if TYPE_CHECKING:
    __version__: str
else:
    # (hidden from type checkers, or any misspelled attribute would type-check)
    def __getattr__(name: str) -> Any:
        if name == "__version__":
            from importlib.metadata import PackageNotFoundError, version

            try:
                return version("yaozarrs")
            except PackageNotFoundError:  # pragma: no cover
                return "uninstalled"
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
