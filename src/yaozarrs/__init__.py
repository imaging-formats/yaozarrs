"""Yet another ome-zarr model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("yaozarrs")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from . import v04, v05
from ._dim_spec import DimSpec
from ._storage import validate_zarr_store
from ._validate import validate_ome_json, validate_ome_object, validate_ome_uri
from ._zarr import ZarrGroup, open_group

__all__ = [
    "DimSpec",
    "ZarrGroup",
    "open_group",
    "v04",
    "v05",
    "validate_ome_json",
    "validate_ome_object",
    "validate_ome_uri",
    "validate_zarr_store",
]
