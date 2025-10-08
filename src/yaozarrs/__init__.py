"""Yet another ome-zarr model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("yaozarrs")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from . import v04, v05
from ._storage import validate_zarr_store
from ._validate import from_uri, validate_ome_json, validate_ome_object
from ._zarr import open_group

__all__ = [
    "from_uri",
    "open_group",
    "v04",
    "v05",
    "validate_ome_json",
    "validate_ome_object",
    "validate_zarr_store",
]
