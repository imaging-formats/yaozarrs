"""Yet another ome-zarr model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("yaozarrs")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from . import v04, v05
from ._validate import from_uri, validate_ome_json, validate_ome_object

__all__ = [
    "from_uri",
    "v04",
    "v05",
    "validate_ome_json",
    "validate_ome_object",
]
