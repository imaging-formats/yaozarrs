from . import v04, v05
from ._base import _BaseModel


class OMEZarr(_BaseModel):
    """Model with *any* valid OME-Zarr structure."""

    ome: v05.OMENode | v04.OMENode
