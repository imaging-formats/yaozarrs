"""Minimal zarr v2/v3 support for reading metadata and structure.

This implementation matches zarr-python's behavior: a zarr group expects
all children to be the same zarr_format version. Mixed v2/v3 hierarchies
are not supported.

Array data access is only supported via conversion to tensorstore or zarr-python.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    import fsspec
    import tensorstore  # type: ignore
    import zarr  # type: ignore


@dataclass
class ZarrMetadata:
    """Metadata from a zarr metadata file."""

    zarr_format: Literal[2, 3]
    node_type: str  # "group" or "array"
    attributes: dict[str, Any]
    shape: tuple[int, ...] | None = None
    data_type: str | None = None


class ZarrAttributes:
    """Dict-like wrapper for zarr attributes.

    ... to match the zarr-python API.
    """

    def __init__(self, attributes: dict[str, Any]) -> None:
        self._attributes = MappingProxyType(attributes)

    def asdict(self) -> Mapping[str, Any]:
        """Return attributes as a dictionary."""
        return self._attributes


class ZarrNode:
    """Base class for zarr nodes (groups and arrays)."""

    __slots__ = ("_mapper", "_metadata", "_path")

    @classmethod
    def from_uri(cls, uri: str | os.PathLike) -> Self:
        """Create a ZarrNode from a URI."""
        from fsspec import get_mapper

        if isinstance(uri, (str, os.PathLike)):
            uri = os.path.expanduser(os.fspath(uri))
        elif hasattr(uri, "store"):
            uri = str(uri.store)
        else:
            raise TypeError(
                "uri must be a string, os.PathLike, or have a 'store' attribute"
            )
        mapper = get_mapper(uri)
        return cls(mapper)

    def __init__(
        self,
        mapper: MutableMapping[str, bytes],
        path: str = "",
        meta: dict[str, Any] | ZarrMetadata | None = None,
    ) -> None:
        """Initialize a zarr node.

        Parameters
        ----------
        mapper : MutableMapping[str, bytes]
            The fsspec mapper for the zarr store.
        path : str
            The path within the zarr store.
        meta : dict[str, Any] | None
            Optional pre-loaded metadata dictionary. If not provided, it will be
            loaded from the store.
        """
        self._mapper = mapper
        self._path = path.rstrip("/")
        if meta is None:
            self._metadata = self._load_metadata()
        elif isinstance(meta, ZarrMetadata):
            if meta.node_type != self.node_type():
                raise ValueError(
                    f"Metadata node_type '{meta.node_type}' does not match "
                    f"expected '{self.node_type()}'"
                )
            self._metadata = meta
        else:
            if "zarr_format" not in meta:
                raise ValueError("Metadata missing 'zarr_format'")
            if meta.get("node_type") != self.node_type():
                raise ValueError(
                    f"Metadata node_type '{meta.get('node_type')}' does not match "
                    f"expected '{self.node_type()}'"
                )
            self._metadata = ZarrMetadata(
                zarr_format=meta["zarr_format"],
                node_type=self.node_type(),
                attributes=meta.get("attributes") or {},
                shape=tuple(meta["shape"]) if "shape" in meta else None,
                data_type=meta.get("data_type"),
            )

    @property
    def attrs(self) -> ZarrAttributes:
        """Return attributes as a read-only mapping."""
        return ZarrAttributes(self._metadata.attributes)

    @property
    def path(self) -> str:
        """Return the path of this node."""
        return self._path

    @property
    def zarr_format(self) -> Literal[2, 3]:
        """Return the zarr format version (2 or 3)."""
        return self._metadata.zarr_format

    @classmethod
    def node_type(cls) -> Literal["group", "array"]:
        """Return the node type (group or array)."""
        raise NotImplementedError("Cannot instantiate base ZarrNode")

    def _load_metadata(self) -> ZarrMetadata:
        """Load and parse zarr metadata (v2 or v3)."""
        prefix = f"{self._path}/" if self._path else ""
        parsers = {
            "zarr.json": self._parse_zarr_json_data,
            ".zgroup": self._parse_zgroup_data,
            ".zarray": self._parse_zarray_data,
        }

        for fname, parser in parsers.items():
            if json_data := self._mapper.get(f"{prefix}{fname}".lstrip("/")):
                meta = json.loads(json_data.decode("utf-8"))
                return parser(meta, prefix)

        raise FileNotFoundError(
            f"No zarr metadata found at '{self._path}' "
            f"(tried {', '.join(parsers.keys())})"
        )

    def _parse_zarr_json_data(self, meta: dict[str, Any], prefix: str) -> ZarrMetadata:
        """Parse metadata from zarr.json file (v3+ format)."""
        return ZarrMetadata(
            zarr_format=meta.get("zarr_format", 3),
            node_type=meta["node_type"],
            attributes=meta.get("attributes", {}),
            shape=tuple(meta["shape"]) if "shape" in meta else None,
            data_type=meta.get("data_type"),
        )

    def _parse_zgroup_data(self, meta: dict[str, Any], prefix: str) -> ZarrMetadata:
        zarr_format = meta.get("zarr_format", 2)
        if zarr_format != 2:
            raise ValueError(f"Expected zarr_format 2 in .zgroup, got {zarr_format}")
        return ZarrMetadata(
            zarr_format=2,
            node_type="group",
            attributes=self._load_v2_attrs(prefix),
            shape=None,
            data_type=None,
        )

    def _parse_zarray_data(self, meta: dict[str, Any], prefix: str) -> ZarrMetadata:
        zarr_format = meta.get("zarr_format", 2)
        if zarr_format != 2:
            raise ValueError(f"Expected zarr_format 2 in .zarray, got {zarr_format}")
        dtype_str = str(np.dtype(meta["dtype"])) if "dtype" in meta else None
        return ZarrMetadata(
            zarr_format=2,
            node_type="array",
            attributes=self._load_v2_attrs(prefix),
            shape=tuple(meta["shape"]) if "shape" in meta else None,
            data_type=dtype_str,
        )

    def _load_v2_attrs(self, prefix: str) -> dict[str, Any]:
        """Load v2 .zattrs file if present."""
        if zattrs_data := self._mapper.get(f"{prefix}.zattrs".lstrip("/")):
            return json.loads(zattrs_data.decode("utf-8"))
        return {}

    def to_zarr_python(self) -> zarr.Array | zarr.Group:
        """Convert to a zarr-python Array or Group object."""
        try:
            import zarr  # type: ignore
        except ImportError as e:
            raise ImportError("zarr package is required for to_zarr_python()") from e

        return zarr.open(self.get_uri(), mode="r")

    def get_uri(self) -> str:
        """Get the URI for this zarr node.

        Returns a URI string in the standard format protocol://path,
        such as:
        - "file:///Users/user/data/array.zarr"
        - "https://example.com/data/array.zarr"
        - "s3://bucket/path/array.zarr"
        - "memory://"

        This URI can be used with various tools including TensorStore,
        fsspec, and other libraries that understand standard URI schemes.

        Returns
        -------
        str
            A URI string that follows standard protocol://path format.

        Raises
        ------
        ValueError
            If the URI cannot be determined from the mapper.
        """
        # Check if we have an FSMap with filesystem info
        if (fs := sys.modules.get("fsspec")) and isinstance(self._mapper, fs.FSMap):  # type: ignore
            mapper = cast("fsspec.FSMap", self._mapper)

            # Build the full path including our internal zarr path
            if self._path:
                full_path = f"{mapper.root.rstrip('/')}/{self._path}"
            else:
                full_path = mapper.root

            return mapper.fs.unstrip_protocol(full_path)

        raise NotImplementedError("Only fsspec mappers are supported for get_uri()")


class ZarrGroup(ZarrNode):
    """Wrapper around a zarr v2/v3 group.

    Matches zarr-python behavior: expects all children to be the same
    zarr_format version as the parent. Does not support mixed hierarchies.
    """

    __slots__ = ()

    @classmethod
    def node_type(cls) -> Literal["group"]:
        """Return the node type (group or array)."""
        return "group"

    def __contains__(self, key: str) -> bool:
        """Check if a child node exists."""
        child_path = f"{self._path}/{key}" if self._path else key

        if self._metadata.zarr_format >= 3:
            return f"{child_path}/zarr.json" in self._mapper
        else:
            return (
                f"{child_path}/.zgroup" in self._mapper
                or f"{child_path}/.zarray" in self._mapper
            )

    def __getitem__(self, key: str) -> ZarrGroup | ZarrArray:
        """Get a child node (group or array)."""
        child_path = f"{self._path}/{key}" if self._path else key

        if self._metadata.zarr_format >= 3:
            return self._getitem_v3(child_path, key)
        else:
            return self._getitem_v2(child_path, key)

    def _getitem_v3(self, child_path: str, key: str) -> ZarrGroup | ZarrArray:
        """Get a v3 child node."""
        data = self._mapper.get(f"{child_path}/zarr.json")
        if data is None:
            raise KeyError(key)

        meta = json.loads(data.decode("utf-8"))
        node_type = meta.get("node_type")

        if node_type == "group":
            return ZarrGroup(self._mapper, child_path, meta)
        elif node_type == "array":
            return ZarrArray(self._mapper, child_path, meta)
        else:
            raise ValueError(f"Unknown node_type: {node_type}")

    def _getitem_v2(self, child_path: str, key: str) -> ZarrGroup | ZarrArray:
        """Get a v2 child node."""
        # Try group
        zgroup_data = self._mapper.get(f"{child_path}/.zgroup")
        if zgroup_data is not None:
            attrs_data = self._mapper.get(f"{child_path}/.zattrs")
            attrs = json.loads(attrs_data.decode("utf-8")) if attrs_data else {}
            meta = {"zarr_format": 2, "node_type": "group", "attributes": attrs}
            return ZarrGroup(self._mapper, child_path, meta)

        # Try array
        zarray_data = self._mapper.get(f"{child_path}/.zarray")
        if zarray_data is not None:
            array_meta = json.loads(zarray_data.decode("utf-8"))
            attrs_data = self._mapper.get(f"{child_path}/.zattrs")
            attrs = json.loads(attrs_data.decode("utf-8")) if attrs_data else {}
            meta = {
                "zarr_format": 2,
                "node_type": "array",
                "attributes": attrs,
                "shape": array_meta.get("shape"),
                "data_type": str(np.dtype(array_meta["dtype"]))
                if "dtype" in array_meta
                else None,
            }
            return ZarrArray(self._mapper, child_path, meta)

        raise KeyError(key)

    if TYPE_CHECKING:

        def to_zarr_python(self) -> zarr.Group:  # type: ignore
            """Convert to a zarr-python Group object."""


class ZarrArray(ZarrNode):
    """Wrapper around a zarr v2/v3 array."""

    __slots__ = ()

    @classmethod
    def node_type(cls) -> Literal["array"]:
        """Return the node type (group or array)."""
        return "array"

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        if self._metadata.shape is None:
            raise ValueError("Array metadata missing 'shape'")
        return len(self._metadata.shape)

    @property
    def dtype(self) -> np.dtype:
        """Return the data type."""
        if self._metadata.data_type is None:
            raise ValueError("Array metadata missing 'data_type'")
        # Data type is already normalized to numpy dtype string in _load_metadata
        return np.dtype(self._metadata.data_type)

    if TYPE_CHECKING:

        def to_zarr_python(self) -> zarr.Array:  # type: ignore
            """Convert to a zarr-python Array object."""

    def to_tensorstore(self) -> tensorstore.TensorStore:
        """Convert to a tensorstore TensorStore object."""
        try:
            import tensorstore as ts  # type: ignore
        except ImportError as e:
            raise ImportError(
                "tensorstore package is required for to_tensorstore()"
            ) from e

        spec = {
            "driver": "zarr3" if self._metadata.zarr_format == 3 else "zarr",
            "kvstore": self.get_uri(),
        }
        future = ts.open(spec)
        return future.result()


def open(uri: str | os.PathLike) -> ZarrGroup | ZarrArray:  # noqa: A001
    """Open a zarr v2/v3 group or array from a URI.

    Parameters
    ----------
    uri : str | os.PathLike
        The URI of the zarr store or a specific group/array within it.

    Returns
    -------
    ZarrGroup | ZarrArray
        The opened zarr group or array.

    Raises
    ------
    FileNotFoundError
        If no zarr metadata is found at the specified URI.
    ValueError
        If the metadata is invalid or inconsistent.
    """
    from fsspec import get_mapper

    uri = os.fspath(uri)
    mapper = get_mapper(uri)
    node = ZarrNode(mapper)
    if node._metadata.node_type == "group":
        return ZarrGroup(mapper, node._path, node._metadata)
    elif node._metadata.node_type == "array":
        return ZarrArray(mapper, node._path, node._metadata)
    else:
        raise ValueError(f"Unknown node_type: {node._metadata.node_type}")
