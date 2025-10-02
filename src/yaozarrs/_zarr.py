"""Minimal zarr v2/v3 support for reading metadata and structure.

Allows avoiding a full dependency on zarr-python
(which pins strictly, and has a much larger API surface).

This implementation matches zarr-python's behavior: a zarr group expects
all children to be the same zarr_format version. Mixed v2/v3 hierarchies
are not supported.

Array data access is only supported via conversion to tensorstore or zarr-python.

This module requires fsspec for filesystem operations.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from fsspec import FSMap, get_mapper
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

    import tensorstore  # type: ignore
    import zarr  # type: ignore

    from yaozarrs._validate import AnyOME


class _CachedMapper(Mapping[str, bytes]):
    """Caching wrapper for FSMap that caches metadata file reads.

    fsspec does NOT cache individual file reads - it only caches directory
    listings and maintains HTTP connection pools. Since zarr validation
    requires reading thousands of small metadata files, we add an application-
    level cache here.

    This cache is unbounded since zarr metadata files are:
    - Small (typically < 10KB each)
    - Immutable (don't change during validation)
    - Limited in number (even large plates have ~10K metadata files)

    For HTTPS stores, this cache works with fsspec's async batch fetching
    via getitems(), which uses aiohttp to fetch up to 1280 files concurrently.
    """

    def __init__(self, mapper: FSMap) -> None:
        self._fsmap = mapper
        self._cache: dict[str, bytes | None | Exception] = {}

    def get(self, key: str, default: bytes | None = None) -> bytes | None:
        """Get a value from the mapper with caching."""
        if key not in self._cache:
            self._cache[key] = val = self._fsmap.get(key, default)
        else:
            val = self._cache[key]
        if isinstance(val, Exception):
            raise val
        return val

    def __contains__(self, key: object) -> bool:
        """Check if a key exists in the mapper."""
        if not isinstance(key, str):
            return False
        if key not in self._cache:
            self._cache[key] = self._fsmap.get(key)
        return self._cache[key] is not None

    def getitems(self, keys: list[str], on_error: str = "omit") -> dict[str, bytes]:
        """Batch fetch multiple items and cache them.

        This is a performance optimization for remote stores that support
        batch fetching via the getitems method.

        Parameters
        ----------
        keys : list[str]
            List of keys to fetch from the mapper.
        on_error : {'omit', 'raise'}, default='omit'
            How to handle missing keys:
            - 'omit': Skip missing keys, return only found items
            - 'raise': Raise exception for missing keys

        Returns
        -------
        dict[str, bytes]
            Dictionary mapping keys to their byte content.
            Keys that don't exist are omitted from the result.
        """
        # Check which keys are not yet cached
        uncached_keys = [k for k in keys if k not in self._cache]

        # Fetch uncached keys in batch using FSMap.getitems()
        # For HTTPS, this uses async aiohttp with concurrency up to 1280 requests
        if uncached_keys:
            try:
                # use on_error='return' to get all results including missing keys
                # which will be stored as exceptions (usually KeyErrors)
                results = self._fsmap.getitems(uncached_keys, on_error="return")
                self._cache.update(results)
            except Exception:
                # If batch fetch fails, fall back to individual gets
                for key in uncached_keys:
                    try:
                        self._cache[key] = self._fsmap.get(key)
                    except Exception:
                        if on_error == "raise":
                            raise
                        self._cache[key] = None

        # Return cached results (only keys with non-None values)
        result = {}
        for key in keys:
            val = self._cache.get(key)
            if val is not None:
                result[key] = val
        return result

    def __iter__(self):
        """Delegate iteration to underlying mapper."""
        return iter(self._fsmap)

    def __len__(self) -> int:
        """Delegate len to underlying mapper."""
        return len(self._fsmap)

    def __getitem__(self, key: str) -> bytes:
        """Get item from mapper (required by Mapping protocol)."""
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying mapper."""
        return getattr(self._fsmap, name)


class ZarrMetadata(BaseModel):
    """Metadata from a zarr metadata file."""

    zarr_format: Literal[2, 3]
    node_type: str  # "group" or "array"
    attributes: dict[str, Any] = Field(default_factory=dict)
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

    def __init__(
        self,
        mapper: _CachedMapper | FSMap,
        path: str = "",
        meta: dict[str, Any] | ZarrMetadata | None = None,
    ) -> None:
        """Initialize a zarr node.

        Parameters
        ----------
        mapper : _CachedMapper | FSMap
            The mapper for the zarr store. Should be a _CachedMapper for best
            performance, or an FSMap which will be wrapped automatically.
        path : str
            The path within the zarr store.
        meta : dict[str, Any] | ZarrMetadata | None
            Optional pre-loaded metadata dictionary. If not provided, it will be
            loaded from the store.
        """
        # Ensure we have a cached mapper for performance
        if isinstance(mapper, FSMap) and not isinstance(mapper, _CachedMapper):
            mapper = _CachedMapper(mapper)

        self._mapper = mapper
        self._path = path.rstrip("/")
        if meta is None:
            self._metadata = self._load_metadata()
        else:
            self._metadata = ZarrMetadata.model_validate(meta)
            if self._metadata.node_type != self.node_type():  # pragma: no cover
                raise ValueError(
                    f"Metadata node_type '{self._metadata.node_type}' does not match "
                    f"expected '{self.node_type()}'"
                )

    def _load_metadata(self) -> ZarrMetadata:
        """Load and parse zarr metadata (v2 or v3)."""
        prefix = f"{self._path}/" if self._path else ""

        # Try v3 first (zarr.json)
        if json_data := self._mapper.get(f"{prefix}zarr.json".lstrip("/")):
            return ZarrMetadata.model_validate_json(json_data)

        # Try v2 (.zgroup or .zarray)
        # v2 metadata files don't have node_type, we need to infer it
        zgroup_data = self._mapper.get(f"{prefix}.zgroup".lstrip("/"))
        if zgroup_data is not None:
            z_group_meta = json.loads(zgroup_data.decode("utf-8"))
            attrs_data = self._mapper.get(f"{prefix}.zattrs".lstrip("/"))
            attrs = json.loads(attrs_data.decode("utf-8")) if attrs_data else {}
            meta = {**z_group_meta, "node_type": "group", "attributes": attrs}
            return ZarrMetadata.model_validate(meta)

        zarray_data = self._mapper.get(f"{prefix}.zarray".lstrip("/"))
        if zarray_data is not None:
            zarray_meta = json.loads(zarray_data.decode("utf-8"))
            attrs_data = self._mapper.get(f"{prefix}.zattrs".lstrip("/"))
            attrs = json.loads(attrs_data.decode("utf-8")) if attrs_data else {}
            meta = {**zarray_meta, "node_type": "array", "attributes": attrs}
            return ZarrMetadata.model_validate(meta)

        raise FileNotFoundError(  # pragma: no cover
            f"No zarr metadata found at '{self._path}' "
            "(tried zarr.json, .zgroup, .zarray)"
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

        This URI can be used with TensorStore, fsspec, and other libraries.

        Returns
        -------
        str
            A URI string that follows standard protocol://path format.
        """
        # Unwrap cached mapper to access underlying FSMap
        mapper = self._mapper
        if isinstance(mapper, _CachedMapper):
            mapper = mapper._fsmap

        # Build the full path including our internal zarr path
        if self._path:
            full_path = f"{mapper.root.rstrip('/')}/{self._path}"
        else:
            full_path = mapper.root

        return mapper.fs.unstrip_protocol(full_path)


class ZarrGroup(ZarrNode):
    """Wrapper around a zarr v2/v3 group.

    Matches zarr-python behavior: expects all children to be the same
    zarr_format version as the parent. Does not support mixed hierarchies.
    """

    __slots__ = ("_ome_model",)

    def ome_model(self) -> AnyOME:
        if not hasattr(self, "_ome_model"):
            from ._validate import validate_ome_object

            self._ome_model = validate_ome_object(self._metadata.attributes)
        return self._ome_model

    @classmethod
    def node_type(cls) -> Literal["group"]:
        """Return the node type (group or array)."""
        return "group"

    def prefetch_children(self, child_keys: Iterable[str]) -> None:
        """Prefetch metadata for multiple children using async batch fetching.

        For HTTPS URIs, this triggers fsspec's async batch fetching which uses
        aiohttp to fetch up to 1280 files concurrently. Results are cached in
        the _CachedMapper for subsequent access.

        This is critical for performance with remote stores - a 384-well plate
        can have 2000+ metadata files. Without batching, sequential fetches would
        take 10+ minutes; with batching, validation completes in ~80 seconds.

        Parameters
        ----------
        child_keys : list[str]
            List of child keys to prefetch.
        """
        # Build list of metadata file paths to fetch
        metadata_paths = []
        for key in child_keys:
            child_path = f"{self._path}/{key}" if self._path else key
            if self._metadata.zarr_format >= 3:
                metadata_paths.append(f"{child_path}/zarr.json")
            else:
                # For v2, we need to check both .zgroup and .zarray
                metadata_paths.extend(
                    [f"{child_path}/.zgroup", f"{child_path}/.zarray"]
                )

        # Batch fetch using getitems - _CachedMapper handles fallback if needed
        try:
            self._mapper.getitems(metadata_paths)
        except Exception:
            # If batch fetching fails, _CachedMapper will fall back to sequential
            # access when files are actually requested via __getitem__
            pass

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

    def get(self, key: str, default: Any = None) -> ZarrGroup | ZarrArray | None:
        """Get a child node (group or array), or return default if not found."""
        try:
            return self[key]
        except KeyError:
            return default

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
                "data_type": array_meta.get("dtype"),
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
        The URI of the zarr store (e.g., "https://...", "s3://...", "/path/to/file")

    Returns
    -------
    ZarrGroup | ZarrArray
        The opened zarr group or array with caching enabled.

    Raises
    ------
    FileNotFoundError
        If no zarr metadata is found at the specified URI.
    ValueError
        If the metadata is invalid or inconsistent.
    """
    uri = os.fspath(uri)
    mapper = get_mapper(uri)

    if not isinstance(mapper, FSMap):  # pragma: no cover
        raise TypeError(f"Expected FSMap from get_mapper, got {type(mapper)}")

    # Wrap in caching layer for metadata-level caching
    cached_mapper = _CachedMapper(mapper)
    node = ZarrNode(cached_mapper)

    if node._metadata.node_type == "group":
        return ZarrGroup(cached_mapper, node._path, node._metadata)
    elif node._metadata.node_type == "array":
        return ZarrArray(cached_mapper, node._path, node._metadata)
    else:  # pragma: no cover
        raise ValueError(f"Unknown node_type: {node._metadata.node_type}")


def open_group(uri: str | os.PathLike | Any) -> ZarrGroup:
    """Open a zarr v2/v3 group from a URI.

    Parameters
    ----------
    uri : str | os.PathLike
        The URI of the zarr store (e.g., "https://...", "s3://...", "/path/to/file")

    Returns
    -------
    ZarrGroup
        The opened zarr group with caching enabled.

    Raises
    ------
    FileNotFoundError
        If no zarr metadata is found at the specified URI.
    ValueError
        If the metadata is invalid or inconsistent, or if the root node is not a group.
    """
    if isinstance(uri, (str, os.PathLike)):
        uri = os.path.expanduser(os.fspath(uri))
    elif hasattr(uri, "store"):
        # Handle both zarr v2 and v3 Group objects
        # v3: str(group.store) returns a URI like "file:///path"
        # v2: group.store.path returns the directory path
        if hasattr(uri.store, "path"):
            # Zarr v2: DirectoryStore has .path attribute
            uri = uri.store.path
        else:
            # Zarr v3: LocalStore's __str__ returns URI
            uri = str(uri.store)
    else:  # pragma: no cover
        raise TypeError(
            "uri must be a string, os.PathLike, or have a 'store' attribute"
        )

    mapper = get_mapper(uri)

    if not isinstance(mapper, FSMap):  # pragma: no cover
        raise TypeError(f"Expected FSMap from get_mapper, got {type(mapper)}")

    # Wrap in caching layer for metadata-level caching
    cached_mapper = _CachedMapper(mapper)
    node = ZarrNode(cached_mapper)

    if node._metadata.node_type != "group":
        raise ValueError(
            f"Expected root node to be 'group', got '{node._metadata.node_type}'"
        )
    return ZarrGroup(cached_mapper, node._path, node._metadata)
