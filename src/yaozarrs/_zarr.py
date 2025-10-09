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
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType, NoneType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, TypeAlias, overload

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from yaozarrs import v04, v05

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TypeVar

    import tensorstore  # type: ignore
    import zarr  # type: ignore
    from fsspec import FSMap

    _T = TypeVar("_T")

__all__ = ["ZarrArray", "ZarrGroup", "ZarrMetadata", "open_group"]

# --------------------------------
# JSON DOCUMENTS
# --------------------------------
"""All of the structures of json docs we might find in a zarr store."""


class ZGroupV2(BaseModel):
    """Metadata for a zarr group, version 2.

    .zgroup file
    """

    zarr_format: Literal[2]


class ZArrayV2(BaseModel):
    """Metadata for a zarr array, version 2.

    .zarray file
    """

    zarr_format: Literal[2]
    shape: list[int]
    chunks: list[int]
    dtype: str | list
    compressor: dict | None
    fill_value: int | float | str | None
    order: Literal["C", "F"]
    filters: list[dict] | None
    dimension_separator: str | None = None


class ZarrJsonArrayV3(BaseModel):
    """Metadata for a zarr array, version 3.

    zarr.json file for an array
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    shape: list[int]
    data_type: str | list
    chunk_grid: dict
    chunk_key_encoding: dict
    fill_value: int | float | str | None
    codecs: list[dict] | None

    storage_transformers: list[dict] | None = None
    dimension_names: list[str] | None = None
    attributes: dict = Field(default_factory=dict)


class ZarrJsonGroupV3(BaseModel):
    """Metadata for a zarr group, version 3.

    zarr.json file for a group
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: dict = Field(default_factory=dict)


# Union of all possible zarr.json structures
ZarrJson: TypeAlias = Annotated[
    ZarrJsonArrayV3 | ZarrJsonGroupV3, Field(discriminator="node_type")
]

AnyOMEMetadata: TypeAlias = v04.OMEZarrGroupJSON | v05.OMEMetadata


class OMEAttributesV5(BaseModel):
    """The attributes field of a zarr.json document that usually appears nested."""

    ome: v05.OMEMetadata


class OMEZarrJSON(BaseModel):
    """A zarr.json document found in any ome-zarr group."""

    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"
    attributes: OMEAttributesV5


# -------------------  Metadata Loading  -------------------


class ZarrMetadata(BaseModel):
    """Metadata from a zarr metadata file.

    We don't differentiate between v2 and v3 here - both are loaded into this
    common format.  Extra fields may be present depending on the zarr version
    and node type, use `getattr()` or `model_dump()` to access them.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", frozen=True)

    zarr_format: Literal[2, 3]
    node_type: Literal["group", "array"]
    attributes: dict[str, Any] = Field(default_factory=dict)
    shape: tuple[int, ...] | None = None
    data_type: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _fix_inputs(cls, val: Any) -> Any:
        """Fix up input dictionary before validation."""
        # cast v2 "dtype" to "data_type"
        if isinstance(val, dict):
            if "dtype" in val and "data_type" not in val:
                val["data_type"] = val.pop("dtype")
        return val

    def ome_metadata(self) -> v05.OMEMetadata | v04.OMEZarrGroupJSON | None:
        """Return the OME metadata if present in attributes, else None."""
        attrs = self.attributes
        if "ome" in attrs:
            return TypeAdapter(v05.OMEMetadata).validate_python(attrs["ome"])
        else:
            return TypeAdapter(v04.OMEZarrGroupJSON).validate_python(attrs)
        return None


def _load_zarr_json(prefix: str, mapper: Mapping[str, bytes]) -> ZarrMetadata | None:
    """Load and parse zarr v3 metadata (zarr.json)."""
    if json_data := mapper.get(f"{prefix}zarr.json".lstrip("/")):
        return ZarrMetadata.model_validate_json(json_data)
    return None


def _load_zgroup(prefix: str, mapper: Mapping[str, bytes]) -> ZarrMetadata | None:
    """Load and parse zarr v2 group metadata (.zgroup)."""
    zgroup_data = mapper.get(f"{prefix}.zgroup".lstrip("/"))
    if zgroup_data is not None:
        z_group_meta = json.loads(zgroup_data.decode("utf-8"))
        attrs_data = mapper.get(f"{prefix}.zattrs".lstrip("/"))
        attrs = json.loads(attrs_data.decode("utf-8")) if attrs_data else {}
        meta = {**z_group_meta, "node_type": "group", "attributes": attrs}
        return ZarrMetadata.model_validate(meta)
    return None


def _load_zarray(prefix: str, mapper: Mapping[str, bytes]) -> ZarrMetadata | None:
    """Load and parse zarr v2 array metadata (.zarray)."""
    zarray_data = mapper.get(f"{prefix}.zarray".lstrip("/"))
    if zarray_data is not None:
        zarray_meta = json.loads(zarray_data.decode("utf-8"))
        attrs_data = mapper.get(f"{prefix}.zattrs".lstrip("/"))
        attrs = json.loads(attrs_data.decode("utf-8")) if attrs_data else {}
        meta = {**zarray_meta, "node_type": "array", "attributes": attrs}
        return ZarrMetadata.model_validate(meta)
    return None


def _load_zarr_metadata(mapper: Mapping[str, bytes], path: str = "") -> ZarrMetadata:
    """Load and parse zarr metadata (v2 or v3).

    First checks for v3 (zarr.json), then v2 (.zgroup or .zarray).  Note that `path`
    does *not* include the name of the metadata file itself (e.g. "zarr.json"), but
    should include any path within the zarr store.

    Parameters
    ----------
    path : str
        The path path within the zarr store (may be empty).
    mapper : Mapping[str, bytes]
        The mapper for the zarr store.

    Returns
    -------
    ZarrMetadata
        The parsed zarr metadata.

    Raises
    ------
    FileNotFoundError
        If no zarr metadata is found at the specified prefix.
    ValidationError
        If the metadata is invalid or inconsistent.
    """
    prefix = f"{path}/" if path else ""

    # Try v3 first (zarr.json)
    if (meta := _load_zarr_json(prefix, mapper)) is not None:
        return meta

    # Try v2 (.zgroup or .zarray)
    if (meta := _load_zgroup(prefix, mapper)) is not None:
        return meta

    if (meta := _load_zarray(prefix, mapper)) is not None:
        return meta

    raise FileNotFoundError(  # pragma: no cover
        f"No zarr metadata found at '{prefix}' (tried zarr.json, .zgroup, .zarray)"
    )


# ---------------------------------------------------


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

    @overload
    def get(self, key: str, /) -> bytes | None: ...
    @overload
    def get(self, key: str, /, default: bytes) -> bytes: ...
    @overload
    def get(self, key: str, /, default: _T) -> _T: ...
    def get(self, key: str, default: _T | None = None) -> _T | None:
        """Get a value from the mapper with caching."""
        if key not in self._cache:
            self._cache[key] = val = self._fsmap.get(key, default)
        else:
            val = self._cache[key]
        if isinstance(val, Exception):
            raise val
        return val  # type: ignore[return-value]

    def __contains__(self, key: object) -> bool:
        """Check if a key exists in the mapper."""
        if not isinstance(key, str):
            return False

        if key in self._cache:
            val = self._cache[key]
        else:
            self._cache[key] = val = self._fsmap.get(key)
        return not isinstance(val, (Exception, NoneType))

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

    def __iter__(self) -> Iterator[str]:
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


class ZarrNode:
    """Base class for zarr nodes (groups and arrays)."""

    __slots__ = ("_metadata", "_path", "_store")

    def __init__(
        self,
        store: _CachedMapper | FSMap,
        path: str = "",
        meta: ZarrMetadata | None = None,
    ) -> None:
        """Initialize a zarr node.

        Parameters
        ----------
        store : _CachedMapper | FSMap
            The mapper for the zarr store. Should be a _CachedMapper for best
            performance, or an FSMap which will be wrapped automatically.  In the future
            it's possible we could support other mapping types.
        path : str
            The path to this node within the zarr store (relative to the store root).
        meta : dict[str, Any] | ZarrMetadata | None
            Optional pre-loaded metadata dictionary. If not provided, it will be
            loaded from the store.
        """
        # Ensure we have a cached mapper for performance
        if not isinstance(store, _CachedMapper):
            store = _CachedMapper(store)

        self._store = store
        self._path = str(path).rstrip("/")
        if meta is None:
            self._metadata = _load_zarr_metadata(self._store, self._path)
        elif isinstance(meta, ZarrMetadata):
            if meta.node_type != self.node_type():  # pragma: no cover
                raise ValueError(
                    f"Metadata node_type '{meta.node_type}' does not match "
                    f"expected '{self.node_type()}'"
                )
            self._metadata = meta
        else:  # pragma: no cover
            raise ValueError("meta must be a ZarrMetadata instance or None")

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Return attributes as a read-only mapping."""
        return MappingProxyType(self._metadata.attributes)

    @property
    def path(self) -> str:
        """Return the path of this node relative to the store root."""
        return self._path

    @property
    def metadata(self) -> ZarrMetadata:
        """Return the parsed zarr metadata."""
        return self._metadata

    @property
    def store(self) -> Mapping[str, bytes]:
        """Return the underlying store mapping (read-only)."""
        return MappingProxyType(self._store)

    def to_zarr_python(self) -> zarr.Array | zarr.Group:
        """Convert to a zarr-python Array or Group object."""
        try:
            import zarr  # type: ignore
        except ImportError as e:
            raise ImportError("zarr package is required for to_zarr_python()") from e

        return zarr.open(self.store_path, mode="r")

    @classmethod
    def node_type(cls) -> Literal["group", "array"]:
        """Return the node type (group or array)."""
        raise NotImplementedError("Cannot instantiate base ZarrNode")

    @property
    def store_path(self) -> str:
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
        mapper = self._store
        if isinstance(mapper, _CachedMapper):
            mapper = mapper._fsmap

        # Build the full path including our internal zarr path
        if self._path:
            full_path = f"{mapper.root.rstrip('/')}/{self._path}"
        else:
            full_path = mapper.root

        # For local file systems, use Path.as_uri() for proper cross-platform
        # URI formatting (especially Windows which needs file:///C:/ not file://C:/)
        protocol = mapper.fs.protocol
        if isinstance(protocol, tuple):
            protocol = protocol[0]
        if protocol in ("file", "local"):
            return Path(full_path).as_uri()

        return mapper.fs.unstrip_protocol(full_path)

    @property
    def zarr_format(self) -> Literal[2, 3]:
        """Return the zarr format version (2 or 3)."""
        return self._metadata.zarr_format

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"<{cls_name} {self.store_path}>"


class ZarrGroup(ZarrNode):
    """Wrapper around a zarr v2/v3 group.

    Matches zarr-python behavior: expects all children to be the same
    zarr_format version as the parent. Does not support mixed hierarchies.
    """

    __slots__ = ("_ome_metadata",)

    def ome_version(self) -> str | None:
        """Return ome_version if present, else None.

        Attempt to determine version as minimally as possible without
        parsing full models.
        """
        attrs = self._metadata.attributes
        if "ome" in attrs:
            if "version" in attrs["ome"]:
                return attrs["ome"]["version"]
        return None  # pragma: no cover

    def ome_metadata(self) -> v05.OMEMetadata | v04.OMEZarrGroupJSON | None:
        if not hasattr(self, "_ome_metadata"):
            self._ome_metadata = self._metadata.ome_metadata()
        return self._ome_metadata

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
            self._store.getitems(metadata_paths)
        except Exception:
            # If batch fetching fails, _CachedMapper will fall back to sequential
            # access when files are actually requested via __getitem__
            pass

    def __contains__(self, key: str) -> bool:
        """Check if a child node exists."""
        child_path = f"{self._path}/{key}" if self._path else key

        if self._metadata.zarr_format >= 3:
            return f"{child_path}/zarr.json" in self._store
        else:
            return (
                f"{child_path}/.zgroup" in self._store
                or f"{child_path}/.zarray" in self._store
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
        prefix = f"{child_path}/"
        if (meta := _load_zarr_json(prefix, self._store)) is not None:
            if meta.node_type == "group":
                return ZarrGroup(self._store, child_path, meta)
            elif meta.node_type == "array":
                return ZarrArray(self._store, child_path, meta)
            else:  # pragma: no cover
                raise ValueError(f"Unknown node_type: {meta.node_type}")

        raise KeyError(key)

    def _getitem_v2(self, child_path: str, key: str) -> ZarrGroup | ZarrArray:
        """Get a v2 child node."""
        prefix = f"{child_path}/"
        # Try group
        if (meta := _load_zgroup(prefix, self._store)) is not None:
            return ZarrGroup(self._store, child_path, meta)

        if (meta := _load_zarray(prefix, self._store)) is not None:
            return ZarrArray(self._store, child_path, meta)

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
        if self._metadata.shape is None:  # pragma: no cover
            raise ValueError("Array metadata missing 'shape'")
        return len(self._metadata.shape)

    @property
    def dtype(self) -> str:
        """Return the data type."""
        if self._metadata.data_type is None:  # pragma: no cover
            raise ValueError("Array metadata missing 'data_type'")
        # Data type is already normalized to numpy dtype string in _load_metadata
        return self._metadata.data_type

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
            "kvstore": _fsmap_to_tensorstore_kvstore(self._store._fsmap, self._path),
        }
        future = ts.open(spec)
        return future.result()


def open_group(uri: str | os.PathLike | Any) -> ZarrGroup:
    """Open a zarr v2/v3 group from a URI.

    Parameters
    ----------
    uri : str | os.PathLike
        The URI of the zarr store (e.g., "https://...", "s3://...", "/path/to/file"),
        or a zarr-python Group.

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
    try:
        from fsspec import FSMap, get_mapper
    except ImportError as e:
        raise ImportError(
            "fsspec package is required for open_group().  "
            "Please install with `pip install yaozarrs[io]` or "
            "`pip install fsspec`."
        ) from e

    if isinstance(uri, (str, os.PathLike)):
        uri = os.path.expanduser(os.fspath(uri))
    elif isinstance(uri, ZarrGroup):
        return uri
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


# ---------------------------------------------------


def _fsmap_to_tensorstore_kvstore(fsmap: FSMap, path: str = "") -> dict:
    """Convert FSMap to tensorstore kvstore spec.

    Parameters
    ----------
    fsmap : fsspec.mapping.FSMap
        The fsspec mapper to convert
    path : str, optional
        Additional path relative to the FSMap root to use as prefix

    Returns
    -------
    dict
        Tensorstore kvstore spec
    """
    # Get the protocol from the filesystem
    protocol = fsmap.fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]  # Take first if multiple aliases

    # Normalize the additional path (strip leading/trailing slashes)
    if path:
        path = path.strip("/")

    # Map fsspec protocols to tensorstore kvstore drivers
    if protocol in ("file", "local"):
        base_path = os.path.abspath(fsmap.root)
        if path:
            base_path = os.path.join(base_path, path)
        return {"driver": "file", "path": base_path}

    elif protocol in ("http", "https"):
        # fsmap.root already contains the full URL for http/https
        base_url = fsmap.root
        if not base_url.startswith(("http://", "https://")):
            base_url = f"{protocol}://{base_url}"

        # Append additional path to URL
        if path:
            base_url = f"{base_url.rstrip('/')}/{path}"

        return {"driver": "http", "base_url": base_url}

    elif protocol == "memory":
        return {"driver": "memory"}

    elif protocol in ("s3", "s3a"):  # pragma: no cover
        # Extract bucket and path from root
        parts = fsmap.root.split("/", 1)
        bucket = parts[0]
        base_path = parts[1] if len(parts) > 1 else ""

        # Combine base path with additional path
        if path:
            base_path = f"{base_path}/{path}" if base_path else path

        spec = {"driver": "s3", "bucket": bucket}
        if base_path:
            spec["path"] = base_path

        # Include S3 credentials if available
        if hasattr(fsmap.fs, "key") and fsmap.fs.key:
            spec["aws_credentials"] = {
                "access_key_id": fsmap.fs.key,
                "secret_access_key": fsmap.fs.secret,
            }

        return spec

    elif protocol in ("gcs", "gs"):  # pragma: no cover
        # Extract bucket and path from root
        parts = fsmap.root.split("/", 1)
        bucket = parts[0]
        base_path = parts[1] if len(parts) > 1 else ""

        # Combine base path with additional path
        if path:
            base_path = f"{base_path}/{path}" if base_path else path

        spec = {"driver": "gcs", "bucket": bucket}
        if base_path:
            spec["path"] = base_path

        return spec

    else:  # pragma: no cover
        raise ValueError(
            f"Cannot map fsspec protocol '{protocol}' to tensorstore kvstore. "
            f"Supported protocols: file, s3, gcs, http/https, memory"
        )
