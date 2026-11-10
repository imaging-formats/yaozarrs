"""Array creation and writing backends (zarr-python, tensorstore, or custom)."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import itertools
import math
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Literal, TypeAlias

    import dask.array as da
    import tensorstore
    import zarr
    from numpy.typing import DTypeLike

    WriterName = Literal["zarr", "tensorstore", "auto"]
    ZarrWriter: TypeAlias = WriterName | "CreateArrayFunc"
    CompressionName = Literal["blosc-zstd", "blosc-lz4", "zstd", "none"]
    AnyZarrArray: TypeAlias = zarr.Array | tensorstore.TensorStore
    ShapeLike: TypeAlias = tuple[int, ...]

    # Actual data arrays - for write functions
    ArrayLike: TypeAlias = Any  # e.g., numpy.ndarray, dask.array.Array, etc.
    ArrayOrPyramid: TypeAlias = ArrayLike | Sequence[ArrayLike]

    # Shape/dtype specification only (no data) - for prepare/add functions
    ShapeAndDType: TypeAlias = tuple[ShapeLike, DTypeLike]
    """A tuple of (shape, dtype) describing an array."""

    ShapeAndDTypeOrPyramid: TypeAlias = ShapeAndDType | Sequence[ShapeAndDType]


@runtime_checkable
class CreateArrayFunc(Protocol):
    """Protocol for custom array creation functions.

    This is the type signature for functions that can be passed as the `writer`
    parameter to many functions in this module.  Use this if you'd like to completely
    customize how arrays are created, e.g., using a different backend, or custom
    storage options.  See `_create_array_zarr` and `_create_array_tensorstore` for
    (our internal) reference implementations.
    """

    def __call__(
        self,
        path: Path,
        shape: tuple[int, ...],
        dtype: Any,
        chunks: tuple[int, ...],
        *,
        shards: tuple[int, ...] | None,  # = None,
        overwrite: bool,  # = False,
        compression: CompressionName,  # = "blosc-zstd",
        dimension_names: list[str] | None,  # = None,
    ) -> Any:
        """Create array structure without writing data.

        Parameters
        ----------
        path : Path
            Path to create array
        shape : tuple[int, ...]
            Array shape
        dtype : dtype
            Data type
        chunks : tuple[int, ...]
            Chunk shape (already resolved by yaozarrs)
        shards : tuple[int, ...] | None
            Shard shape for Zarr v3 sharding, or None
        dimension_names : list[str] | None
            Names for each dimension
        overwrite : bool
            Whether to overwrite existing array
        compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none"
            Compression codec to use

        Returns
        -------
        Any
            Array object that supports numpy-style indexing for writing
            (e.g., `zarr.Array` or `tensorstore.TensorStore`).  Other arrays are
            written with a single `array[:] = data`, computing dask data in memory.
        """
        ...


# TODO: I suspect there are better chunk calculation algorithms in the backends.
def _resolve_chunks(
    shape: tuple[int, ...],
    dtype: Any,
    chunk_shape: tuple[int, ...] | Literal["auto"] | None,
) -> tuple[int, ...]:
    """Resolve chunk shape based on user input."""
    if chunk_shape == "auto":
        # FIXME: numpy is not listed in any of our extras...
        # this is a big assumption, and could be avoided by writing our own itemsize()
        import numpy as np

        # Convert to np.dtype to ensure we have itemsize (handles types like np.uint16)
        dtype = np.dtype(dtype)
        return _calculate_auto_chunks(shape, dtype.itemsize)
    elif chunk_shape is None:
        return shape
    else:
        # Clamp to array shape
        return tuple(min(c, s) for c, s in zip(chunk_shape, shape, strict=False))


def _calculate_auto_chunks(
    shape: tuple[int, ...],
    dtype_itemsize: int,
    target_mb: int = 4,
) -> tuple[int, ...]:
    """Calculate chunk shape targeting approximately target_mb chunk size.

    Strategy:
    - Set non-spatial dims (T, C) to 1 for efficient single-plane access
    - Iteratively halve largest spatial dimension until under target size
    """
    target_elements = (target_mb * 1024 * 1024) // dtype_itemsize
    # (the last 3 dims, or all if fewer, are taken to be spatial)
    n_non_spatial = max(0, len(shape) - 3)
    spatial = list(shape[n_non_spatial:])
    while math.prod(spatial) > target_elements and max(spatial) > 1:
        i = spatial.index(max(spatial))
        spatial[i] = max(1, spatial[i] // 2)
    return (1,) * n_non_spatial + tuple(spatial)


# ######################## Array Creation Functions #############################


def _create_array_zarr(
    path: Path,
    shape: tuple[int, ...],
    dtype: Any,
    chunks: tuple[int, ...],
    *,
    shards: tuple[int, ...] | None,
    dimension_names: list[str] | None,
    overwrite: bool,
    compression: CompressionName,
) -> Any:
    """Create zarr array structure using zarr-python, return array object."""
    import zarr
    from zarr.codecs import BloscCodec, BytesCodec, ZstdCodec

    serializer = BytesCodec(endian="little")
    if compression == "blosc-zstd":
        compressors = (BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),)
    elif compression == "blosc-lz4":
        compressors = (BloscCodec(cname="lz4", clevel=5, shuffle="shuffle"),)
    elif compression == "zstd":
        compressors = (ZstdCodec(level=3),)
    elif compression == "none":
        compressors = ()
    else:
        raise ValueError(f"Unknown compression: {compression}")

    return zarr.create_array(
        str(path),
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        dimension_names=dimension_names,
        zarr_format=3,
        overwrite=overwrite,
        serializer=serializer,
        compressors=compressors,
    )


def _create_array_tensorstore(
    path: Path,
    shape: tuple[int, ...],
    dtype: Any,
    chunks: tuple[int, ...],
    *,
    shards: tuple[int, ...] | None,
    dimension_names: list[str] | None,
    overwrite: bool,
    compression: CompressionName,
) -> Any:
    """Create zarr array using tensorstore, return store object."""
    import tensorstore as ts

    if compression == "blosc-zstd":
        chunk_codecs = [
            {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 3}}
        ]
    elif compression == "blosc-lz4":
        chunk_codecs = [
            {"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5}}
        ]
    elif compression == "zstd":
        chunk_codecs = [{"name": "zstd", "configuration": {"level": 3}}]
    elif compression == "none":
        chunk_codecs = []
    else:
        raise ValueError(f"Unknown compression: {compression}")

    codecs = chunk_codecs
    chunk_layout = {"chunk": {"shape": list(chunks)}}
    if shards is not None:
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {"chunk_shape": list(chunks), "codecs": chunk_codecs},
            }
        ]
        chunk_layout = {"write_chunk": {"shape": list(shards)}}

    domain: dict = {"shape": list(shape)}
    if dimension_names:
        domain["labels"] = dimension_names

    # Get dtype string - handle both np.dtype objects and type classes
    try:
        dtype_str = dtype.name  # np.dtype object
    except AttributeError:
        dtype_str = str(dtype)  # fallback

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "schema": {
            "dtype": dtype_str,
            "domain": domain,
            "chunk_layout": chunk_layout,
            "codec": {"driver": "zarr3", "codecs": codecs},
        },
        "create": True,
        "delete_existing": overwrite,
    }
    return ts.open(spec).result()


def _align_dask_chunks(data: da.Array, unit: tuple[int, ...]) -> da.Array:
    """Rechunk axes whose dask chunk boundaries don't fall on `unit` boundaries."""
    new_chunks: list[int | tuple[int, ...]] = []
    for chunks, size in zip(data.chunks, unit, strict=False):
        bounds = itertools.accumulate(chunks[:-1])
        new_chunks.append(chunks if all(b % size == 0 for b in bounds) else size)
    if tuple(new_chunks) == data.chunks:
        return data
    return data.rechunk(tuple(new_chunks))


def _is_loaded_instance(obj: Any, module: str, cls: str) -> bool:
    """Return whether `obj` is a `module.cls`, without importing `module`."""
    mod = sys.modules.get(module)
    return mod is not None and isinstance(obj, getattr(mod, cls))


class _TensorStoreTarget:
    """A `da.store` target that writes each block to a tensorstore array."""

    def __init__(self, array: Any) -> None:
        self._array = array

    def __setitem__(self, key: Any, value: Any) -> None:
        self._array[key].write(value).result()


def _write_to_array(array: Any, data: ArrayLike, *, progress: bool) -> None:
    """Write data to an already-created array (zarr, tensorstore, or custom)."""
    is_dask = "dask" in sys.modules and hasattr(data, "compute")
    if is_dask:
        import dask.array as da

        dask_data = cast("da.Array", data)

        if progress:
            from dask.diagnostics.progress import ProgressBar

            ctx = ProgressBar()
        else:
            ctx = nullcontext()

        with ctx:
            # With lock=False, dask chunks must not split a write unit (shard, else
            # chunk): concurrent read-modify-writes of the same unit would race, and
            # the last writer wins (losing data).
            if _is_loaded_instance(array, "zarr", "Array"):
                write_unit = getattr(array, "shards", None) or array.chunks
                dask_data = _align_dask_chunks(dask_data, write_unit)
                da.store(dask_data, array, lock=False)  # ty: ignore[invalid-argument-type]
                return
            if _is_loaded_instance(array, "tensorstore", "TensorStore"):
                # (block by block, rather than computing the whole array first)
                write_unit = tuple(array.chunk_layout.write_chunk.shape)
                dask_data = _align_dask_chunks(dask_data, write_unit)
                target = _TensorStoreTarget(array)
                da.store(dask_data, target, lock=False)  # ty: ignore[invalid-argument-type]
                return
            # The write unit of any other array is unknown (custom arrays may not
            # even expose one), so concurrent chunk writes could lose data.
            data = dask_data.compute()

    if _is_loaded_instance(array, "tensorstore", "TensorStore"):
        array[:].write(data).result()
    else:
        array[:] = data


# ######################## Array Writing Functions #############################


def _get_create_func(writer: str | CreateArrayFunc) -> CreateArrayFunc:
    if isinstance(writer, CreateArrayFunc):
        return writer

    if writer == "auto":
        for candidate in ["tensorstore", "zarr"]:
            try:
                return _get_create_func(candidate)
            except ImportError:
                continue
        raise ImportError(
            "No suitable writer found for OME-Zarr writing. "
            "Please install either yaozarrs[write-zarr] or yaozarrs[write-tensorstore]"
        )

    if writer == "tensorstore":
        if not importlib.util.find_spec("tensorstore"):
            raise ImportError(
                "tensorstore is required for the 'tensorstore' writer. "
                "Please pip install with yaozarrs[write-tensorstore]"
            )
        return _create_array_tensorstore

    if writer == "zarr":
        if not importlib.util.find_spec("zarr"):
            raise ImportError(
                "zarr-python is required for the 'zarr' writer. "
                "Please pip install with yaozarrs[write-zarr]"
            )
        zarr_version_str = importlib.metadata.version("zarr")
        if int(zarr_version_str.split(".")[0]) < 3:
            raise ImportError(
                f"zarr v3 or higher is required for OME-Zarr writing, "
                f"but zarr v{zarr_version_str} is installed. "
                "Please upgrade zarr to v3 or higher."
            )
        return _create_array_zarr

    raise ValueError(
        f"Unknown writer option: {writer}.  "
        "Must be 'zarr', 'tensorstore', 'auto', or a custom function."
    )
