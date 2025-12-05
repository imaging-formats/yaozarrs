"""OME-Zarr v0.5 writing functionality.

This module provides convenience functions to write OME-Zarr v0.5 groups.

The general pattern is:
1. Create your OME-Zarr metadata model using the yaozarrs.v05 models.
2. Prepare your array data as numpy or dask arrays.
3. Use the appropriate write function (e.g. `write_image` or `write_bioformats2raw`)
   to write the data and metadata to a Zarr store.
4. optionally: Customize chunking, sharding, and writing backend (zarr, tensorstore, or
   your own function) as needed.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, overload, runtime_checkable

import tensorstore

from ._bf2raw import Bf2Raw
from ._series import Series

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    import numpy as np
    import zarr
    from typing_extensions import Literal, TypeAlias, Unpack

    from yaozarrs.v05._image import Multiscale

    from ._image import Image
    from ._zarr_json import OMEMetadata

    class _PrepareKwargs(TypedDict):
        chunks: tuple[int, ...] | Literal["auto"] | None
        shards: tuple[int, ...] | None
        overwrite: bool
        compression: CompressionName

    class ArrayLike(Protocol):
        """Protocol for array-like objects."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Shape of the array."""
            ...

        @property
        def dtype(self) -> np.dtype[Any]:
            """Data type of the array."""
            ...

    WriterName = Literal["zarr", "zarrs", "tensorstore", "auto"]
    ZarrWriter: TypeAlias = WriterName | "CreateArrayFunc"
    CompressionName = Literal["blosc-zstd", "blosc-lz4", "zstd", "none"]
    AnyZarrArray: TypeAlias = zarr.Array | tensorstore.TensorStore


@runtime_checkable
class CreateArrayFunc(Protocol):
    """Protocol for custom array creation functions.

    Custom functions should create and return an array object that supports
    numpy-style indexing (e.g., zarr.Array or tensorstore.TensorStore).
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
        Zarr Array object that supports numpy-style indexing for writing
        (e.g., zarr.Array or tensorstore.TensorStore).
        """
        ...


# ######################## Public API ##########################################


def write_image(
    dest: str | PathLike,
    datasets: Sequence[ArrayLike],
    image: Image,
    *,
    writer: ZarrWriter = "auto",
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    if len(image.multiscales) != 1:
        raise NotImplementedError("Image must have exactly one multiscale")

    multiscale = image.multiscales[0]
    shapes, dtypes = _shapes_and_dtypes(datasets, multiscale)

    # Create arrays using prepare_image (handles both built-in and custom)
    dest_path, arrays = prepare_image(
        dest,
        image,
        shapes,
        dtypes,
        chunks=chunks,
        shards=shards,
        writer=writer,
        overwrite=overwrite,
        compression=compression,
    )

    # Write data to arrays
    for data_array, dataset_meta in zip(datasets, multiscale.datasets):
        _write_to_array(arrays[dataset_meta.path], data_array, progress=progress)

    return dest_path


@overload
def prepare_image(
    dest: str | PathLike,
    image: Image,
    shapes: dict[str, tuple[int, ...]],
    dtypes: dict[str, Any],
    *,
    writer: Literal["zarr", "zarrs"],
    **kwargs: Unpack[_PrepareKwargs],
) -> tuple[Path, dict[str, zarr.Array]]: ...
@overload
def prepare_image(
    dest: str | PathLike,
    image: Image,
    shapes: dict[str, tuple[int, ...]],
    dtypes: dict[str, Any],
    *,
    writer: Literal["tensorstore"],
    **kwargs: Unpack[_PrepareKwargs],
) -> tuple[Path, dict[str, tensorstore.TensorStore]]: ...
@overload
def prepare_image(
    dest: str | PathLike,
    image: Image,
    shapes: dict[str, tuple[int, ...]],
    dtypes: dict[str, Any],
    *,
    writer: Literal["auto"] | CreateArrayFunc = ...,
    **kwargs: Unpack[_PrepareKwargs],
) -> tuple[Path, dict[str, AnyZarrArray]]: ...


def prepare_image(
    dest: str | PathLike,
    image: Image,
    shapes: dict[str, tuple[int, ...]],
    dtypes: dict[str, Any],
    *,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
) -> tuple[Path, dict[str, Any]]:
    if len(image.multiscales) != 1:
        raise NotImplementedError("Image must have exactly one multiscale")
    multiscale = image.multiscales[0]

    # Validate inputs
    if set(shapes) != set(dtypes):
        raise ValueError("shapes and dtypes must have the same keys")

    dataset_paths = {ds.path for ds in multiscale.datasets}
    if set(shapes) != dataset_paths:
        extras = set(shapes) - dataset_paths
        missing = dataset_paths - set(shapes)
        raise ValueError(
            "Shapes and dtypes keys must match dataset paths in metadata.\n"
            f"  Extra keys: {extras}\n"
            f"  Missing keys: {missing}"
        )

    # Get create function
    create_func = _get_create_func(writer)

    # Create zarr group with Image metadata
    dest_path = Path(dest)
    _create_zarr3_group(dest_path, image, overwrite)

    dimension_names = [ax.name for ax in multiscale.axes]

    # Create arrays for each dataset
    arrays = {}
    for dataset_meta in multiscale.datasets:
        ds_path = dataset_meta.path
        shape = shapes[ds_path]
        dtype = dtypes[ds_path]
        # Create array
        arrays[ds_path] = create_func(
            path=dest_path / ds_path,
            shape=shape,
            dtype=dtype,
            chunks=_resolve_chunks(shape, dtype, chunks),
            shards=shards,
            dimension_names=dimension_names,
            overwrite=overwrite,
            compression=compression,
        )

    return dest_path, arrays


def write_bioformats2raw(
    dest: str | PathLike,
    images: dict[str, tuple[Sequence[ArrayLike], Image]],
    *,
    ome_xml: str | None = None,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    progress: bool = False,
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
) -> Path:
    # Extract shapes and dtypes from data arrays for each series
    images_spec = {}
    data_mapping = {}
    for series_name, (datasets, image_model) in images.items():
        if len(image_model.multiscales) != 1:
            raise NotImplementedError("Image must have exactly one multiscale")
        multiscale = image_model.multiscales[0]
        shapes, dtypes = _shapes_and_dtypes(datasets, multiscale)
        images_spec[series_name] = (image_model, shapes, dtypes)
        data_mapping[series_name] = {
            ds.path: data for data, ds in zip(datasets, multiscale.datasets)
        }

    # Create the zarr structure and get array objects
    dest_path, arrays = prepare_bioformats2raw(
        dest,
        images_spec,
        ome_xml=ome_xml,
        chunks=chunks,
        shards=shards,
        writer=writer,
        overwrite=overwrite,
        compression=compression,
    )

    # Write data to each array
    for series_name, dataset_dict in data_mapping.items():
        for dataset_path, data_array in dataset_dict.items():
            array_key = f"{series_name}/{dataset_path}"
            _write_to_array(arrays[array_key], data_array, progress=progress)

    return dest_path


def prepare_bioformats2raw(
    dest: str | PathLike,
    images: dict[str, tuple[Image, dict[str, tuple[int, ...]], dict[str, Any]]],
    *,
    ome_xml: str | None = None,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
) -> tuple[Path, dict[str, Any]]:
    dest_path = Path(dest)

    # Create root zarr.json with bioformats2raw.layout
    bf2raw = Bf2Raw(bioformats2raw_layout=3)  # type: ignore
    _create_zarr3_group(dest_path, bf2raw, overwrite)

    # Create OME/zarr.json with series list
    ome_path = dest_path / "OME"
    series = Series(series=list(images))
    _create_zarr3_group(ome_path, series, overwrite)

    # Write METADATA.ome.xml if provided
    if ome_xml is not None:
        (ome_path / "METADATA.ome.xml").write_text(ome_xml)

    # Create arrays for each series
    all_arrays = {}
    for series_name, (image_model, shapes_dict, dtypes_dict) in images.items():
        _root_path, series_arrays = prepare_image(
            dest_path / series_name,
            image_model,
            shapes_dict,
            dtypes_dict,
            chunks=chunks,
            shards=shards,
            writer=writer,
            overwrite=overwrite,
            compression=compression,
        )
        # Flatten into all_arrays with "series/dataset" keys
        for dataset_path, arr in series_arrays.items():
            all_arrays[f"{series_name}/{dataset_path}"] = arr

    return dest_path, all_arrays


# ######################## Internal Helpers ####################################


def _shapes_and_dtypes(
    datasets: Sequence[ArrayLike], multiscale: Multiscale
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    if len(datasets) != len(multiscale.datasets):
        raise ValueError(
            f"Number of data arrays ({len(datasets)}) must match "
            f"number of datasets in metadata ({len(multiscale.datasets)})"
        )

    # Extract shapes and dtypes from data arrays
    shapes = {}
    dtypes = {}
    for data_array, dataset_meta in zip(datasets, multiscale.datasets):
        shapes[dataset_meta.path] = data_array.shape
        dtypes[dataset_meta.path] = data_array.dtype
    return shapes, dtypes


def _create_zarr3_group(
    dest_path: Path, ome_model: OMEMetadata, overwrite: bool = False
) -> None:
    """Create a zarr group directory with OME metadata in zarr.json."""
    if dest_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Zarr group already exists at {dest_path}. "
                "Use overwrite=True to replace it."
            )
        shutil.rmtree(dest_path, ignore_errors=True)

    dest_path.mkdir(parents=True, exist_ok=True)
    zarr_json = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": ome_model.model_dump(mode="json", exclude_none=True),
        },
    }
    (dest_path / "zarr.json").write_text(json.dumps(zarr_json, indent=2))


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
        if isinstance(dtype, str):
            import numpy as np

            dtype = np.dtype(dtype)
        return _calculate_auto_chunks(shape, dtype.itemsize)
    elif chunk_shape is None:
        return shape
    else:
        # Clamp to array shape
        return tuple(min(c, s) for c, s in zip(chunk_shape, shape))


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
    chunks = list(shape)
    ndim = len(chunks)

    # Set non-spatial dims to 1 (assume last 2-3 are spatial)
    n_spatial = min(3, ndim)
    for i in range(ndim - n_spatial):
        chunks[i] = 1

    # Work on spatial dimensions
    spatial_start = ndim - n_spatial
    spatial_chunks = [shape[i] for i in range(spatial_start, ndim)]

    # Iteratively halve largest dimension
    while math.prod(spatial_chunks) > target_elements and max(spatial_chunks) > 1:
        max_idx = spatial_chunks.index(max(spatial_chunks))
        spatial_chunks[max_idx] = max(1, spatial_chunks[max_idx] // 2)

    # Apply back
    for i, val in enumerate(spatial_chunks):
        chunks[spatial_start + i] = val

    return tuple(chunks)


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

    # Configure compression codecs
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


def _create_array_zarrs(
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
    """Create zarr array using zarrs-python, return array object."""
    import zarr
    import zarrs  # noqa: F401

    # Configure zarr to use zarrs codec pipeline within this context
    with zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}):
        return _create_array_zarr(
            path,
            shape,
            dtype,
            chunks,
            shards=shards,
            dimension_names=dimension_names,
            overwrite=overwrite,
            compression=compression,
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

    # Configure compression codecs
    if compression == "blosc-zstd":
        chunk_codecs = [
            {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 3}},
        ]
    elif compression == "blosc-lz4":
        chunk_codecs = [
            {"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5}},
        ]
    elif compression == "zstd":
        chunk_codecs = [
            {"name": "zstd", "configuration": {"level": 3}},
        ]
    elif compression == "none":
        chunk_codecs = []
    else:
        raise ValueError(f"Unknown compression: {compression}")

    # Build codec chain and chunk layout
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
    store = ts.open(spec).result()
    return store


def _write_to_array(array: Any, data: ArrayLike, *, progress: bool) -> None:
    """Write data to an already-created array (zarr or tensorstore)."""
    is_dask = hasattr(data, "compute")
    if is_dask:
        import dask.array as da

        if progress:
            from dask.diagnostics.progress import ProgressBar

            with ProgressBar():
                # Handle both zarr and tensorstore
                if hasattr(array, "store"):  # zarr.Array
                    da.store(data, array, lock=False)  # type: ignore
                else:  # tensorstore
                    computed = data.compute()
                    array[:].write(computed).result()
        else:
            if hasattr(array, "store"):  # zarr.Array
                da.store(data, array, lock=False)  # type: ignore
            else:  # tensorstore
                computed = data.compute()
                array[:].write(computed).result()
    else:
        if hasattr(array, "store"):  # zarr.Array
            array[:] = data  # type: ignore
        else:  # tensorstore
            array[:].write(data).result()  # type: ignore


# ######################## Array Writing Functions #############################


def _get_create_func(writer: str | CreateArrayFunc) -> CreateArrayFunc:
    """Get the appropriate array create function for the writer."""
    if isinstance(writer, CreateArrayFunc):
        return writer

    if writer in {"tensorstore", "auto"}:
        if importlib.util.find_spec("tensorstore"):
            return _create_array_tensorstore
        elif writer == "tensorstore":
            raise ImportError(
                "tensorstore is required for the 'tensorstore' writer. "
                "Please pip install with yaozarrs[write-tensorstore]"
            )
    if have_zarr := bool(importlib.util.find_spec("zarr")):
        zarr_version_str = importlib.metadata.version("zarr")
        zarr_major_version = int(zarr_version_str.split(".")[0])
        if zarr_major_version < 3 and writer in {"zarr", "zarrs"}:
            raise ImportError(
                f"zarr v3 or higher is required for OME-Zarr v0.5 writing, "
                f"but zarr v{zarr_version_str} is installed. "
                "Please upgrade zarr to v3 or higher."
            )
    if writer in {"zarrs", "auto"}:
        if importlib.util.find_spec("zarrs") and have_zarr:
            return _create_array_zarrs
        raise ImportError(
            "zarrs is required for the 'zarrs' writer. "
            "Please pip install with yaozarrs[write-zarrs]"
        )
    if writer in {"zarr", "auto"}:
        if have_zarr:
            return _create_array_zarr
        raise ImportError(
            "zarr-python is required for the 'zarr' writer. "
            "Please pip install with yaozarrs[write-zarr]"
        )
    if writer == "auto":
        raise ImportError(
            "No suitable writer found for OME-Zarr writing. "
            "Please install either tensorstore, zarr, or zarrs-python."
        )
    raise ValueError(f"Unknown writer: {writer}")
