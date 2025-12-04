"""OME-Zarr v0.5 writing functionality.

This module provides functions to write OME-Zarr v0.5 stores using either
zarr-python or tensorstore backends.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ._bf2raw import Bf2Raw
from ._series import Series

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    from typing_extensions import Protocol, TypeAlias

    from ._image import Image
    from ._zarr_json import OMEMetadata

    ZarrBackend: TypeAlias = Literal["zarr", "tensorstore", "auto"]

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


# ######################## Public API ##########################################


def write_image(
    dest: str | PathLike,
    datasets: Sequence[ArrayLike],
    image: Image,
    *,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    backend: ZarrBackend = "auto",
    progress: bool = False,
) -> Path:
    """Write an OME-Zarr image with one or more resolution levels.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Zarr store.
    datasets : Sequence[ArrayLike]
        Array data for each resolution level, ordered from highest to lowest
        resolution. For single-resolution images, pass a list with one array.
    image : Image
        yaozarrs Image model with multiscales metadata. The number of
        datasets in the model must match the number of data arrays.
    chunks : tuple | "auto" | None
        Chunk shape. "auto" calculates optimal chunks (~4MB target),
        None uses full array shape (single chunk). Applied to each
        resolution level (clamped to array size).
    shards : tuple | None
        Shard shape for Zarr v3 sharding. If provided, enables sharded
        storage where each shard contains multiple chunks.
    backend : "zarr" | "tensorstore" | "auto"
        Backend for array writing. "auto" tries tensorstore first,
        then falls back to zarr-python.
    progress : bool
        Show progress bar during writing.

    Returns
    -------
    Path
        Path to created store.

    Examples
    --------
    >>> from yaozarrs import v05
    >>> import numpy as np
    >>>
    >>> data = np.random.rand(10, 256, 256).astype(np.float32)
    >>> image = v05.Image(
    ...     multiscales=[
    ...         v05.Multiscale(
    ...             axes=[
    ...                 v05.TimeAxis(name="t"),
    ...                 v05.SpaceAxis(name="y", unit="micrometer"),
    ...                 v05.SpaceAxis(name="x", unit="micrometer"),
    ...             ],
    ...             datasets=[
    ...                 v05.Dataset(
    ...                     path="0",
    ...                     coordinateTransformations=[
    ...                         v05.ScaleTransformation(scale=[1.0, 0.5, 0.5])
    ...                     ],
    ...                 )
    ...             ],
    ...         )
    ...     ]
    ... )
    >>> v05.write_image("output.zarr", [data], image)
    """
    dest_path = Path(dest)
    write_array = _get_write_func(backend)

    multiscale = image.multiscales[0]
    dim_names = [ax.name for ax in multiscale.axes]

    if len(datasets) != len(multiscale.datasets):
        raise ValueError(
            f"Number of data arrays ({len(datasets)}) must match "
            f"number of datasets in metadata ({len(multiscale.datasets)})"
        )

    # Create the zarr group with OME metadata
    _create_zarr_group(dest_path, image)

    # Write each resolution level
    for data, dataset_meta in zip(datasets, multiscale.datasets):
        resolved_chunks = _resolve_chunks(data, chunks)
        write_array(
            path=dest_path / dataset_meta.path,
            data=data,
            chunks=resolved_chunks,
            shard_shape=shards,
            dimension_names=dim_names,
            progress=progress,
        )

    return dest_path


def write_bioformats2raw(
    dest: str | PathLike,
    images: dict[str, tuple[Sequence[ArrayLike], Image]],
    *,
    ome_xml: str | None = None,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    backend: ZarrBackend = "auto",
    progress: bool = False,
) -> Path:
    """Write OME-Zarr with bioformats2raw layout for multiple series.

    Creates a store with the bioformats2raw layout:
    - Root zarr.json with bioformats2raw.layout attribute
    - OME/ directory with series metadata and optional METADATA.ome.xml
    - Each image in a numbered subdirectory (0/, 1/, etc.)

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Zarr store.
    images : dict[str, tuple[Sequence[ArrayLike], Image]]
        Mapping of series name -> (datasets, image_metadata).
        Series names become subdirectory names (typically "0", "1", etc.).
        Each datasets is a sequence of arrays (one per resolution level).
    ome_xml : str | None
        Optional OME-XML string for METADATA.ome.xml file.
    chunks : tuple | "auto" | None
        Chunk shape.
    shards : tuple | None
        Shard shape for Zarr v3 sharding.
    backend : "zarr" | "tensorstore" | "auto"
        Backend for array writing.
    progress : bool
        Show progress bar during writing.

    Returns
    -------
    Path
        Path to created store.

    Examples
    --------
    >>> from yaozarrs import v05
    >>> import numpy as np
    >>>
    >>> # Create two images
    >>> data1 = np.random.rand(256, 256).astype(np.float32)
    >>> data2 = np.random.rand(256, 256).astype(np.float32)
    >>>
    >>> def make_image(name):
    ...     return v05.Image(
    ...         multiscales=[
    ...             v05.Multiscale(
    ...                 name=name,
    ...                 axes=[
    ...                     v05.SpaceAxis(name="y", unit="micrometer"),
    ...                     v05.SpaceAxis(name="x", unit="micrometer"),
    ...                 ],
    ...                 datasets=[
    ...                     v05.Dataset(
    ...                         path="0",
    ...                         coordinateTransformations=[
    ...                             v05.ScaleTransformation(scale=[0.5, 0.5])
    ...                         ],
    ...                     )
    ...                 ],
    ...             )
    ...         ]
    ...     )
    >>>
    >>> images = {
    ...     "0": (data1, make_image("position_0")),
    ...     "1": (data2, make_image("position_1")),
    ... }
    >>> v05.write_bioformats2raw("output.zarr", images)
    """
    dest_path = Path(dest)

    # Write root zarr.json with bioformats2raw.layout
    bf2raw = Bf2Raw(bioformats2raw_layout=3)  # pyright: ignore[reportCallIssue]
    _create_zarr_group(dest_path, bf2raw)

    # Write OME/zarr.json with series list
    ome_path = dest_path / "OME"
    series_names = list(images.keys())
    series = Series(series=series_names)
    _create_zarr_group(ome_path, series)

    # Write METADATA.ome.xml if provided
    if ome_xml is not None:
        (ome_path / "METADATA.ome.xml").write_text(ome_xml)

    # Write each image
    for series_name, (datasets, image_model) in images.items():
        write_image(
            dest_path / series_name,
            datasets,
            image_model,
            chunks=chunks,
            shards=shards,
            backend=backend,
            progress=progress,
        )

    return dest_path


# ######################## Internal Helpers ####################################


def _create_zarr_group(
    dest_path: Path,
    ome_model: OMEMetadata,
    zarr_version: Literal[3] = 3,
) -> None:
    """Create a zarr group directory with OME metadata in zarr.json."""
    dest_path.mkdir(parents=True, exist_ok=True)
    meta_dict = ome_model.model_dump(mode="json", exclude_none=True)

    zarr_json = {
        "zarr_format": zarr_version,
        "node_type": "group",
        "attributes": {"ome": meta_dict},
    }
    (dest_path / "zarr.json").write_text(json.dumps(zarr_json, indent=2))


def _resolve_chunks(
    data: ArrayLike,
    chunk_shape: tuple[int, ...] | Literal["auto"] | None,
) -> tuple[int, ...]:
    """Resolve chunk shape based on user input."""
    if chunk_shape == "auto":
        return _calculate_auto_chunks(data.shape, data.dtype.itemsize)
    elif chunk_shape is None:
        return data.shape
    else:
        # Clamp to array shape
        return tuple(min(c, s) for c, s in zip(chunk_shape, data.shape))


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
    while np.prod(spatial_chunks) > target_elements and max(spatial_chunks) > 1:
        max_idx = spatial_chunks.index(max(spatial_chunks))
        spatial_chunks[max_idx] = max(1, spatial_chunks[max_idx] // 2)

    # Apply back
    for i, val in enumerate(spatial_chunks):
        chunks[spatial_start + i] = val

    return tuple(chunks)


# ######################## Array Writing Functions #############################


def _write_array_zarr(
    path: Path,
    data: Any,
    chunks: tuple[int, ...],
    shard_shape: tuple[int, ...] | None = None,
    dimension_names: list[str] | None = None,
    progress: bool = False,
    zarr_format: Literal[3] = 3,
) -> None:
    """Write array using zarr-python backend."""
    import zarr
    import zarr.storage
    from zarr.codecs import BloscCodec

    is_dask = hasattr(data, "compute")

    store = zarr.storage.LocalStore(str(path))
    compressor = BloscCodec(cname="zstd", clevel=3)

    arr = zarr.create_array(
        store,
        shape=data.shape,
        chunks=chunks,
        shards=shard_shape,
        dtype=data.dtype,
        compressors=compressor,
        dimension_names=dimension_names,
        zarr_format=zarr_format,
    )

    if is_dask:
        import dask.array as da

        if progress:
            from dask.diagnostics.progress import ProgressBar

            with ProgressBar():
                da.store(data, arr, lock=False)
        else:
            da.store(data, arr, lock=False)
    else:
        arr[:] = data


def _write_array_tensorstore(
    path: Path,
    data: Any,
    chunks: tuple[int, ...],
    shard_shape: tuple[int, ...] | None = None,
    dimension_names: list[str] | None = None,
    progress: bool = False,
    zarr_format: Literal[3] = 3,
) -> None:
    """Write array using tensorstore backend."""
    import tensorstore as ts

    if zarr_format == 3:
        zarr_driver = "zarr3"
    else:
        raise ValueError(f"Unsupported zarr_format: {zarr_format}")

    is_dask = hasattr(data, "compute")

    metadata: dict[str, Any] = {
        "shape": list(data.shape),
        "data_type": str(data.dtype),
        "chunk_grid": {
            "name": "regular",
            "configuration": {
                "chunk_shape": list(shard_shape) if shard_shape else list(chunks)
            },
        },
    }

    if dimension_names:
        metadata["dimension_names"] = dimension_names

    if shard_shape is not None:
        metadata["codecs"] = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": list(chunks),
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {
                            "name": "blosc",
                            "configuration": {"cname": "zstd", "clevel": 3},
                        },
                    ],
                },
            }
        ]
    else:
        metadata["codecs"] = [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 3}},
        ]

    spec: dict[str, Any] = {
        "driver": zarr_driver,
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": metadata,
        "create": True,
        "delete_existing": True,
    }
    store = ts.open(spec).result()

    if is_dask:
        if progress:
            from dask.diagnostics.progress import ProgressBar

            with ProgressBar():
                computed = data.compute()
        else:
            computed = data.compute()
        store[:].write(computed).result()
    else:
        store[:].write(data).result()


def _get_write_func(backend: ZarrBackend) -> Any:
    """Get the appropriate array write function for the backend."""
    if backend in {"tensorstore", "auto"}:
        if importlib.util.find_spec("tensorstore"):
            return _write_array_tensorstore
        elif backend == "tensorstore":
            raise ImportError(
                "tensorstore is required for the 'tensorstore' backend. "
                "Install with: pip install tensorstore"
            )
    if backend in {"zarr", "auto"}:
        if importlib.util.find_spec("zarr"):
            return _write_array_zarr
        raise ImportError(
            "zarr-python is required for the 'zarr' backend. "
            "Install with: pip install zarr"
        )
    if backend == "auto":
        raise ImportError(
            "No suitable backend found for OME-Zarr writing. "
            "Please install either tensorstore or zarr."
        )
    raise ValueError(f"Unknown backend: {backend}")
