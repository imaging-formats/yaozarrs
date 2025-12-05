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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ._bf2raw import Bf2Raw
from ._series import Series

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    import numpy as np
    from typing_extensions import TypeAlias

    from ._image import Image
    from ._zarr_json import OMEMetadata

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
        dimension_names: list[str] | None,  # = None,
        overwrite: bool,  # = False,
        compression: CompressionName,  # = "blosc-zstd",
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
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    progress: bool = False,
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
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
        Shard shape for Zarr v3 sharding.
    writer : "zarr" | "zarrs" | "tensorstore" | "auto" | CreateArrayFunc
        Writer for array creation. Can be:
        - "auto": tries tensorstore first, falls back to zarr-python
        - "zarr": use zarr-python
        - "zarrs": use zarrs-python (Rust-accelerated zarr)
        - "tensorstore": use tensorstore
        - Custom CreateArrayFunc: A function that creates array structures.
          See CreateArrayFunc protocol for the expected signature.
    progress : bool
        Show progress bar during writing (used by built-in writers only).
    overwrite : bool
        Whether to overwrite existing group (default: False).
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none"
        Compression codec to use. Choose based on your priorities:

        - **"blosc-zstd"** (default): Good balance for imaging data
            - Uses: blosc meta-compressor with zstd + shuffle filter
            - Speed: Good (multi-threaded)
            - Compression: Excellent for correlated imaging data
              (2-3x better than plain zstd)
            - Use when: You want good compression with reasonable speed
        - **"blosc-lz4"**: Fastest with good compression
            - Uses: blosc meta-compressor with lz4 + shuffle filter
            - Speed: Fastest compression/decompression
            - Compression: Good (shuffle helps with imaging data)
            - Use when: Speed is critical, file size is less important
        - **"zstd"**: Simple, reliable compression
            - Uses: zstd at level 3 (zarr-python's default)
            - Speed: Good
            - Compression: Good, but misses shuffle benefits for imaging data
            - Use when: You want standard compression without blosc's complexity
        - **"none"**: No compression
            - Uses: Raw bytes only
            - Speed: Fastest I/O (no CPU overhead)
            - Compression: None (largest files)
            - Use when: Local SSD storage, need maximum read speed

    Returns
    -------
    Path
        Path to created store.

    Examples
    --------
    Simple usage with defaults:

    >>> from yaozarrs import v05
    >>> import numpy as np
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> # Create sample data
    >>> data = np.random.rand(10, 256, 256).astype("float32")
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
    >>>
    >>> # Write to temporary directory
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = v05.write_image(Path(tmpdir) / "output.zarr", [data], image)
    ...     assert path.exists()
    ...     assert (path / "zarr.json").exists()
    ...     assert (path / "0" / "zarr.json").exists()

    Custom array creator with full zarr-python control:

    >>> def my_array_creator(
    ...     path,
    ...     shape,
    ...     dtype,
    ...     chunks,
    ...     shards=None,
    ...     dimension_names=None,
    ...     overwrite=False,
    ...     compression="blosc-zstd",
    ... ):
    ...     import zarr
    ...
    ...     arr = zarr.create_array(
    ...         str(path),
    ...         shape=shape,
    ...         chunks=chunks,
    ...         shards=shards,
    ...         dtype=dtype,
    ...         dimension_names=dimension_names,
    ...         overwrite=overwrite,
    ...     )
    ...     return arr
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = v05.write_image(
    ...         Path(tmpdir) / "custom.zarr", [data], image, writer=my_array_creator
    ...     )
    ...     assert path.exists()

    For cloud storage or other advanced features, pass a custom writer
    function that configures the implementation exactly as needed.
    """
    if len(image.multiscales) != 1:
        raise NotImplementedError("Image must have exactly one multiscale")

    multiscale = image.multiscales[0]
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
        Shard shape for Zarr v3 sharding. Passed to both built-in and
        custom writers.
    writer : "zarr" | "zarrs" | "tensorstore" | "auto" | CreateArrayFunc
        Writer for array creation. Can be a string ("zarr", "zarrs",
        "tensorstore", "auto") or a custom CreateArrayFunc. See write_image
        for details.
    progress : bool
        Show progress bar during writing (used by built-in writers only).
    overwrite : bool
        Whether to overwrite existing groups (default: False).
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none"
        Compression codec to use. See write_image for detailed descriptions.
        Default is "blosc-zstd" (best balance for imaging data).

    Returns
    -------
    Path
        Path to created store.

    Examples
    --------
    >>> from yaozarrs import v05
    >>> import numpy as np
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> # Create two images
    >>> data1 = np.random.rand(256, 256).astype("float32")
    >>> data2 = np.random.rand(256, 256).astype("float32")
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
    ...     "0": ([data1], make_image("position_0")),
    ...     "1": ([data2], make_image("position_1")),
    ... }
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = v05.write_bioformats2raw(Path(tmpdir) / "output.zarr", images)
    ...     assert path.exists()
    ...     assert (path / "OME" / "zarr.json").exists()
    ...     assert (path / "0" / "zarr.json").exists()
    ...     assert (path / "1" / "zarr.json").exists()
    """
    # Extract shapes and dtypes from data arrays for each series
    images_spec = {}
    data_mapping = {}
    for series_name, (datasets, image_model) in images.items():
        shapes = {}
        dtypes = {}
        multiscale = image_model.multiscales[0]
        for data_array, dataset_meta in zip(datasets, multiscale.datasets):
            shapes[dataset_meta.path] = data_array.shape
            dtypes[dataset_meta.path] = data_array.dtype
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


def create_array(
    path: str | PathLike,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    dimension_names: list[str] | None = None,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
) -> Any:
    """
    Create a single zarr array without writing data.

    This is the low-level function for creating individual arrays. Use
    `prepare_image` or `prepare_bioformats2raw` for creating complete
    OME-Zarr structures.

    Parameters
    ----------
    path : str | PathLike
        Path where the array will be created.
    shape : tuple[int, ...]
        Array shape.
    dtype : dtype
        Data type (e.g., 'float32', 'uint16', 'int8', etc.).
        String dtypes are preferred for clarity.
    dimension_names : list[str] | None
        Names for each dimension (e.g., ["c", "y", "x"]).
    chunks : tuple | "auto" | None
        Chunk shape. "auto" calculates optimal chunks (~4MB target),
        None uses full array shape (single chunk).
    shards : tuple | None
        Shard shape for Zarr v3 sharding.
    writer : "zarr" | "zarrs" | "tensorstore" | "auto" | CreateArrayFunc
        Writer backend or custom CreateArrayFunc (default: "auto").
    overwrite : bool
        Whether to overwrite existing array (default: False).
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none"
        Compression codec to use (default: "blosc-zstd").

    Returns
    -------
    Array object (zarr.Array or tensorstore.TensorStore depending on writer)
    that supports numpy-style indexing for writing.

    Examples
    --------
    >>> import numpy as np
    >>> from yaozarrs import v05
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Create array structure
    ...     arr = v05.create_array(
    ...         Path(tmpdir) / "data.zarr",
    ...         shape=(100, 512, 512),
    ...         dtype="float32",
    ...         dimension_names=["z", "y", "x"],
    ...         chunks=(1, 512, 512),
    ...     )
    ...     # Write data plane by plane
    ...     for i in range(100):
    ...         plane = np.random.rand(512, 512).astype("float32")
    ...         arr[i] = plane
    """
    # Normalize dtype to ensure consistent handling
    try:
        import numpy as np

        dtype = np.dtype(dtype)
    except ImportError:
        # If numpy not available, assume dtype is already valid
        pass

    # Get create function
    create_func = _get_create_func(writer) if isinstance(writer, str) else writer
    if not isinstance(create_func, CreateArrayFunc):
        raise TypeError("writer must be a string or a CreateArrayFunc")

    # Resolve chunks
    resolved_chunks = _resolve_chunks(shape, dtype, chunks)

    # Create and return array
    return create_func(
        path=Path(path),
        shape=shape,
        dtype=dtype,
        chunks=resolved_chunks,
        shards=shards,
        dimension_names=dimension_names,
        overwrite=overwrite,
        compression=compression,
    )


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
    """
    Create OME-Zarr Image structure (metadata + arrays) without writing data.

    This creates the zarr group with Image metadata and all array structures,
    returning array objects that you can write to incrementally.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Zarr store.
    image : Image
        yaozarrs Image model with multiscales metadata.
    shapes : dict[str, tuple[int, ...]]
        Mapping of dataset path -> shape. Keys must match
        image.multiscales[0].datasets[].path
    dtypes : dict[str, dtype]
        Mapping of dataset path -> dtype (e.g., 'float32', 'uint16').
    chunks : tuple | "auto" | None
        Chunk shape (default: "auto").
    shards : tuple | None
        Shard shape for Zarr v3 sharding.
    writer : "zarr" | "zarrs" | "tensorstore" | "auto" | CreateArrayFunc
        Writer backend or custom CreateArrayFunc (default: "auto").
    overwrite : bool
        Whether to overwrite existing group (default: False).
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none"
        Compression codec to use (default: "blosc-zstd").

    Returns
    -------
    root_path : Path
        Path to created store.
    arrays : dict[str, Array]
        Mapping of dataset path -> array object.
        Keys are dataset paths like "0", "1", "2".

    Examples
    --------
    Single resolution, write plane by plane:

    >>> from yaozarrs import v05
    >>> import numpy as np
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> image = v05.Image(
    ...     multiscales=[
    ...         v05.Multiscale(
    ...             axes=[
    ...                 v05.ChannelAxis(name="c"),
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
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     root_path, arrays = v05.prepare_image(
    ...         Path(tmpdir) / "output.zarr",
    ...         image=image,
    ...         shapes={"0": (2, 512, 512)},
    ...         dtypes={"0": "float32"},
    ...         chunks=(1, 512, 512),
    ...     )
    ...     # Write channel by channel
    ...     for c in range(2):
    ...         arrays["0"][c] = np.random.rand(512, 512).astype("float32")

    Multi-resolution pyramid:

    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     pyramid_image = v05.Image(
    ...         multiscales=[
    ...             v05.Multiscale(
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
    ...                     ),
    ...                     v05.Dataset(
    ...                         path="1",
    ...                         coordinateTransformations=[
    ...                             v05.ScaleTransformation(scale=[1.0, 1.0])
    ...                         ],
    ...                     ),
    ...                 ],
    ...             )
    ...         ]
    ...     )
    ...     root_path, arrays = v05.prepare_image(
    ...         Path(tmpdir) / "pyramid.zarr",
    ...         image=pyramid_image,
    ...         shapes={"0": (1024, 1024), "1": (512, 512)},
    ...         dtypes={"0": "uint16", "1": "uint16"},
    ...     )
    ...     # arrays["0"] is full resolution, arrays["1"] is downsampled
    ...     arrays["0"][:] = np.random.randint(0, 1000, (1024, 1024), dtype=np.uint16)
    ...     arrays["1"][:] = np.random.randint(0, 1000, (512, 512), dtype=np.uint16)
    """
    if len(image.multiscales) != 1:
        raise NotImplementedError("Image must have exactly one multiscale")
    multiscale = image.multiscales[0]

    # Validate inputs
    if set(shapes.keys()) != set(dtypes.keys()):
        raise ValueError("shapes and dtypes must have the same keys")

    dataset_paths = {ds.path for ds in multiscale.datasets}
    if set(shapes.keys()) != dataset_paths:
        raise ValueError(
            f"shapes keys {set(shapes.keys())} must match dataset paths {dataset_paths}"
        )

    # Get create function
    create_func = _get_create_func(writer) if isinstance(writer, str) else writer
    if not isinstance(create_func, CreateArrayFunc):
        raise TypeError("writer must be a string or a CreateArrayFunc")

    # Create zarr group with Image metadata
    dest_path = Path(dest)
    _create_zarr_group(dest_path, image, overwrite)

    # Create arrays for each dataset
    arrays = {}
    dimension_names = [ax.name for ax in multiscale.axes]

    for dataset_meta in multiscale.datasets:
        ds_path = dataset_meta.path
        shape = shapes[ds_path]
        dtype = dtypes[ds_path]

        # Normalize dtype to ensure consistent handling
        try:
            import numpy as np

            dtype = np.dtype(dtype)
        except ImportError:
            # If numpy not available, assume dtype is already valid
            pass

        # Resolve chunks for this shape
        resolved_chunks = _resolve_chunks(shape, dtype, chunks)

        # Create array
        arr = create_func(
            path=dest_path / ds_path,
            shape=shape,
            dtype=dtype,
            chunks=resolved_chunks,
            shards=shards,
            dimension_names=dimension_names,
            overwrite=overwrite,
            compression=compression,
        )
        arrays[ds_path] = arr

    return dest_path, arrays


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
    """
    Create bioformats2raw structure without writing data.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Zarr store.
    images : dict[str, tuple[Image, shapes_dict, dtypes_dict]]
        Mapping of series_name -> (image_metadata, shapes, dtypes).
        shapes and dtypes are dicts mapping dataset path -> shape/dtype.
    ome_xml : str | None
        Optional OME-XML string for METADATA.ome.xml file.
    chunks : tuple | "auto" | None
        Chunk shape (default: "auto").
    shards : tuple | None
        Shard shape for Zarr v3 sharding.
    writer : "zarr" | "zarrs" | "tensorstore" | "auto" | CreateArrayFunc
        Writer backend or custom CreateArrayFunc (default: "auto").
    overwrite : bool
        Whether to overwrite existing groups (default: False).
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none"
        Compression codec to use (default: "blosc-zstd").

    Returns
    -------
    root_path : Path
        Path to created store.
    arrays : dict[str, Array]
        Flat mapping of "series/dataset" path -> array object.
        Keys are paths like "0/0", "0/1", "1/0", "1/1".

    Examples
    --------
    >>> from yaozarrs import v05
    >>> import numpy as np
    >>> from pathlib import Path
    >>> import tempfile
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
    >>> images_spec = {
    ...     "0": (make_image("series_0"), {"0": (256, 256)}, {"0": "float32"}),
    ...     "1": (make_image("series_1"), {"0": (256, 256)}, {"0": "float32"}),
    ... }
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     root_path, arrays = v05.prepare_bioformats2raw(
    ...         Path(tmpdir) / "output.zarr", images_spec
    ...     )
    ...     # Write to series 0
    ...     arrays["0/0"][:] = np.random.rand(256, 256).astype("float32")
    ...     # Write to series 1
    ...     arrays["1/0"][:] = np.random.rand(256, 256).astype("float32")
    """
    dest_path = Path(dest)

    # Create root zarr.json with bioformats2raw.layout
    bf2raw = Bf2Raw(bioformats2raw_layout=3)  # type: ignore
    _create_zarr_group(dest_path, bf2raw, overwrite)

    # Create OME/zarr.json with series list
    ome_path = dest_path / "OME"
    series = Series(series=list(images))
    _create_zarr_group(ome_path, series, overwrite)

    # Write METADATA.ome.xml if provided
    if ome_xml is not None:
        (ome_path / "METADATA.ome.xml").write_text(ome_xml)

    # Create arrays for each series
    all_arrays = {}
    for series_name, (image_model, shapes_dict, dtypes_dict) in images.items():
        _, series_arrays = prepare_image(
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


def _create_zarr_group(
    dest_path: Path,
    ome_model: OMEMetadata,
    overwrite: bool = False,
) -> None:
    """Create a zarr group directory with OME metadata in zarr.json."""
    dest_path.mkdir(parents=True, exist_ok=True)

    zarr_json_path = dest_path / "zarr.json"
    if not overwrite and zarr_json_path.exists():
        raise FileExistsError(
            f"Zarr group already exists at {dest_path}. "
            "Use overwrite=True to replace it."
        )

    meta_dict = ome_model.model_dump(mode="json", exclude_none=True)
    zarr_json = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"ome": meta_dict},
    }
    zarr_json_path.write_text(json.dumps(zarr_json, indent=2))


def _resolve_chunks(
    shape: tuple[int, ...],
    dtype: Any,
    chunk_shape: tuple[int, ...] | Literal["auto"] | None,
) -> tuple[int, ...]:
    """Resolve chunk shape based on user input."""
    if chunk_shape == "auto":
        # Need dtype.itemsize for auto calculation, so normalize it
        try:
            import numpy as np

            dtype = np.dtype(dtype)
        except ImportError:
            # Assume dtype already has itemsize attribute
            pass
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

    arr = zarr.create_array(
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
    return arr


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


def _get_create_func(writer: str) -> Any:
    """Get the appropriate array create function for the writer."""
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
