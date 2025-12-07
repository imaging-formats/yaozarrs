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
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypedDict,
    cast,
    overload,
    runtime_checkable,
)

from typing_extensions import Self

from yaozarrs.v05._bf2raw import Bf2Raw
from yaozarrs.v05._series import Series
from yaozarrs.v05._well import FieldOfView, Well, WellDef

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from os import PathLike

    import numpy as np
    import tensorstore
    import zarr
    from typing_extensions import Literal, TypeAlias

    from yaozarrs.v05._image import Image
    from yaozarrs.v05._plate import Plate
    from yaozarrs.v05._zarr_json import OMEMetadata

    WriterName = Literal["zarr", "tensorstore", "auto"]
    ZarrWriter: TypeAlias = WriterName | "CreateArrayFunc"
    CompressionName = Literal["blosc-zstd", "blosc-lz4", "zstd", "none"]
    AnyZarrArray: TypeAlias = zarr.Array | tensorstore.TensorStore
    DTypeLike: TypeAlias = str | np.dtype[Any]
    ShapeLike: TypeAlias = tuple[int, ...]

    class WriteKwargs(TypedDict, total=False):
        chunk_shape: tuple[int, ...] | Literal["auto"] | None
        """Read chunk shape. "auto" is passed to backend."""
        shard_shape: tuple[int, ...] | None
        """Shard shape (storage chunks). chunk_shape must evenly divide shard_shape."""
        fill_value: Any
        """Fill value for uninitialized regions."""
        compression: CompressionName | None
        """Compression codec."""
        compression_level: int | None
        """Compression level (0-9 for most, 0-22 for zstd)."""
        overwrite: bool
        """Whether to overwrite existing array at this location."""

    class ArrayLike(Protocol):
        """Protocol for array-like objects (numpy arrays, dask arrays, etc.)."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Shape of the array."""
            ...

        @property
        def dtype(self) -> Any:
            """Data type of the array."""
            ...

    ArraySpec: TypeAlias = ArrayLike | tuple[DTypeLike, ShapeLike]
    ImageWithDatasets: TypeAlias = tuple[Image, Sequence[ArrayLike]]


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

# ------------------------ High Level Write Functions --------------------------


def write_image(
    dest: str | PathLike,
    image: Image,
    datasets: Sequence[ArrayLike],
    *,
    writer: ZarrWriter = "auto",
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write an OME-Zarr v0.5 Image group with data.

    This is the high-level function for writing a complete OME-Zarr image.
    It creates the Zarr group hierarchy, writes metadata, and stores data
    in a single call.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Zarr group. Will be created if it doesn't exist.
    image : Image
        OME-Zarr Image metadata model. Must have exactly one multiscale, with
        one Dataset entry per array in `datasets`.
    datasets : Sequence[ArrayLike]
        Data arrays to write (numpy, dask, or any array with shape/dtype).
        Must be in the same order as `image.multiscales[0].datasets`.
        For a multiscale pyramid, provide one array per resolution level.
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend to use for writing arrays. "auto" prefers tensorstore if
        available, otherwise falls back to zarr-python. Pass a custom function
        matching the `CreateArrayFunc` protocol for custom backends.
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape for storage. "auto" (default) calculates ~4MB chunks with
        non-spatial dims set to 1. None uses the full array shape (single chunk).
        Tuple values are clamped to the array shape.
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding codec. None (default) disables sharding.
    overwrite : bool, optional
        If True, overwrite existing Zarr group at `dest`. Default is False.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. "blosc-zstd" (default) provides good compression with
        shuffle filter. "zstd" uses raw zstd without blosc container.
    progress : bool, optional
        Show progress bar when writing dask arrays. Default is False.

    Returns
    -------
    Path
        Path to the created Zarr group.

    Raises
    ------
    NotImplementedError
        If the Image model has multiple multiscales (not yet supported).
    ValueError
        If the number of datasets doesn't match the metadata.
    FileExistsError
        If `dest` exists and `overwrite` is False.
    ImportError
        If no suitable writer backend is installed.

    Examples
    --------
    Write a simple 3D image (CYX):

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from yaozarrs import v05
    >>> from yaozarrs.write.v05 import write_image
    >>>
    >>> data = np.zeros((2, 64, 64), dtype=np.uint16)
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
    >>> dest = Path(tmpdir) / "example.zarr"
    >>> result = write_image(dest, image, [data])
    >>> assert result.exists()

    See Also
    --------
    prepare_image : Create arrays without writing data (for custom write logic).
    write_bioformats2raw : Write multi-series bioformats2raw layout.
    """
    if len(image.multiscales) != 1:
        raise NotImplementedError("Image must have exactly one multiscale")

    multiscale = image.multiscales[0]

    # Create arrays using prepare_image (handles both built-in and custom)
    dest_path, arrays = prepare_image(
        dest,
        image,
        datasets,
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


def write_plate(
    dest: str | PathLike,
    images: Mapping[tuple[str, str, str], ImageWithDatasets],
    *,
    plate: Plate | dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write an OME-Zarr v0.5 Plate group with data.

    This is the high-level function for writing a complete OME-Zarr plate.
    It creates the plate hierarchy (plate/wells/fields), writes metadata,
    and stores image data in a single call.

    The plate structure::

        dest/
        ├── zarr.json          # Plate metadata
        ├── A/
        │   ├── 1/
        │   │   ├── zarr.json  # Well metadata (auto-generated)
        │   │   ├── 0/         # Field 0
        │   │   │   ├── zarr.json  # Image metadata
        │   │   │   └── 0/     # dataset arrays
        │   │   └── 1/         # Field 1 (if multiple fields)
        │   └── 2/
        └── B/
            └── ...

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Plate Zarr group.
    images : Mapping[tuple[str, str, str], ImageWithDatasets]
        Mapping of `{(row, col, fov) -> (image_model, [datasets, ...])}`.
        Each tuple key specifies (row_name, column_name, field_of_view) like
        ("A", "1", "0"). Row and column names are auto-extracted from the keys.
    plate : Plate | dict[str, Any] | None, optional
        Optional plate metadata. Can be:
        - None (default): Auto-generate from images dict keys
        - dict: Merge with auto-generated metadata (user values take precedence)
        - Plate: Use as-is (must match images dict)
        Common dict keys: 'name', 'acquisitions', 'field_count'.
        Auto-generated: 'rows', 'columns', 'wells'.
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend to use for writing arrays. Default is "auto".
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape for all arrays. See `write_image` for details.
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding. Default is None.
    overwrite : bool, optional
        If True, overwrite existing Zarr groups. Default is False.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. Default is "blosc-zstd".
    progress : bool, optional
        Show progress bar when writing dask arrays. Default is False.

    Returns
    -------
    Path
        Path to the created Plate Zarr group.

    Raises
    ------
    ValueError
        If image keys don't match plate wells, or if datasets don't match metadata.
    FileExistsError
        If `dest` exists and `overwrite` is False.
    ImportError
        If no suitable writer backend is installed.

    Examples
    --------
    Write a simple 2x2 plate with auto-generated metadata:

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from yaozarrs import v05
    >>> from yaozarrs.write.v05 import write_plate
    >>>
    >>> # Create image metadata (same for all fields)
    >>> def make_image():
    ...     return v05.Image(
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
    ...                     )
    ...                 ],
    ...             )
    ...         ]
    ...     )
    >>>
    >>> # Rows, columns, and wells are auto-generated from the images dict!
    >>> images = {
    ...     ("A", "1", "0"): (make_image(), [np.zeros((64, 64), dtype=np.uint16)]),
    ...     ("A", "2", "0"): (make_image(), [np.zeros((64, 64), dtype=np.uint16)]),
    ...     ("B", "1", "0"): (make_image(), [np.zeros((64, 64), dtype=np.uint16)]),
    ...     ("B", "2", "0"): (make_image(), [np.zeros((64, 64), dtype=np.uint16)]),
    ... }
    >>>
    >>> dest = Path(tmpdir) / "my_plate.zarr"
    >>> result = write_plate(dest, images)
    >>> assert (result / "A" / "1" / "0" / "zarr.json").exists()
    >>>
    >>> # Or add custom metadata like a name
    >>> result2 = write_plate(
    ...     dest,
    ...     images,
    ...     plate={"name": "My Experiment"},
    ...     overwrite=True,
    ... )  # doctest: +ELLIPSIS

    See Also
    --------
    PlateBuilder : Builder class for incremental well/field writing.
    write_image : Write a single Image group.
    """
    # Merge user-provided plate metadata with auto-generated
    plate_obj = _merge_plate_metadata(images, plate)

    # Use PlateBuilder to handle the writing
    builder = PlateBuilder(
        dest,
        plate=plate_obj,
        writer=writer,
        chunks=chunks,
        shards=shards,
        overwrite=overwrite,
        compression=compression,
    )

    # Group images by well: {(row, col): {fov: (Image, datasets)}}
    wells_data: dict[tuple[str, str], dict[str, ImageWithDatasets]] = {}
    for (row, col, fov), image_data in images.items():
        wells_data.setdefault((row, col), {})[fov] = image_data

    # Write each well with all its fields
    for (row, col), fields_data in wells_data.items():
        builder.write_well(row=row, col=col, fields=fields_data, progress=progress)

    return builder.root_path


def write_bioformats2raw(
    dest: str | PathLike,
    # mapping of {series_name -> ( Image, [datasets, ...] )}
    images: Mapping[str, ImageWithDatasets],
    *,
    ome_xml: str | None = None,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    progress: bool = False,
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
) -> Path:
    """Write a bioformats2raw-layout OME-Zarr with multiple series.

    The bioformats2raw layout is a convention for storing multiple images
    (series) in a single Zarr hierarchy. It includes a root group with
    `bioformats2raw.layout` version, an `OME/` group with series metadata,
    and each series as a separate Image subgroup.

    This is the high-level function for writing all series at once. For
    incremental writes, use `Bf2RawBuilder` directly.

    Writes the following structure:

        dest/
        ├── zarr.json              # root: attributes["ome"]["bioformats2raw.layout"]
        ├── 0/                     # first series (images["0"])
        │   ├── zarr.json          # Image metadata (images["0"][0])
        │   ├── 0/                 # first resolution level
        │   │   ├── zarr.json      # array metadata
        │   │   └── c/             # chunks directory
        │   └── 1/                 # second resolution level (if multiscale)
        │       ├── zarr.json
        │       └── c/
        ├── 1/                     # second series (images["1"])
        │   └── ...
        └── OME/
            ├── zarr.json          # attributes["ome"]["series"] = ["0", "1", ...]
            └── METADATA.ome.xml   # optional OME-XML (if ome_xml provided)

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the root Zarr group.
    images : dict[str, ImageWithDatasets]
        Mapping of `{series_name -> (image_model, [datasets, ...])}`.
        Each series name (e.g., "0", "1") becomes a subgroup in the root group, with
        the Image model defining the zarr.json and the datasets providing the data
        arrays.
    ome_xml : str | None, optional
        OME-XML string to store as `OME/METADATA.ome.xml`.
        Useful for preserving full metadata from converted files.
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape for all arrays. See `write_image` for details.
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding. None disables sharding.
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend to use for writing arrays.
    progress : bool, optional
        Show progress bar when writing dask arrays. Default is False.
    overwrite : bool, optional
        If True, overwrite existing Zarr groups. Default is False.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. Default is "blosc-zstd".

    Returns
    -------
    Path
        Path to the root Zarr group.

    Raises
    ------
    FileExistsError
        If `dest` exists and `overwrite` is False.
    ValueError
        If any series has mismatched datasets/metadata.
    ImportError
        If no suitable writer backend is installed.

    Examples
    --------
    Write a multi-series OME-Zarr:

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from yaozarrs import v05
    >>> from yaozarrs.write.v05 import write_bioformats2raw
    >>>
    >>> def make_image():
    ...     return v05.Image(
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
    ...                     )
    ...                 ],
    ...             )
    ...         ]
    ...     )
    >>> images = {
    ...     "0": (make_image(), [np.zeros((64, 64), dtype=np.uint16)]),
    ...     "1": (make_image(), [np.zeros((32, 32), dtype=np.uint16)]),
    ... }
    >>> dest = Path(tmpdir) / "multi_series.zarr"
    >>> result = write_bioformats2raw(dest, images)
    >>> (result / "OME" / "zarr.json").exists()
    True
    >>> (result / "0" / "zarr.json").exists()
    True

    See Also
    --------
    Bf2RawBuilder : Builder class for incremental series writing.
    write_image : Write a single Image group.
    """
    builder = Bf2RawBuilder(
        dest,
        ome_xml=ome_xml,
        writer=writer,
        chunks=chunks,
        shards=shards,
        overwrite=overwrite,
        compression=compression,
    )

    for series_name, (image_model, datasets) in images.items():
        builder.write_image(series_name, image_model, datasets, progress=progress)

    return builder.root_path


# ------------------------ Lower Level Prepare Functions --------------------------


@overload
def prepare_image(
    dest: str | PathLike,
    image: Image,
    datasets: Sequence[ArraySpec],
    *,
    writer: Literal["zarr"],
    chunks: tuple[int, ...] | Literal["auto"] | None = ...,
    shards: tuple[int, ...] | None = ...,
    overwrite: bool = ...,
    compression: CompressionName = ...,
) -> tuple[Path, dict[str, zarr.Array]]: ...
@overload
def prepare_image(
    dest: str | PathLike,
    image: Image,
    datasets: Sequence[ArraySpec],
    *,
    writer: Literal["tensorstore"],
    chunks: tuple[int, ...] | Literal["auto"] | None = ...,
    shards: tuple[int, ...] | None = ...,
    overwrite: bool = ...,
    compression: CompressionName = ...,
) -> tuple[Path, dict[str, tensorstore.TensorStore]]: ...
@overload
def prepare_image(
    dest: str | PathLike,
    image: Image,
    datasets: Sequence[ArraySpec],
    *,
    writer: Literal["auto"] | CreateArrayFunc = ...,
    chunks: tuple[int, ...] | Literal["auto"] | None = ...,
    shards: tuple[int, ...] | None = ...,
    overwrite: bool = ...,
    compression: CompressionName = ...,
) -> tuple[Path, dict[str, AnyZarrArray]]: ...


def prepare_image(
    dest: str | PathLike,
    image: Image,
    datasets: Sequence[ArraySpec],
    *,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
) -> tuple[Path, dict[str, Any]]:
    """Create OME-Zarr v0.5 Image structure and return array handles for writing.

    This is a lower-level function that creates the Zarr group hierarchy and
    empty arrays, but does not write data. Use this when you need custom control
    over how data is written (e.g., chunk-by-chunk streaming, parallel writes).

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Zarr group.
    image : Image
        OME-Zarr Image metadata model.
    datasets : Sequence[ArraySpec]
        Specifications for each dataset. Can be either:

        - Array-like objects with `shape` and `dtype` attributes
          (numpy arrays, dask arrays, etc.)
        - Tuples of `(dtype, shape)` for specifying without actual data

        Must match the order of `image.multiscales[0].datasets`.
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape. See `write_image` for details.
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding.
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend for creating arrays. When you specify "zarr" or "tensorstore",
        the return type is narrowed to the specific array type.
    overwrite : bool, optional
        If True, overwrite existing Zarr group. Default is False.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. Default is "blosc-zstd".

    Returns
    -------
    tuple[Path, dict[str, Array]]
        A tuple of (path, arrays) where `arrays` maps dataset paths (e.g., "0")
        to array objects. The array type depends on the writer:

        - `writer="zarr"`: Returns `dict[str, zarr.Array]`
        - `writer="tensorstore"`: Returns `dict[str, tensorstore.TensorStore]`
        - `writer="auto"`: Returns whichever is available

    Raises
    ------
    NotImplementedError
        If the Image model has multiple multiscales.
    ValueError
        If the number of dataset specs doesn't match the metadata.
    TypeError
        If a dataset spec is not ArrayLike or (dtype, shape) tuple.
    FileExistsError
        If `dest` exists and `overwrite` is False.
    ImportError
        If no suitable writer backend is installed.

    Examples
    --------
    Create arrays and write data in chunks:

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from yaozarrs import v05
    >>> from yaozarrs.write.v05 import prepare_image
    >>> image = v05.Image(
    ...     multiscales=[
    ...         v05.Multiscale(
    ...             axes=[
    ...                 v05.SpaceAxis(name="y", unit="micrometer"),
    ...                 v05.SpaceAxis(name="x", unit="micrometer"),
    ...             ],
    ...             datasets=[
    ...                 v05.Dataset(
    ...                     path="0",
    ...                     coordinateTransformations=[
    ...                         v05.ScaleTransformation(scale=[0.5, 0.5])
    ...                     ],
    ...                 )
    ...             ],
    ...         )
    ...     ]
    ... )
    >>> # Prepare with just shape/dtype (no data yet)
    >>> dest = Path(tmpdir) / "prepared.zarr"
    >>> path, arrays = prepare_image(dest, image, [(np.dtype("uint16"), (64, 64))])
    >>> arrays["0"][:] = np.zeros((64, 64), dtype=np.uint16)
    >>> assert path.exists()

    See Also
    --------
    write_image : High-level function that writes data immediately.
    Bf2RawBuilder.prepare : Prepare multiple series at once.
    """
    if len(image.multiscales) != 1:
        raise NotImplementedError("Image must have exactly one multiscale")

    multiscale = image.multiscales[0]

    if len(datasets) != len(multiscale.datasets):
        raise ValueError(
            f"Number of data arrays ({len(datasets)}) must match "
            f"number of datasets in metadata ({len(multiscale.datasets)})"
        )

    # Get create function
    create_func = _get_create_func(writer)

    # Create zarr group with Image metadata
    dest_path = Path(dest)
    _create_zarr3_group(dest_path, image, overwrite)

    dimension_names = [ax.name for ax in multiscale.axes]

    # Create arrays for each dataset
    arrays = {}
    for array_spec, dataset_meta in zip(datasets, multiscale.datasets):
        ds_path = dataset_meta.path
        if isinstance(array_spec, tuple) and len(array_spec) == 2:
            dtype, shape = array_spec
        elif hasattr(array_spec, "dtype") and hasattr(array_spec, "shape"):
            dtype = array_spec.dtype
            shape = array_spec.shape
        else:
            raise TypeError(
                f"Dataset spec for path '{ds_path}' must be ArrayLike or ArraySpec"
            )
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


class Bf2RawBuilder:
    """Builder for bioformats2raw layout hierarchies.

    The bioformats2raw layout is a convention for storing multiple OME-Zarr
    images in a single hierarchy. It includes:

    - A root group with `bioformats2raw.layout` version attribute
    - An `OME/` subgroup listing all series names
    - Each series as a separate Image subgroup (e.g., `0/`, `1/`)
    - Optional `OME/METADATA.ome.xml` with full OME-XML metadata

    This builder supports two workflows:

    1. **Immediate write** (simpler): Use `write_image()` to write each series
       with its data immediately. The builder manages root structure and series
       list automatically.

    2. **Prepare-only** (flexible): Use `add_series()` to register all series,
       then `prepare()` to create the hierarchy with empty arrays. Write data
       to the returned arrays yourself.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the root Zarr group.
    ome_xml : str | None, optional
        Original OME-XML string to store as `OME/METADATA.ome.xml`.
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend to use for writing arrays. Default is "auto".
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape for all arrays. Default is "auto".
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding. Default is None (no sharding).
    overwrite : bool, optional
        If True, overwrite existing groups. Default is False.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. Default is "blosc-zstd".

    Examples
    --------
    **Immediate write workflow:**

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from yaozarrs import v05
    >>> from yaozarrs.write.v05 import Bf2RawBuilder
    >>> def make_image():
    ...     return v05.Image(
    ...         multiscales=[
    ...             v05.Multiscale(
    ...                 axes=[v05.SpaceAxis(name="y"), v05.SpaceAxis(name="x")],
    ...                 datasets=[
    ...                     v05.Dataset(
    ...                         path="0",
    ...                         coordinateTransformations=[
    ...                             v05.ScaleTransformation(scale=[1.0, 1.0])
    ...                         ],
    ...                     )
    ...                 ],
    ...             )
    ...         ]
    ...     )
    >>> dest = Path(tmpdir) / "builder_immediate.zarr"
    >>> builder = Bf2RawBuilder(dest)
    >>> builder.write_image("0", make_image(), [np.zeros((32, 32), dtype=np.uint16)])
    <Bf2RawBuilder: 1 images>
    >>> builder.write_image("1", make_image(), [np.zeros((16, 16), dtype=np.uint16)])
    <Bf2RawBuilder: 2 images>
    >>> assert (dest / "0" / "zarr.json").exists()

    **Prepare-only workflow:**

    >>> dest2 = Path(tmpdir) / "builder_prepare.zarr"
    >>> builder2 = Bf2RawBuilder(dest2)
    >>> data1 = np.zeros((32, 32), dtype=np.uint16)
    >>> data2 = np.zeros((16, 16), dtype=np.uint16)
    >>> builder2.add_series("0", make_image(), [data1])
    <Bf2RawBuilder: 1 images>
    >>> builder2.add_series("1", make_image(), [data2])
    <Bf2RawBuilder: 2 images>
    >>> path, arrays = builder2.prepare()
    >>> arrays["0/0"][:] = data1  # Write data yourself
    >>> arrays["1/0"][:] = data2
    >>> assert path.exists()

    See Also
    --------
    write_bioformats2raw : High-level function to write all series at once.
    """

    def __init__(
        self,
        dest: str | PathLike,
        *,
        ome_xml: str | None = None,
        writer: ZarrWriter = "auto",
        chunks: ShapeLike | Literal["auto"] | None = "auto",
        shards: ShapeLike | None = None,
        overwrite: bool = False,
        compression: CompressionName = "blosc-zstd",
        indent: int = 2,
    ) -> None:
        self._dest = Path(dest)
        self._ome_xml = ome_xml
        self._writer: ZarrWriter = writer
        self._chunks: ShapeLike | Literal["auto"] | None = chunks
        self._shards = shards
        self._overwrite = overwrite
        self._compression: CompressionName = compression
        self._indent = indent

        # For prepare-only workflow: {series_name: (image, dataset_specs)}
        self._series: dict[str, ImageWithDatasets] = {}

        # For immediate write workflow
        self._initialized = False
        self._written_series: list[str] = []

    @property
    def root_path(self) -> Path:
        """Path to the root of the bioformats2raw hierarchy."""
        return self._dest

    def __repr__(self) -> str:
        total_images = len(self._series) + len(self._written_series)
        return f"<{self.__class__.__name__}: {total_images} images>"

    def _validate_series_name(self, name: str) -> None:
        if name in self._written_series:
            raise ValueError(f"Series '{name}' already written via write_image().")
        if name in self._series:
            raise ValueError(f"Series '{name}' already added via add_series().")

    def write_image(
        self,
        name: str,
        image: Image,
        datasets: Sequence[ArrayLike],
        *,
        progress: bool = False,
    ) -> Self:
        """Write a series immediately with its data.

        This method creates the series structure and writes data in one call.
        The root structure and OME metadata are created/updated automatically.
        Use this for the "immediate write" workflow.

        Parameters
        ----------
        name : str
            Series name (becomes the subgroup path, e.g., "0", "1").
        image : Image
            OME-Zarr Image metadata model for this series.
        datasets : Sequence[ArrayLike]
            Data arrays for each resolution level.
        progress : bool, optional
            Show progress bar for dask arrays. Default is False.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If a series with this name was already written or added.
        NotImplementedError
            If the Image has multiple multiscales.
        """
        self._validate_series_name(name)

        # Initialize root structure if needed
        self._ensure_initialized()

        # Update OME/zarr.json with this series
        self._update_ome_series(name)

        # Write the series using the existing write_image function
        write_image(
            self._dest / name,
            image,
            datasets,
            writer=self._writer,
            chunks=self._chunks,
            shards=self._shards,
            overwrite=self._overwrite,
            compression=self._compression,
            progress=progress,
        )

        return self

    def add_series(
        self,
        name: str,
        image: Image,
        datasets: Sequence[ArrayLike],
    ) -> Self:
        """Add a series for the prepare-only workflow.

        Registers a series to be created when `prepare()` is called. Use this
        when you want to create the Zarr structure without writing data
        immediately. After calling `prepare()`, write data to the returned
        array handles.

        Parameters
        ----------
        name : str
            Series name (becomes the subgroup path, e.g., "0", "1").
        image : Image
            OME-Zarr Image metadata model for this series.
        datasets : Sequence[ArrayLike]
            Array-like objects specifying shape and dtype for each resolution
            level. These provide the array specifications; data is not written.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If a series with this name was already added or written, or if the
            number of datasets doesn't match the metadata.
        NotImplementedError
            If the Image has multiple multiscales.
        """
        self._validate_series_name(name)

        if len(image.multiscales) != 1:
            raise NotImplementedError(
                f"Series '{name}': Image must have exactly one multiscale"
            )

        multiscale = image.multiscales[0]
        if len(datasets) != len(multiscale.datasets):
            raise ValueError(
                f"Series '{name}': Number of data arrays ({len(datasets)}) must match "
                f"number of datasets in metadata ({len(multiscale.datasets)})"
            )

        self._series[name] = (image, datasets)
        return self

    def prepare(self) -> tuple[Path, dict[str, Any]]:
        """Create the Zarr hierarchy and return array handles.

        Creates the complete bioformats2raw structure including root metadata,
        OME directory with series list, and empty arrays for all registered
        series. Call this after registering all series with `add_series()`.

        The returned arrays support numpy-style indexing for writing data:
        `arrays["series/dataset"][:] = data`.

        Returns
        -------
        tuple[Path, dict[str, Any]]
            A tuple of (root_path, arrays) where `arrays` maps composite keys
            like `"0/0"` (series name / dataset path) to array objects. The
            array type depends on the configured writer (zarr.Array or
            tensorstore.TensorStore).

        Raises
        ------
        ValueError
            If no series have been added with `add_series()`.
        FileExistsError
            If destination exists and `overwrite` is False.
        ImportError
            If no suitable writer backend is installed.
        """
        if not self._series:  # pragma: no cover
            raise ValueError("No series added. Use add_series() before prepare().")

        # Create root zarr.json with bioformats2raw.layout
        bf2raw = Bf2Raw(bioformats2raw_layout=3)  # type: ignore
        _create_zarr3_group(self._dest, bf2raw, self._overwrite)

        # Create OME/zarr.json with series list
        ome_path = self._dest / "OME"
        series_model = Series(series=list(self._series))
        _create_zarr3_group(ome_path, series_model, self._overwrite)

        # Write METADATA.ome.xml if provided
        if self._ome_xml is not None:
            (ome_path / "METADATA.ome.xml").write_text(self._ome_xml)

        # Create arrays for each series using prepare_image
        all_arrays: dict[str, Any] = {}
        for series_name, (image_model, dataset_specs) in self._series.items():
            _root_path, series_arrays = prepare_image(
                self._dest / series_name,
                image_model,
                dataset_specs,
                chunks=self._chunks,
                shards=self._shards,
                writer=self._writer,
                overwrite=self._overwrite,
                compression=self._compression,
            )
            # Flatten into all_arrays with "series/dataset" keys
            for dataset_path, arr in series_arrays.items():
                all_arrays[f"{series_name}/{dataset_path}"] = arr

        return self._dest, all_arrays

    def _ensure_initialized(self) -> None:
        """Create root structure if not already done."""
        if self._initialized:
            return

        # Create root zarr.json with bioformats2raw.layout
        bf2raw = Bf2Raw(bioformats2raw_layout=3)  # type: ignore
        _create_zarr3_group(self._dest, bf2raw, self._overwrite)

        # Create OME directory and write METADATA.ome.xml if provided
        ome_path = self._dest / "OME"
        ome_path.mkdir(parents=True, exist_ok=True)
        if self._ome_xml is not None:
            (ome_path / "METADATA.ome.xml").write_text(self._ome_xml)

        self._initialized = True

    def _update_ome_series(self, series_name: str) -> None:
        """Update OME/zarr.json with new series name."""
        if series_name in self._written_series:
            # already added ... this is an internal method, don't need to raise
            return  # pragma: no cover

        self._written_series.append(series_name)
        series_model = Series(series=self._written_series)
        zarr_json = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": series_model.model_dump(mode="json", exclude_none=True),
            },
        }
        (self._dest / "OME" / "zarr.json").write_text(
            json.dumps(zarr_json, indent=self._indent)
        )


class PlateBuilder:
    """Builder for OME-Zarr v0.5 Plate hierarchies with auto-generated metadata.

    The Plate hierarchy includes:
    - A root Plate group with metadata (auto-generated from written wells)
    - Well subgroups (e.g., A/1/, B/2/, etc...) each containing Well metadata
    - Field subgroups (e.g., 0/, 1/) within each well, each an Image

    This builder supports two workflows:

    1. **Immediate write** (simpler): Use `write_well()` to write each well
       with its field data immediately. The builder auto-generates and updates
       plate metadata (rows, columns, wells) after each call, similar to how
       Bf2RawBuilder auto-updates the series list.

    2. **Prepare-only** (flexible): Use `add_well()` to register all wells,
       then `prepare()` to create the hierarchy with empty arrays. Plate
       metadata is auto-generated from all registered wells.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Plate Zarr group.
    plate : Plate | None, optional
        Optional OME-Zarr Plate metadata model. If None (default), plate
        metadata (rows, columns, wells) is auto-generated from written/added
        wells. If provided, validates that written wells match the metadata.
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend to use for writing arrays. Default is "auto".
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape for all arrays. Default is "auto".
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding. Default is None (no sharding).
    overwrite : bool, optional
        If True, overwrite existing groups. Default is False.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. Default is "blosc-zstd".

    Examples
    --------
    **Auto-generation workflow (recommended):**

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from yaozarrs import v05
    >>> from yaozarrs.write.v05 import PlateBuilder
    >>>
    >>> def make_image():
    ...     return v05.Image(
    ...         multiscales=[
    ...             v05.Multiscale(
    ...                 axes=[v05.SpaceAxis(name="y"), v05.SpaceAxis(name="x")],
    ...                 datasets=[
    ...                     v05.Dataset(
    ...                         path="0",
    ...                         coordinateTransformations=[
    ...                             v05.ScaleTransformation(scale=[1.0, 1.0])
    ...                         ],
    ...                     )
    ...                 ],
    ...             )
    ...         ]
    ...     )
    >>>
    >>> # No plate metadata needed - it's auto-generated!
    >>> dest = Path(tmpdir) / "plate_auto.zarr"
    >>> builder = PlateBuilder(dest)
    >>> builder.write_well(
    ...     row="A",
    ...     col="1",
    ...     fields={"0": (make_image(), [np.zeros((32, 32), dtype=np.uint16)])},
    ... )
    <PlateBuilder: 1 wells>
    >>> builder.write_well(
    ...     row="A",
    ...     col="2",
    ...     fields={"0": (make_image(), [np.zeros((32, 32), dtype=np.uint16)])},
    ... )
    <PlateBuilder: 2 wells>
    >>> assert (dest / "zarr.json").exists()  # Plate metadata auto-updated

    **With explicit plate metadata:**

    >>> plate = v05.Plate(
    ...     plate=v05.PlateDef(
    ...         columns=[v05.Column(name="1")],
    ...         rows=[v05.Row(name="A")],
    ...         wells=[v05.PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
    ...     )
    ... )
    >>> dest2 = Path(tmpdir) / "plate_explicit.zarr"
    >>> builder2 = PlateBuilder(dest2, plate=plate)
    >>> builder2.write_well(
    ...     row="A",
    ...     col="1",
    ...     fields={"0": (make_image(), [np.zeros((32, 32), dtype=np.uint16)])},
    ... )
    <PlateBuilder: 1 wells>

    See Also
    --------
    write_plate : High-level function to write all wells at once.
    """

    def __init__(
        self,
        dest: str | PathLike,
        *,
        plate: Plate | None = None,
        writer: ZarrWriter = "auto",
        chunks: ShapeLike | Literal["auto"] | None = "auto",
        shards: ShapeLike | None = None,
        overwrite: bool = False,
        compression: CompressionName = "blosc-zstd",
    ) -> None:
        self._dest = Path(dest)
        self._user_plate = plate  # Store user-provided plate (if any)
        self._writer: ZarrWriter = writer
        self._chunks: ShapeLike | Literal["auto"] | None = chunks
        self._shards = shards
        self._overwrite = overwrite
        self._compression: CompressionName = compression

        # For prepare-only workflow: {well_path: {fov: (Image, datasets)}}
        self._wells: dict[str, dict[str, ImageWithDatasets]] = {}

        # For immediate write workflow
        self._initialized = False
        # Track written wells: {(row, col): {fov: (Image, datasets)}}
        self._written_wells_data: dict[
            tuple[str, str], dict[str, ImageWithDatasets]
        ] = {}

    @property
    def root_path(self) -> Path:
        """Path to the root of the plate hierarchy."""
        return self._dest

    def __repr__(self) -> str:
        total_wells = len(self._wells) + len(self._written_wells_data)
        return f"<{self.__class__.__name__}: {total_wells} wells>"

    def _validate_well_coordinates(self, row: str, col: str) -> None:
        """Validate that well coordinates haven't been used yet."""
        # Check if already written or added
        well_coords = (row, col)
        if well_coords in self._written_wells_data:
            raise ValueError(f"Well ({row}, {col}) already written via write_well().")
        well_path = f"{row}/{col}"
        if well_path in self._wells:
            raise ValueError(f"Well '{well_path}' already added via add_well().")

        # If user provided a plate, validate against it
        if self._user_plate is not None:
            valid_well_paths = [well.path for well in self._user_plate.plate.wells]
            if well_path not in valid_well_paths:
                raise ValueError(
                    f"Well path '{well_path}' not found in plate metadata. "
                    f"Valid wells are: {valid_well_paths}"
                )

    def _ensure_initialized(self) -> None:
        """Create plate root directory if not already done.

        Note: Plate zarr.json is created/updated by _update_plate_metadata(),
        not here. This allows starting with zero wells.
        """
        if self._initialized:
            return

        # Create root directory
        self._dest.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def _generate_current_plate_metadata(self) -> Plate:
        """Generate plate metadata from currently written wells.

        If user provided a Plate, use that. Otherwise, auto-generate from
        written wells (similar to write_plate auto-generation).
        """
        # If user provided a complete Plate, use it
        if self._user_plate is not None:
            return self._user_plate

        # Auto-generate from written wells
        # Convert written wells to the format expected by _autogenerate_plate_metadata
        images_dict: dict[tuple[str, str, str], ImageWithDatasets] = {}
        for (row, col), fields in self._written_wells_data.items():
            for fov, image_data in fields.items():
                images_dict[(row, col, fov)] = image_data

        # Generate and merge with user-provided dict (if any)
        return _merge_plate_metadata(images_dict, self._user_plate)

    def _update_plate_metadata(self) -> None:
        """Update plate zarr.json with current wells.

        Similar to Bf2RawBuilder._update_ome_series(), this regenerates
        the plate metadata from currently written wells and rewrites zarr.json.
        """
        plate = self._generate_current_plate_metadata()
        zarr_json = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": plate.model_dump(mode="json", exclude_none=True),
            },
        }
        (self._dest / "zarr.json").write_text(json.dumps(zarr_json, indent=2))

        # Create row directories if needed
        row_names = {row for (row, _col) in self._written_wells_data.keys()} | {
            row for row_path in self._wells.keys() for row in [row_path.split("/")[0]]
        }
        for row_name in row_names:
            row_path = self._dest / row_name
            if not row_path.exists():
                _create_zarr3_group(row_path, ome_model=None, overwrite=self._overwrite)

    def _generate_well_metadata(self, well_path: str, field_names: list[str]) -> Well:
        """Generate Well metadata from field names.

        Parameters
        ----------
        well_path : str
            Well path like "A/1"
        field_names : list[str]
            List of field of view identifiers like ["0", "1"]

        Returns
        -------
        Well
            Well metadata with images list populated.
        """
        # Auto-generate Well metadata
        # Sort field_names for consistent ordering
        images = [
            FieldOfView(path=fov, acquisition=None) for fov in sorted(field_names)
        ]

        return Well(well=WellDef(images=images))

    def write_well(
        self,
        *,
        row: str,
        col: str,
        fields: Mapping[str, ImageWithDatasets],
        progress: bool = False,
    ) -> Self:
        """Write a well immediately with its field images and data.

        This method creates the well structure and writes all field data in one
        call. The plate structure and well metadata are created/updated
        automatically. Plate metadata (rows, columns, wells) is auto-generated
        from all written wells and rewritten after each call.

        Parameters
        ----------
        row : str
            Row name like "A", "B", etc.
        col : str
            Column name like "1", "2", etc.
        fields : Mapping[str, ImageWithDatasets]
            Mapping of `{fov -> (image_model, datasets)}` where fov is the
            field of view identifier like "0", "1", etc.
        progress : bool, optional
            Show progress bar for dask arrays. Default is False.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If row/col combination was already written or added, or if a user-
            provided Plate doesn't include this well.
        NotImplementedError
            If any Image has multiple multiscales.
        """
        # Validate well hasn't been used
        self._validate_well_coordinates(row, col)

        # Initialize plate structure if needed
        self._ensure_initialized()

        # Track this well's data before writing
        well_coords = (row, col)
        self._written_wells_data[well_coords] = dict(fields)

        # Update plate metadata with the new well
        self._update_plate_metadata()

        # Generate Well metadata for this well
        well_path = f"{row}/{col}"
        well_metadata = self._generate_well_metadata(well_path, list(fields.keys()))

        # Create well subgroup with metadata
        well_group_path = self._dest / well_path
        _create_zarr3_group(well_group_path, well_metadata, self._overwrite)

        # Write each field image
        for fov, (image_model, datasets) in fields.items():
            field_path = well_group_path / fov
            write_image(
                field_path,
                image_model,
                datasets,
                writer=self._writer,
                chunks=self._chunks,
                shards=self._shards,
                overwrite=self._overwrite,
                compression=self._compression,
                progress=progress,
            )

        return self

    def add_well(
        self,
        *,
        row: str,
        col: str,
        fields: Mapping[str, ImageWithDatasets],
    ) -> Self:
        """Add a well for the prepare-only workflow.

        Registers a well with its fields to be created when `prepare()` is called.
        Use this when you want to create the Zarr structure without writing data
        immediately. After calling `prepare()`, write data to the returned array
        handles.

        Parameters
        ----------
        row : str
            Row name like "A", "B", etc.
        col : str
            Column name like "1", "2", etc.
        fields : Mapping[str, ImageWithDatasets]
            Mapping of `{fov -> (image_model, datasets)}` where datasets can
            be array-like objects or (dtype, shape) tuples.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If row/col combination was already added/written, or if a user-
            provided Plate doesn't include this well, or if field datasets
            don't match Image metadata.
        NotImplementedError
            If any Image has multiple multiscales.
        """
        # Validate well hasn't been used
        self._validate_well_coordinates(row, col)

        # Validate all fields before accepting
        well_path = f"{row}/{col}"
        for fov, (image_model, datasets) in fields.items():
            if len(image_model.multiscales) != 1:
                raise NotImplementedError(
                    f"Well '{well_path}', field '{fov}': "
                    "Image must have exactly one multiscale"
                )

            multiscale = image_model.multiscales[0]
            if len(datasets) != len(multiscale.datasets):
                raise ValueError(
                    f"Well '{well_path}', field '{fov}': "
                    f"Number of data arrays ({len(datasets)}) must match "
                    f"number of datasets in metadata ({len(multiscale.datasets)})"
                )

        self._wells[well_path] = dict(fields)
        return self

    def prepare(self) -> tuple[Path, dict[str, Any]]:
        """Create the Zarr hierarchy and return array handles.

        Creates the complete Plate structure including plate metadata (auto-
        generated from registered wells), well subgroups with Well metadata,
        and empty arrays for all registered fields. Call this after registering
        all wells with `add_well()`.

        The returned arrays support numpy-style indexing for writing data:
        `arrays["well/field/dataset"][:] = data`.

        Returns
        -------
        tuple[Path, dict[str, Any]]
            A tuple of (root_path, arrays) where `arrays` maps composite keys
            like `"A/1/0/0"` (well_path / field / dataset_path) to array
            objects. The array type depends on the configured writer
            (zarr.Array or tensorstore.TensorStore).

        Raises
        ------
        ValueError
            If no wells have been added with `add_well()`.
        FileExistsError
            If destination exists and `overwrite` is False.
        ImportError
            If no suitable writer backend is installed.
        """
        if not self._wells:
            raise ValueError("No wells added. Use add_well() before prepare().")

        # Generate plate metadata from registered wells
        # Convert wells dict to the format expected by _autogenerate_plate_metadata
        images_dict: dict[tuple[str, str, str], ImageWithDatasets] = {}
        for well_path, fields in self._wells.items():
            row, col = well_path.split("/")
            for fov, image_data in fields.items():
                images_dict[(row, col, fov)] = image_data

        plate = _merge_plate_metadata(images_dict, self._user_plate)

        # Create plate zarr.json
        _create_zarr3_group(self._dest, plate, self._overwrite)

        # Create arrays for each well/field combination
        all_arrays: dict[str, Any] = {}

        for well_path, fields in self._wells.items():
            # Generate Well metadata
            well_metadata = self._generate_well_metadata(well_path, list(fields.keys()))

            # Create well group with metadata
            well_group_path = self._dest / well_path
            _create_zarr3_group(well_group_path, well_metadata, self._overwrite)

            # Create arrays for each field
            for fov, (image_model, datasets) in fields.items():
                field_path = well_group_path / fov

                _field_path, field_arrays = prepare_image(
                    field_path,
                    image_model,
                    datasets,
                    chunks=self._chunks,
                    shards=self._shards,
                    writer=self._writer,
                    overwrite=self._overwrite,
                    compression=self._compression,
                )

                # Flatten into all_arrays with "well/field/dataset" keys
                for dataset_path, arr in field_arrays.items():
                    composite_key = f"{well_path}/{fov}/{dataset_path}"
                    all_arrays[composite_key] = arr

        return self._dest, all_arrays


# ######################## Internal Helpers ####################################


def _row_name_to_index(row_name: str) -> int:
    """Convert row name to index (A=0, B=1, ..., Z=25, AA=26, etc.)."""
    if not row_name or not row_name.isalpha() or not row_name.isupper():
        raise ValueError(
            f"Row name must be uppercase letters (A-Z, AA-ZZ, etc.), got: {row_name}"
        )

    # Convert like Excel columns: A=0, B=1, ..., Z=25, AA=26, AB=27, etc.
    index = 0
    for char in row_name:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _column_name_to_index(col_name: str) -> int:
    """Convert column name to index (1=0, 2=1, 10=9, etc.)."""
    try:
        # Try parsing as integer (handles "1", "2", "10", "1", etc.)
        return int(col_name) - 1
    except ValueError:
        raise ValueError(
            f"Column name must be numeric (1, 2, 10, etc.), got: {col_name}"
        ) from None


def _autogenerate_plate_metadata(
    images: Mapping[tuple[str, str, str], ImageWithDatasets],
) -> dict[str, Any]:
    """Auto-generate plate metadata from images dict keys.

    Row indices follow the convention: A=0, B=1, ..., Z=25, AA=26, etc.
    Column indices follow: 1=0, 2=1, ..., 10=9, etc.

    All intermediate rows/columns are created (e.g., if you have rows A and D,
    rows B and C are also created, even if they have no wells).

    Parameters
    ----------
    images : Mapping[tuple[str, str, str], ImageWithDatasets]
        Images mapping with (row, col, fov) keys.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'rows', 'columns', 'wells' keys for PlateDef.
    """
    from yaozarrs.v05._plate import Column, PlateWell, Row

    # Extract unique row and column names
    rows_set: set[str] = set()
    cols_set: set[int] = set()
    wells_set: set[tuple[str, str]] = set()

    for row, col, _fov in images.keys():
        rows_set.add(row)
        cols_set.add(int(col))
        wells_set.add((row, col))

    # Find the maximum row and column to fill in all intermediates
    max_row_idx = max(_row_name_to_index(r) for r in rows_set)
    max_col_idx = max(cols_set) - 1  # Convert to 0-indexed

    # Create all rows from A to max_row (e.g., A, B, C, D if max is D)
    rows = []
    for idx in range(max_row_idx + 1):
        # Convert index back to row name (0=A, 1=B, etc.)
        if idx < 26:
            row_name = chr(ord("A") + idx)
        else:
            # Handle AA, AB, etc. (Excel-style)
            first = chr(ord("A") + (idx // 26) - 1)
            second = chr(ord("A") + (idx % 26))
            row_name = first + second
        rows.append(Row(name=row_name))

    # Create all columns from 1 to max_col (e.g., 1, 2, 3, 4, 5 if max is 5)
    columns = [Column(name=str(i + 1)) for i in range(max_col_idx + 1)]

    # Create well objects only for wells that have images
    wells = []
    for row, col in sorted(wells_set):
        row_idx = _row_name_to_index(row)
        col_idx = _column_name_to_index(col)
        wells.append(
            PlateWell(path=f"{row}/{col}", rowIndex=row_idx, columnIndex=col_idx)
        )

    return {"rows": rows, "columns": columns, "wells": wells}


def _merge_plate_metadata(
    images: Mapping[tuple[str, str, str], ImageWithDatasets],
    user_plate: Plate | dict[str, Any] | None,
) -> Plate:
    """Merge auto-generated and user-provided plate metadata.

    Parameters
    ----------
    images : Mapping[tuple[str, str, str], ImageWithDatasets]
        Images mapping used to auto-generate metadata.
    user_plate : Plate | dict[str, Any] | None
        User-provided plate metadata (takes precedence over auto-generated).

    Returns
    -------
    Plate
        Final Plate object with merged metadata.

    Raises
    ------
    ValueError
        If user-provided metadata conflicts with images dict.
    """
    from yaozarrs.v05._plate import Plate, PlateDef

    # Auto-generate base metadata from images
    auto_metadata = _autogenerate_plate_metadata(images)

    # If user provided a full Plate object, validate and return
    if isinstance(user_plate, Plate):
        _validate_plate_matches_images(user_plate, images)
        return user_plate

    # If user provided a dict, merge with auto-generated
    if user_plate is not None:
        # User-provided values take precedence
        merged = auto_metadata.copy()
        for key, value in user_plate.items():
            merged[key] = value
    else:
        merged = auto_metadata

    # Construct Plate object from merged metadata
    plate = Plate(plate=PlateDef(**merged))

    # Validate that all images have valid coordinates
    _validate_plate_matches_images(plate, images)

    return plate


def _validate_plate_matches_images(
    plate: Plate,
    images: Mapping[tuple[str, str, str], ImageWithDatasets],
) -> None:
    """Validate that plate metadata matches images dict.

    Parameters
    ----------
    plate : Plate
        Plate metadata to validate.
    images : Mapping[tuple[str, str, str], ImageWithDatasets]
        Images mapping to validate against.

    Raises
    ------
    ValueError
        If any image coordinates don't match plate metadata.
    """
    # Get valid row and column names from plate
    valid_rows = {row.name for row in plate.plate.rows}
    valid_cols = {col.name for col in plate.plate.columns}
    valid_wells = {well.path for well in plate.plate.wells}

    # Check all images have valid coordinates
    for row, col, _fov in images.keys():
        if row not in valid_rows:
            raise ValueError(
                f"Image row '{row}' not found in plate rows: {sorted(valid_rows)}"
            )
        if col not in valid_cols:
            raise ValueError(
                f"Image column '{col}' not found in plate columns: {sorted(valid_cols)}"
            )
        well_path = f"{row}/{col}"
        if well_path not in valid_wells:
            raise ValueError(
                f"Image well '{well_path}' not found in plate wells: "
                f"{sorted(valid_wells)}"
            )


# ######################## Zarr Group Creation ##################################


def _create_zarr3_group(
    dest_path: Path,
    ome_model: OMEMetadata | None = None,
    overwrite: bool = False,
    indent: int = 2,
) -> None:
    """Create a zarr group directory with optional OME metadata in zarr.json."""
    zarr_json_path = dest_path / "zarr.json"
    if dest_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Zarr group already exists at {dest_path}. "
                "Use overwrite=True to replace it."
            )
        # Be cautious before deleting.
        # If it doesn't look like a zarr group, raise an error rather than deleting.
        if not zarr_json_path.exists():
            raise FileExistsError(
                f"Destination {dest_path} exists, but is not a Zarr group. "
                "Refusing to overwrite.  Please delete manually."
            )

        shutil.rmtree(dest_path, ignore_errors=True)

    dest_path.mkdir(parents=True, exist_ok=True)
    zarr_json: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "group",
    }
    if ome_model is not None:
        zarr_json["attributes"] = {
            "ome": ome_model.model_dump(mode="json", exclude_none=True),
        }
    zarr_json_path.write_text(json.dumps(zarr_json, indent=indent))


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
            # Handle both zarr and tensorstore
            if hasattr(array, "store"):  # zarr.Array
                da.store(dask_data, array, lock=False)  # type: ignore
            else:  # tensorstore
                computed = dask_data.compute()
                array[:].write(computed).result()

    else:
        if hasattr(array, "store"):  # zarr.Array
            array[:] = data  # type: ignore
        else:  # tensorstore
            array[:].write(data).result()  # type: ignore


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
        if importlib.util.find_spec("tensorstore"):
            return _create_array_tensorstore
        elif writer == "tensorstore":
            raise ImportError(
                "tensorstore is required for the 'tensorstore' writer. "
                "Please pip install with yaozarrs[write-tensorstore]"
            )
    elif writer == "zarr":
        if importlib.util.find_spec("zarr"):
            zarr_version_str = importlib.metadata.version("zarr")
            zarr_major_version = int(zarr_version_str.split(".")[0])
            if zarr_major_version < 3 and writer in {"zarr"}:
                raise ImportError(
                    f"zarr v3 or higher is required for OME-Zarr v0.5 writing, "
                    f"but zarr v{zarr_version_str} is installed. "
                    "Please upgrade zarr to v3 or higher."
                )
            return _create_array_zarr
        raise ImportError(
            "zarr-python is required for the 'zarr' writer. "
            "Please pip install with yaozarrs[write-zarr]"
        )

    raise ValueError(
        f"Unknown writer option: {writer}.  "
        "Must be 'zarr', 'tensorstore', 'auto', or a custom function."
    )
