"""OME-Zarr v0.6 writing functionality.

This module provides convenience functions to write OME-Zarr v0.6 groups.

The general pattern is:
1. Decide what kind of OME-Zarr best matches your data:
   - Single Image (<=5D): use Image model and write_image function.
   - Multi-Well Plate: use Plate model and write_plate function.
   - Collection of images (e.g. 5D images at multiple positions, or any other 6+D data):
     use bioformats2raw layout with Bf2Raw model and write_bioformats2raw function.
2. Create your OME-Zarr metadata model using the yaozarrs.v06 models.
3. Decide whether to use high level write functions (write_image, write_plate, etc...)
   or lower level prepare/Builder methods (`prepare_image`, `PlateBuilder`, etc...).
   (Note: for Plates and Bf2Raw collections, lower level builders are recommended)
   - Use high level write_* functions for simple one-shot writes, where you can provide
     the full data arrays up front (either as numpy, dask, etc...)
   - Use lower level prepare_*/Builder APIs when you need to customize how data is
     written, perhaps in a streaming, or slice-by-slice manner.
4. Call the appropriate function with your metadata model and data arrays.

A key observation here is that only the Dataset entries in any of the models actually
corresponds to a zarr Array... the rest of the format is just metadata and group
hierarchy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from yaozarrs.v06 import (
    Bf2Raw,
    Image,
    LabelImage,
    LabelsGroup,
    Plate,
    PlateDef,
)
from yaozarrs.v06._bf2raw import Series
from yaozarrs.v06._plate import FieldOfView, Well, WellDef
from yaozarrs.write import _core
from yaozarrs.write._core import CreateArrayFunc  # noqa: TC001  (public re-export)

__all__ = [
    "Bf2RawBuilder",
    "LabelsBuilder",
    "PlateBuilder",
    "prepare_image",
    "write_bioformats2raw",
    "write_image",
    "write_plate",
]

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike
    from pathlib import Path
    from typing import Any

    import tensorstore
    import zarr
    from typing_extensions import Literal, TypeAlias

    from yaozarrs.write._core import (
        AnyZarrArray,
        ArrayOrPyramid,
        CompressionName,
        ShapeAndDTypeOrPyramid,
        ZarrWriter,
    )

    ImageWithDatasets: TypeAlias = tuple[Image, ArrayOrPyramid]

# The v06-specific model classes the shared writer core builds/accepts.
_MODELS = _core.OMEModels(
    version="0.6",
    Image=Image,
    LabelImage=LabelImage,
    LabelsGroup=LabelsGroup,
    Plate=Plate,
    PlateDef=PlateDef,
    Well=Well,
    WellDef=WellDef,
    FieldOfView=FieldOfView,
    Bf2Raw=Bf2Raw,
    Series=Series,
)


# ######################## Public API ##########################################

# ------------------------ High Level Write Functions --------------------------


def write_image(
    dest: str | PathLike,
    image: Image,
    datasets: ArrayOrPyramid,
    *,
    labels: Mapping[str, tuple[LabelImage, ArrayOrPyramid]] | None = None,
    extra_attributes: dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write an OME-Zarr v0.6 Image group with data.

    This is the high-level function for writing a complete OME-Zarr image.
    It creates the Zarr group hierarchy, writes metadata, and stores data
    in a single call.

    See Also
    --------
    - [`prepare_image`][yaozarrs.write.v06.prepare_image] : Create arrays without
      writing data (for custom write logic).
    - [`write_bioformats2raw`][yaozarrs.write.v06.write_bioformats2raw] : Write
      multi-series bioformats2raw layout.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Zarr group. Will be created if it doesn't exist.
    image : Image
        OME-Zarr Image metadata model. Must have exactly one multiscale, with
        one Dataset entry per array in `datasets`.
    datasets : ArrayOrPyramid
        Data array(s) to write (numpy, dask, or any array with shape/dtype).
        - For a single dataset, pass the array directly:
          `write_image(dest, image, data)`
        - For multiple datasets (e.g., multiscale pyramid), pass a sequence:
          `write_image(dest, image, [data0, data1, ...])`
        Must match the number and order of `image.multiscales[0].datasets`.
    labels : Mapping[str, tuple[LabelImage, ArrayOrPyramid]] | None, optional
        Optional label images to write alongside the image. Keys are label names
        (e.g., "cells", "nuclei"), values are (LabelImage, datasets) tuples.
        Labels will be written to `dest/labels/{name}/`. Default is None.
    extra_attributes : dict[str, Any] | None, optional
        Additional attributes to write alongside "ome" in zarr.json.
        For example, `{"custom": {...}}` will produce
        `attributes: {"ome": {...}, "custom": {...}}`.
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend to use for writing arrays. "auto" prefers tensorstore if
        available, otherwise falls back to zarr-python. Pass a custom function
        matching the `CreateArrayFunc` protocol for custom backends.
    overwrite : bool, optional
        If True, overwrite existing Zarr group at `dest`. Default is False.
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape for storage. "auto" (default) calculates ~4MB chunks with
        non-spatial dims set to 1. None uses the full array shape (single chunk).
        Tuple values are clamped to the array shape.
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding. Default is None (no sharding).
        When present, shard_shape must be divisible by chunk shape.
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
    Write a simple 3D image (CYX) - single dataset:

    >>> import numpy as np
    >>> from yaozarrs import DimSpec, v06
    >>> from yaozarrs.write.v06 import write_image
    >>>
    >>> data = np.zeros((2, 64, 64), dtype=np.uint16)
    >>> # v0.6 nests `axes` inside named coordinate systems; `from_dims` builds
    >>> # the coordinateSystems + input/output transforms for you.
    >>> image = v06.Image(
    ...     multiscales=[
    ...         v06.Multiscale.from_dims(
    ...             [
    ...                 DimSpec(name="c"),
    ...                 DimSpec(name="y", unit="micrometer"),
    ...                 DimSpec(name="x", unit="micrometer"),
    ...             ]
    ...         )
    ...     ]
    ... )
    >>> result = write_image("example.ome.zarr", image, data)
    >>> assert result.exists()

    Write a 3D image (CYX) with associated labels:

    >>> # Create label images for segmentation
    >>> cells_label = v06.LabelImage(
    ...     **image.model_dump(),
    ...     image_label={"colors": [{"label_value": 1, "rgba": [255, 0, 0, 255]}]},
    ... )
    >>> cells_data = np.zeros((2, 64, 64), dtype=np.uint8)
    >>> nuclei_label = v06.LabelImage(**image.model_dump(), image_label={})
    >>> nuclei_data = np.zeros((2, 64, 64), dtype=np.uint8)
    >>> result = write_image(
    ...     "example.ome.zarr",
    ...     image,
    ...     data,
    ...     labels={
    ...         "cells": (cells_label, cells_data),
    ...         "nuclei": (nuclei_label, nuclei_data),
    ...     },
    ...     overwrite=True,
    ... )
    >>> assert (result / "labels" / "cells" / "0" / "zarr.json").exists()
    """
    return _core.write_image(
        dest,
        image,
        datasets,
        models=_MODELS,
        labels_builder=LabelsBuilder,
        labels=labels,
        extra_attributes=extra_attributes,
        writer=writer,
        overwrite=overwrite,
        chunks=chunks,
        shards=shards,
        compression=compression,
        progress=progress,
    )


def write_plate(
    dest: str | PathLike,
    images: Mapping[tuple[str, str, str], ImageWithDatasets],
    *,
    plate: Plate | dict[str, Any] | None = None,
    extra_attributes: dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write an OME-Zarr v0.6 Plate group with data.

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

    See Also
    --------
    - [`PlateBuilder`][yaozarrs.write.v06.PlateBuilder]: Builder class for incremental
      well/field writing.
    - [`write_image`][yaozarrs.write.v06.write_image]: Write a single Image group.

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
    extra_attributes : dict[str, Any] | None, optional
        Additional attributes to write alongside "ome" in zarr.json.
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend to use for writing arrays. Default is "auto".
    overwrite : bool, optional
        If True, overwrite existing Zarr groups. Default is False.
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape for all arrays. See `write_image` for details.
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding. Default is None (no sharding).
        When present, shard_shape must be divisible by chunk shape.
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
    >>> from yaozarrs import DimSpec, v06
    >>> from yaozarrs.write.v06 import write_plate
    >>>
    >>> # Create image metadata (same for all fields)
    >>> def make_image():
    ...     return v06.Image(
    ...         multiscales=[
    ...             v06.Multiscale.from_dims(
    ...                 [
    ...                     DimSpec(name="y", unit="micrometer"),
    ...                     DimSpec(name="x", unit="micrometer"),
    ...                 ]
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
    >>> result = write_plate("my_plate1.ome.zarr", images)
    >>> assert (result / "A" / "1" / "0" / "zarr.json").exists()
    >>>
    >>> # Or add custom metadata like a name
    >>> result2 = write_plate(
    ...     "my_plate2.ome.zarr",
    ...     images,
    ...     plate={"name": "My Experiment"},
    ...     overwrite=True,
    ... )
    """
    return _core.write_plate(
        dest,
        images,
        plate_builder=PlateBuilder,
        plate=plate,
        extra_attributes=extra_attributes,
        writer=writer,
        overwrite=overwrite,
        chunks=chunks,
        shards=shards,
        compression=compression,
        progress=progress,
    )


def write_bioformats2raw(
    dest: str | PathLike,
    # mapping of {series_name -> ( Image, [datasets, ...] )}
    images: Mapping[str, ImageWithDatasets],
    *,
    ome_xml: str | None = None,
    extra_attributes: dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write a bioformats2raw-layout OME-Zarr with multiple series.

    The bioformats2raw layout is a convention for storing multiple images
    (series) in a single Zarr hierarchy. It includes a root group with
    `bioformats2raw.layout` version, an `OME/` group with series metadata,
    and each series as a separate Image subgroup.

    This is the high-level function for writing all series at once. For
    incremental writes, use [`Bf2RawBuilder`][yaozarrs.write.v06.Bf2RawBuilder]
    directly.

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

    See Also
    --------
    - [`Bf2RawBuilder`][yaozarrs.write.v06.Bf2RawBuilder] : Builder class for
      incremental series writing.
    - [`write_image`][yaozarrs.write.v06.write_image] : Write a single Image group.

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
    extra_attributes : dict[str, Any] | None, optional
        Additional attributes to write alongside "ome" in zarr.json.
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend to use for writing arrays.
    overwrite : bool, optional
        If True, overwrite existing Zarr groups. Default is False.
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape for all arrays. See `write_image` for details.
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding. Default is None (no sharding).
        When present, shard_shape must be divisible by chunk shape.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. Default is "blosc-zstd".
    progress : bool, optional
        Show progress bar when writing dask arrays. Default is False.

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
    >>> from yaozarrs import DimSpec, v06
    >>> from yaozarrs.write.v06 import write_bioformats2raw
    >>>
    >>> def make_image():
    ...     return v06.Image(
    ...         multiscales=[
    ...             v06.Multiscale.from_dims(
    ...                 [
    ...                     DimSpec(name="y", unit="micrometer"),
    ...                     DimSpec(name="x", unit="micrometer"),
    ...                 ]
    ...             )
    ...         ]
    ...     )
    >>> images = {
    ...     "0": (make_image(), [np.zeros((64, 64), dtype=np.uint16)]),
    ...     "1": (make_image(), [np.zeros((32, 32), dtype=np.uint16)]),
    ... }
    >>> result = write_bioformats2raw("multi_series.zarr", images)
    >>> (result / "OME" / "zarr.json").exists()
    True
    >>> (result / "0" / "zarr.json").exists()
    True
    """
    return _core.write_bioformats2raw(
        dest,
        images,
        bf2raw_builder=Bf2RawBuilder,
        ome_xml=ome_xml,
        extra_attributes=extra_attributes,
        writer=writer,
        overwrite=overwrite,
        chunks=chunks,
        shards=shards,
        compression=compression,
        progress=progress,
    )


# ------------------------ Lower Level Prepare Functions --------------------------


@overload
def prepare_image(
    dest: str | PathLike,
    image: Image,
    datasets: ShapeAndDTypeOrPyramid,
    *,
    extra_attributes: dict[str, Any] | None = ...,
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
    datasets: ShapeAndDTypeOrPyramid,
    *,
    extra_attributes: dict[str, Any] | None = ...,
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
    datasets: ShapeAndDTypeOrPyramid,
    *,
    extra_attributes: dict[str, Any] | None = ...,
    writer: Literal["auto"] | CreateArrayFunc = ...,
    chunks: tuple[int, ...] | Literal["auto"] | None = ...,
    shards: tuple[int, ...] | None = ...,
    overwrite: bool = ...,
    compression: CompressionName = ...,
) -> tuple[Path, dict[str, AnyZarrArray]]: ...
def prepare_image(
    dest: str | PathLike,
    image: Image,
    datasets: ShapeAndDTypeOrPyramid,
    *,
    extra_attributes: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
) -> tuple[Path, dict[str, Any]]:
    """Create OME-Zarr v0.6 Image structure and return array handles for writing.

    This is a lower-level function that creates the Zarr group hierarchy and
    empty arrays, but does not write data. Use this when you need custom control
    over how data is written (e.g., chunk-by-chunk streaming, parallel writes).

    To write data immediately, use `write_image` instead.

    See Also
    --------
    - [`write_image`][yaozarrs.write.v06.write_image] : High-level function that writes
      data immediately.
    - [`Bf2RawBuilder.prepare`][yaozarrs.write.v06.Bf2RawBuilder.prepare] : Prepare
      multiple series at once.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the Zarr group.
    image : Image
        OME-Zarr Image metadata model.
    datasets : ShapeAndDType | Sequence[ShapeAndDType]
        Shape and dtype specification(s) for each dataset, as `(shape, dtype)`
        tuples. Can be:

        - Single `(shape, dtype)`: For one dataset, no wrapping needed
        - Sequence of `(shape, dtype)`: For multiple datasets (multiscale pyramid)

        Must match the number and order of `image.multiscales[0].datasets`.
    extra_attributes : dict[str, Any] | None, optional
        Additional attributes to write alongside "ome" in zarr.json.
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape. See `write_image` for details.
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding. Default is None (no sharding).
        When present, shard_shape must be divisible by chunk shape.
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
    FileExistsError
        If `dest` exists and `overwrite` is False.
    ImportError
        If no suitable writer backend is installed.

    Examples
    --------
    Create arrays and write data in chunks - single dataset, no list needed:

    >>> import numpy as np
    >>> from yaozarrs import DimSpec, v06
    >>> from yaozarrs.write.v06 import prepare_image
    >>> image = v06.Image(
    ...     multiscales=[
    ...         v06.Multiscale.from_dims(
    ...             [
    ...                 DimSpec(name="y", unit="micrometer"),
    ...                 DimSpec(name="x", unit="micrometer"),
    ...             ]
    ...         )
    ...     ]
    ... )
    >>> # Prepare with just shape/dtype (no data yet) - no list wrapping!
    >>> path, arrays = prepare_image("prepared.zarr", image, ((64, 64), "uint16"))
    >>> arrays["0"][:] = np.zeros((64, 64), dtype=np.uint16)
    >>> assert path.exists()
    """
    return _core.prepare_image(
        dest,
        image,
        datasets,
        models=_MODELS,
        extra_attributes=extra_attributes,
        chunks=chunks,
        shards=shards,
        writer=writer,
        overwrite=overwrite,
        compression=compression,
    )


# ------------------------ Builders --------------------------------------------


class Bf2RawBuilder(_core.Bf2RawBuilderBase[Image]):
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


    See Also
    --------
    - [`write_bioformats2raw`][yaozarrs.write.v06.write_bioformats2raw] : High-level
      function to write all series at once.

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
        When present, shard_shape must be divisible by chunk shape.
    overwrite : bool, optional
        If True, overwrite existing groups. Default is False.
        Note: existing directories that don't look like zarr groups will NOT be removed,
        an exception will be raised instead.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. Default is "blosc-zstd".

    Examples
    --------
    **Immediate write workflow:**

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from yaozarrs import DimSpec, v06
    >>> from yaozarrs.write.v06 import Bf2RawBuilder
    >>> def make_image():
    ...     return v06.Image(
    ...         multiscales=[
    ...             v06.Multiscale.from_dims([DimSpec(name="y"), DimSpec(name="x")])
    ...         ]
    ...     )
    >>> builder = Bf2RawBuilder("builder_immediate.zarr")
    >>> builder.write_image("0", make_image(), np.zeros((32, 32), dtype=np.uint16))
    <Bf2RawBuilder: 1 images>
    >>> builder.write_image("1", make_image(), np.zeros((16, 16), dtype=np.uint16))
    <Bf2RawBuilder: 2 images>
    >>> assert (builder.root_path / "0" / "zarr.json").exists()

    **Prepare-only workflow:**

    >>> builder2 = Bf2RawBuilder("builder_prepare.zarr")
    >>> builder2.add_series("0", make_image(), ((32, 32), np.uint16))  # shape, dtype
    <Bf2RawBuilder: 1 images>
    >>> builder2.add_series("1", make_image(), ((16, 16), np.uint16))
    <Bf2RawBuilder: 2 images>
    >>> path, arrays = builder2.prepare()
    >>> arrays["0/0"][:] = np.zeros((32, 32), dtype=np.uint16)  # Write data yourself
    >>> arrays["1/0"][:] = np.zeros((16, 16), dtype=np.uint16)
    >>> assert path.exists()
    """

    _models = _MODELS


class PlateBuilder(_core.PlateBuilderBase[Image, Plate]):
    """Builder for OME-Zarr v0.6 Plate hierarchies with auto-generated metadata.

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

    See Also
    --------
    - [`write_plate`][yaozarrs.write.v06.write_plate] : High-level function to write
      all wells at once.

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
        When present, shard_shape must be divisible by chunk shape.
    overwrite : bool, optional
        If True, overwrite existing groups. Default is False.
        Note: existing directories that don't look like zarr groups will NOT be removed,
        an exception will be raised instead.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. Default is "blosc-zstd".

    Examples
    --------
    **Auto-generation workflow (recommended):**

    >>> import numpy as np
    >>> from pathlib import Path
    >>> from yaozarrs import DimSpec, v06
    >>> from yaozarrs.write.v06 import PlateBuilder
    >>>
    >>> def make_image():
    ...     return v06.Image(
    ...         multiscales=[
    ...             v06.Multiscale.from_dims([DimSpec(name="y"), DimSpec(name="x")])
    ...         ]
    ...     )
    >>>
    >>> # No plate metadata needed - it's auto-generated!
    >>> builder = PlateBuilder("plate_auto.zarr")
    >>> builder.write_well(
    ...     row="A",
    ...     col="1",
    ...     images={"0": (make_image(), np.zeros((32, 32), dtype=np.uint16))},
    ... )
    <PlateBuilder: 1 wells>
    >>> builder.write_well(
    ...     row="A",
    ...     col="2",
    ...     images={"0": (make_image(), np.zeros((32, 32), dtype=np.uint16))},
    ... )
    <PlateBuilder: 2 wells>
    >>> assert (builder.root_path / "zarr.json").exists()  # Plate metadata auto-updated

    **With explicit plate metadata:**

    >>> plate = v06.Plate(
    ...     plate=v06.PlateDef(
    ...         columns=[v06.Column(name="1")],
    ...         rows=[v06.Row(name="A")],
    ...         wells=[v06.PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
    ...     )
    ... )
    >>> builder2 = PlateBuilder("plate_explicit.zarr", plate=plate)
    >>> builder2.write_well(
    ...     row="A",
    ...     col="1",
    ...     images={"0": (make_image(), np.zeros((32, 32), dtype=np.uint16))},
    ... )
    <PlateBuilder: 1 wells>
    """

    _models = _MODELS


class LabelsBuilder(_core.LabelsBuilderBase[LabelImage]):
    """Builder for labels groups within an Image.

    The labels group structure includes:
    - A labels group with LabelsGroup metadata listing all label names
    - Each label as a separate LabelImage subgroup (e.g., `cells/`, `nuclei/`)

    This builder supports two workflows:

    1. **Immediate write** (simpler): Use `write_label()` to write each label
       with its data immediately. The builder auto-generates and updates
       LabelsGroup metadata after each call.

    2. **Prepare-only** (flexible): Use `add_label()` to register all labels,
       then `prepare()` to create the hierarchy with empty arrays. Write data
       to the returned array handles yourself.

    See Also
    --------
    - [`write_image`][yaozarrs.write.v06.write_image] : High-level function with labels
      parameter for writing everything at once.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the labels Zarr group (typically `image_path/labels`).
    writer : "zarr" | "tensorstore" | "auto" | CreateArrayFunc, optional
        Backend to use for writing arrays. Default is "auto".
    chunks : tuple[int, ...] | "auto" | None, optional
        Chunk shape for all arrays. Default is "auto".
    shards : tuple[int, ...] | None, optional
        Shard shape for Zarr v3 sharding. Default is None (no sharding).
        When present, shard_shape must be divisible by chunk shape.
    overwrite : bool, optional
        If True, overwrite existing groups. Default is False.
        Note: existing directories that don't look like zarr groups will NOT be removed,
        an exception will be raised instead.
    compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
        Compression codec. Default is "blosc-zstd".

    Examples
    --------
    **Immediate write workflow:**

    >>> import numpy as np
    >>> from yaozarrs import DimSpec, v06
    >>> from yaozarrs.write.v06 import LabelsBuilder
    >>> def make_label_image():
    ...     return v06.LabelImage(
    ...         multiscales=[
    ...             v06.Multiscale.from_dims([DimSpec(name="y"), DimSpec(name="x")])
    ...         ],
    ...         image_label=v06.ImageLabel(),
    ...     )
    >>> builder = LabelsBuilder("my_image.zarr/labels")
    >>> builder.write_label(
    ...     "cells", make_label_image(), np.zeros((64, 64), dtype=np.uint32)
    ... )
    <LabelsBuilder: 1 labels>
    >>> builder.write_label(
    ...     "nuclei", make_label_image(), np.zeros((64, 64), dtype=np.uint32)
    ... )
    <LabelsBuilder: 2 labels>

    **Prepare-only workflow:**

    >>> builder2 = LabelsBuilder("my_image2.zarr/labels")
    >>> builder2.add_label(
    ...     "cells",
    ...     make_label_image(),
    ...     ((64, 64), np.uint32),  # shape, dtype spec
    ... )
    <LabelsBuilder: 1 labels>
    >>> path, arrays = builder2.prepare()
    >>> arrays["cells/0"][:] = np.random.randint(0, 10, (64, 64), dtype=np.uint32)
    """

    _models = _MODELS
