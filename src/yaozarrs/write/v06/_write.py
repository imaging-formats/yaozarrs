"""OME-Zarr v0.6 writing functionality.

This module provides convenience functions to write OME-Zarr v0.6 groups.

The general pattern is:
1. Decide what kind of OME-Zarr best matches your data:
   - Single Image (<=5D): use Image model and write_image function.
   - Multi-Well Plate: use Plate model and write_plate function.
   - Collection of images (e.g. 5D images at multiple positions, or any other 6+D data):
     use bioformats2raw layout with Bf2Raw model and write_bioformats2raw function.
   - Images that share a spatial relationship (e.g. tiles, or multiple views of a
     sample): use Scene model and write_scene function.
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

Notes specific to v0.6:

- Arrays get `dimension_names` from the axes of the multiscale's intrinsic coordinate
  system (no longer required by the spec, but still useful to Zarr tools).
- No transformation between an image and its labels is added.  Without one, they are
  understood to share the intrinsic coordinate system; if they don't, add one to the
  image's `Multiscale.coordinateTransformations`.
"""

from __future__ import annotations

import posixpath
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from yaozarrs.v06 import Bf2Raw, Image, LabelImage, LabelsGroup, Plate, Scene
from yaozarrs.v06._bf2raw import Series
from yaozarrs.v06._plate import FieldOfView, Well, WellDef
from yaozarrs.v06._storage import (
    _components,
    _find_coordinate_system,
    _fmt,
    _multiscale_graph,
)
from yaozarrs.v06._transforms import check_transform_ndim
from yaozarrs.write._common._arrays import CreateArrayFunc, _get_create_func
from yaozarrs.write._common._builders import (
    Bf2RawBuilderBase,
    LabelsBuilderBase,
    PlateBuilderBase,
    _check_image_args,
    _prepare_image,
    _write_bioformats2raw,
    _write_image,
    _write_plate,
)
from yaozarrs.write._common._groups import _create_zarr3_group, _ensure_parent_groups
from yaozarrs.write._common._models import _check_model

__all__ = [
    "Bf2RawBuilder",
    "CreateArrayFunc",  # (documented as `yaozarrs.write.v06._write.CreateArrayFunc`)
    "LabelsBuilder",
    "PlateBuilder",
    "SceneBuilder",
    "prepare_image",
    "write_bioformats2raw",
    "write_image",
    "write_plate",
    "write_scene",
]

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from os import PathLike

    import tensorstore
    import zarr
    from typing_extensions import Literal, Self, TypeAlias

    from yaozarrs.v06._coordinate_systems import CoordinateSystem
    from yaozarrs.v06._storage import Node
    from yaozarrs.write._common._arrays import (
        AnyZarrArray,
        ArrayOrPyramid,
        CompressionName,
        ShapeAndDTypeOrPyramid,
        ShapeLike,
        ZarrWriter,
    )

    # Compound types for images with data or specs
    ImageWithDatasets: TypeAlias = tuple[Image, ArrayOrPyramid]
    # (an image added to a SceneBuilder: its dataset specs and array options)
    AddedImage: TypeAlias = tuple[Image, ShapeAndDTypeOrPyramid, dict[str, Any]]


class _Default(Enum):
    """Sentinel for a per-image setting that defaults to the builder's."""

    BUILDER = "builder"


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
        Tuple values must have one entry per array dimension, and are clamped
        to the array shape.
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
    TypeError
        If given a model from another OME-Zarr version (e.g. a v05.Image).
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
    >>> image = v06.Image(
    ...     multiscales=[
    ...         v06.Multiscale.from_dims(
    ...             [
    ...                 DimSpec(name="c"),
    ...                 DimSpec(name="y", scale=0.5, unit="micrometer"),
    ...                 DimSpec(name="x", scale=0.5, unit="micrometer"),
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
    return _write_image(
        Path(dest),
        image,
        datasets,
        image_type=Image,
        labels=labels,
        labels_builder_type=LabelsBuilder,
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
    TypeError
        If given a model from another OME-Zarr version (e.g. a v05.Image).
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
    ...                     DimSpec(name="y", scale=0.5, unit="micrometer"),
    ...                     DimSpec(name="x", scale=0.5, unit="micrometer"),
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
    return _write_plate(
        PlateBuilder,
        dest,
        images,
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
    TypeError
        If given a model from another OME-Zarr version (e.g. a v05.Image).
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
    ...                     DimSpec(name="y", scale=0.5, unit="micrometer"),
    ...                     DimSpec(name="x", scale=0.5, unit="micrometer"),
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
    return _write_bioformats2raw(
        Bf2RawBuilder,
        dest,
        images,
        ome_xml=ome_xml,
        extra_attributes=extra_attributes,
        writer=writer,
        overwrite=overwrite,
        chunks=chunks,
        shards=shards,
        compression=compression,
        progress=progress,
    )


def write_scene(
    dest: str | PathLike,
    scene: Scene,
    images: Mapping[str, ImageWithDatasets],
    *,
    extra_attributes: dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write an OME-Zarr v0.6 scene: images that share a spatial relationship.

    The scene group holds the [`Scene`][yaozarrs.v06.Scene] metadata (transformations
    between the coordinate systems of its images, and optionally coordinate systems
    of its own), and each image is written as a subgroup at its path:

        dest/
        ├── zarr.json          # attributes["ome"]["scene"]
        ├── tile_0/            # images["tile_0"]
        │   ├── zarr.json      # Image metadata
        │   └── 0/             # dataset arrays
        └── tile_1/
            └── ...

    Before anything is written, the scene's references are checked: every
    transformation `input`/`output` with a `path` must name one of `images`, with a
    coordinate system of that `name`, the transformation's dimensions must fit the
    coordinate systems it links, and the coordinate systems of the scene and its
    images must form one connected graph.  A reference to a multiscale image *inside*
    one of `images` (such as one of its labels) is allowed, but not checked, as that
    image isn't written here; nor is the graph, in that case.

    `chunks`, `shards` and `compression` apply to every image.  If the images have
    different numbers of dimensions, use `chunks="auto"` (the default) or `None`, or
    write them with a [`SceneBuilder`][yaozarrs.write.v06.SceneBuilder], which takes
    these settings per image.

    With `overwrite=True`, an existing store at `dest` is replaced once the checks pass.

    Transformation parameters stored in Zarr arrays (e.g. an `affine` with a `path`)
    are not written by this function: write them below `dest` afterwards (the spec
    recommends a `coordinateTransformations` group).

    See Also
    --------
    - [`SceneBuilder`][yaozarrs.write.v06.SceneBuilder] : Builder class for
      incremental image writing.
    - [`write_image`][yaozarrs.write.v06.write_image] : Write a single Image group.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the scene Zarr group.
    scene : Scene
        OME-Zarr Scene metadata model.
    images : Mapping[str, ImageWithDatasets]
        Mapping of `{path -> (image_model, [datasets, ...])}`.  Each path (e.g.
        "tile_0", or "tiles/0") becomes an Image subgroup of the scene group.
    extra_attributes : dict[str, Any] | None, optional
        Additional attributes to write alongside "ome" in the scene's zarr.json.
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
        Path to the scene Zarr group.

    Raises
    ------
    TypeError
        If given a model from another OME-Zarr version (e.g. a v05.Image).
    ValueError
        If the scene references an image (or coordinate system) that isn't in
        `images`, if its coordinate systems aren't all connected, or if any image
        has mismatched datasets/metadata.
    NotImplementedError
        If any Image has multiple multiscales.
    FileExistsError
        If `dest` exists and `overwrite` is False.
    ImportError
        If no suitable writer backend is installed.

    Examples
    --------
    Two tiles, placed side by side in a "world" coordinate system:

    >>> import numpy as np
    >>> from yaozarrs import DimSpec, v06
    >>> from yaozarrs.write.v06 import write_scene
    >>>
    >>> def make_image():
    ...     dims = [
    ...         DimSpec(name="y", scale=0.5, unit="micrometer"),
    ...         DimSpec(name="x", scale=0.5, unit="micrometer"),
    ...     ]
    ...     return v06.Image(multiscales=[v06.Multiscale.from_dims(dims)])
    >>>
    >>> world = v06.CoordinateSystem(
    ...     name="world",
    ...     axes=[
    ...         v06.SpaceAxis(name="y", unit="micrometer"),
    ...         v06.SpaceAxis(name="x", unit="micrometer"),
    ...     ],
    ... )
    >>> scene = v06.Scene(
    ...     scene=v06.SceneDef(
    ...         coordinateSystems=[world],
    ...         coordinateTransformations=[
    ...             v06.TranslationTransformation(
    ...                 translation=[0, 32 * i],
    ...                 input={"path": f"tile_{i}", "name": "intrinsic"},
    ...                 output={"name": "world"},
    ...             )
    ...             for i in range(2)
    ...         ],
    ...     )
    ... )
    >>> images = {
    ...     f"tile_{i}": (make_image(), np.zeros((64, 64), dtype=np.uint16))
    ...     for i in range(2)
    ... }
    >>> result = write_scene("my_scene.zarr", scene, images)
    >>> (result / "tile_1" / "zarr.json").exists()
    True
    """
    builder = SceneBuilder(
        dest,
        scene,
        extra_attributes=extra_attributes,
        writer=writer,
        chunks=chunks,
        shards=shards,
        overwrite=overwrite,
        compression=compression,
    )
    # (check everything first, so a bad image doesn't leave a partial scene behind)
    seen: list[str] = []
    for path, (image_model, datasets) in images.items():
        builder._check_image(path, image_model, datasets, others=seen)
        seen.append(path)
    _check_scene(scene, {path: image for path, (image, _) in images.items()})

    builder._ensure_initialized()
    for path, (image_model, datasets) in images.items():
        builder.write_image(path, image_model, datasets, progress=progress)

    return builder.root_path


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
    TypeError
        If given a model from another OME-Zarr version (e.g. a v05.Image).
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
    ...                 DimSpec(name="y", scale=0.5, unit="micrometer"),
    ...                 DimSpec(name="x", scale=0.5, unit="micrometer"),
    ...             ]
    ...         )
    ...     ]
    ... )
    >>> # Prepare with just shape/dtype (no data yet) - no list wrapping!
    >>> path, arrays = prepare_image("prepared.zarr", image, ((64, 64), "uint16"))
    >>> arrays["0"][:] = np.zeros((64, 64), dtype=np.uint16)
    >>> assert path.exists()
    """
    return _prepare_image(
        Path(dest),
        image,
        datasets,
        image_type=Image,
        extra_attributes=extra_attributes,
        chunks=chunks,
        shards=shards,
        writer=writer,
        overwrite=overwrite,
        compression=compression,
    )


class Bf2RawBuilder(Bf2RawBuilderBase[Image]):
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
    extra_attributes : dict[str, Any] | None, optional
        Additional attributes to write alongside "ome" in the root zarr.json.
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

    _image_type = Image
    _bf2raw_type = Bf2Raw
    _series_type = Series

    def __init__(
        self,
        dest: str | PathLike,
        *,
        ome_xml: str | None = None,
        extra_attributes: dict[str, Any] | None = None,
        writer: ZarrWriter = "auto",
        chunks: ShapeLike | Literal["auto"] | None = "auto",
        shards: ShapeLike | None = None,
        overwrite: bool = False,
        compression: CompressionName = "blosc-zstd",
    ) -> None:
        super().__init__(
            Path(dest),
            ome_xml=ome_xml,
            extra_attributes=extra_attributes,
            writer=writer,
            chunks=chunks,
            shards=shards,
            overwrite=overwrite,
            compression=compression,
        )


class PlateBuilder(PlateBuilderBase[Image]):
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
    extra_attributes : dict[str, Any] | None, optional
        Additional attributes to write alongside "ome" in the root zarr.json.
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

    _image_type = Image
    _plate_type = Plate
    _well_type = Well
    _well_def_type = WellDef
    _field_of_view_type = FieldOfView

    def __init__(
        self,
        dest: str | PathLike,
        *,
        plate: Plate | None = None,
        extra_attributes: dict[str, Any] | None = None,
        writer: ZarrWriter = "auto",
        chunks: ShapeLike | Literal["auto"] | None = "auto",
        shards: ShapeLike | None = None,
        overwrite: bool = False,
        compression: CompressionName = "blosc-zstd",
    ) -> None:
        super().__init__(
            Path(dest),
            plate=plate,
            extra_attributes=extra_attributes,
            writer=writer,
            chunks=chunks,
            shards=shards,
            overwrite=overwrite,
            compression=compression,
        )


class LabelsBuilder(LabelsBuilderBase[LabelImage]):
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

    _image_type = Image
    _labels_group_type = LabelsGroup

    def __init__(
        self,
        dest: str | PathLike,
        *,
        writer: ZarrWriter = "auto",
        chunks: ShapeLike | Literal["auto"] | None = "auto",
        shards: ShapeLike | None = None,
        overwrite: bool = False,
        compression: CompressionName = "blosc-zstd",
    ) -> None:
        super().__init__(
            Path(dest),
            writer=writer,
            chunks=chunks,
            shards=shards,
            overwrite=overwrite,
            compression=compression,
        )


class SceneBuilder:
    """Builder for OME-Zarr v0.6 scene hierarchies.

    A scene group holds [`Scene`][yaozarrs.v06.Scene] metadata, describing the
    spatial relationship between the images in its subgroups.  Unlike a plate's, the
    scene metadata is not generated: it is given up front, and written as is.

    This builder supports two workflows:

    1. **Immediate write** (simpler): Use `write_image()` to write each image with
       its data immediately.  Each image is checked against the scene's references to
       it (that the coordinate systems they name exist) as it is written; that every
       referenced image gets written is up to you.
    2. **Prepare-only** (flexible): Use `add_image()` to register all images, then
       `prepare()` to create the hierarchy with empty arrays.  `prepare()` checks the
       scene as a whole first (as [`write_scene`][yaozarrs.write.v06.write_scene]
       does): every referenced image is there, and the graph is connected.

    `chunks`, `shards` and `compression` are defaults for every image: `write_image()`
    and `add_image()` can override them per image (e.g. for images with different
    numbers of dimensions).

    See Also
    --------
    - [`write_scene`][yaozarrs.write.v06.write_scene] : High-level function to write
      all images at once.

    Parameters
    ----------
    dest : str | PathLike
        Destination path for the scene Zarr group.
    scene : Scene
        OME-Zarr Scene metadata model.
    extra_attributes : dict[str, Any] | None, optional
        Additional attributes to write alongside "ome" in the scene's zarr.json.
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
    >>> import numpy as np
    >>> from yaozarrs import DimSpec, v06
    >>> from yaozarrs.write.v06 import SceneBuilder
    >>>
    >>> def make_image():
    ...     return v06.Image(
    ...         multiscales=[
    ...             v06.Multiscale.from_dims([DimSpec(name="y"), DimSpec(name="x")])
    ...         ]
    ...     )
    >>> scene = v06.Scene(
    ...     scene=v06.SceneDef(
    ...         coordinateTransformations=[
    ...             v06.TranslationTransformation(
    ...                 translation=[0, 100],
    ...                 input={"path": "b", "name": "intrinsic"},
    ...                 output={"path": "a", "name": "intrinsic"},
    ...             )
    ...         ]
    ...     )
    ... )

    **Immediate write workflow:**

    >>> builder = SceneBuilder("scene_immediate.zarr", scene)
    >>> builder.write_image("a", make_image(), np.zeros((32, 32), dtype=np.uint16))
    <SceneBuilder: 1 images>
    >>> builder.write_image("b", make_image(), np.zeros((32, 32), dtype=np.uint16))
    <SceneBuilder: 2 images>

    **Prepare-only workflow:**

    >>> builder2 = SceneBuilder("scene_prepare.zarr", scene)
    >>> builder2.add_image("a", make_image(), ((32, 32), np.uint16))
    <SceneBuilder: 1 images>
    >>> builder2.add_image("b", make_image(), ((32, 32), np.uint16))
    <SceneBuilder: 2 images>
    >>> path, arrays = builder2.prepare()
    >>> arrays["b/0"][:] = np.zeros((32, 32), dtype=np.uint16)
    """

    def __init__(
        self,
        dest: str | PathLike,
        scene: Scene,
        *,
        extra_attributes: dict[str, Any] | None = None,
        writer: ZarrWriter = "auto",
        chunks: ShapeLike | Literal["auto"] | None = "auto",
        shards: ShapeLike | None = None,
        overwrite: bool = False,
        compression: CompressionName = "blosc-zstd",
    ) -> None:
        _check_model(scene, Scene, "scene: ")
        self._dest = Path(dest)
        self._scene = scene
        self._extra_attributes = extra_attributes
        self._writer: ZarrWriter = writer
        self._chunks: ShapeLike | Literal["auto"] | None = chunks
        self._shards = shards
        self._overwrite = overwrite
        self._compression: CompressionName = compression

        # For prepare-only workflow: {path: (image, dataset_specs, array_options)}
        self._images: dict[str, AddedImage] = {}
        # For immediate write workflow: {path: image}
        self._initialized = False
        self._written_images: dict[str, Image] = {}

    @property
    def root_path(self) -> Path:
        """Path to the scene group."""
        return self._dest

    def write_image(
        self,
        path: str,
        image: Image,
        datasets: ArrayOrPyramid,
        *,
        chunks: ShapeLike | Literal["auto"] | _Default | None = _Default.BUILDER,
        shards: ShapeLike | _Default | None = _Default.BUILDER,
        compression: CompressionName | _Default = _Default.BUILDER,
        progress: bool = False,
    ) -> Self:
        """Write an image of the scene immediately, with its data.

        The scene group is created along with the first image.

        Parameters
        ----------
        path : str
            Path of the image subgroup, relative to the scene group (e.g. "tile_0",
            or "tiles/0").
        image : Image
            OME-Zarr Image metadata model for this image.
        datasets : ArrayLike | Sequence[ArrayLike]
            Data array(s) for each resolution level. For a single dataset,
            pass the array directly without wrapping in a list.
        chunks : tuple[int, ...] | "auto" | None, optional
            Chunk shape for this image's arrays.  Default is the builder's.
        shards : tuple[int, ...] | None, optional
            Shard shape for this image's arrays.  Default is the builder's.
        compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
            Compression codec for this image's arrays.  Default is the builder's.
        progress : bool, optional
            Show progress bar when writing dask arrays. Default is False.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If the path is invalid or already used, if the scene references a
            coordinate system this image doesn't have, or if datasets don't match
            the metadata.
        NotImplementedError
            If the Image has multiple multiscales.
        """
        self._check_image(path, image, datasets, chunks=chunks, shards=shards)
        # (only the references to this image: the others may not be written yet)
        _check_scene(self._scene, {path: image}, complete=False)

        self._ensure_initialized()
        _ensure_parent_groups(self._dest, path)
        _write_image(
            self._dest / path,
            image,
            datasets,
            image_type=Image,
            writer=self._writer,
            overwrite=self._overwrite,
            progress=progress,
            **self._array_options(chunks, shards, compression),
        )
        self._written_images[path] = image
        return self

    def add_image(
        self,
        path: str,
        image: Image,
        datasets: ShapeAndDTypeOrPyramid,
        *,
        chunks: ShapeLike | Literal["auto"] | _Default | None = _Default.BUILDER,
        shards: ShapeLike | _Default | None = _Default.BUILDER,
        compression: CompressionName | _Default = _Default.BUILDER,
    ) -> Self:
        """Add an image for the prepare-only workflow.

        Registers an image to be created when `prepare()` is called.

        Parameters
        ----------
        path : str
            Path of the image subgroup, relative to the scene group (e.g. "tile_0",
            or "tiles/0").
        image : Image
            OME-Zarr Image metadata model for this image.
        datasets : ShapeAndDType | Sequence[ShapeAndDType]
            Shape and dtype specification(s) for each resolution level, as
            `(shape, dtype)` tuples. For a single dataset, pass the tuple
            directly without wrapping in a list.
        chunks : tuple[int, ...] | "auto" | None, optional
            Chunk shape for this image's arrays.  Default is the builder's.
        shards : tuple[int, ...] | None, optional
            Shard shape for this image's arrays.  Default is the builder's.
        compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
            Compression codec for this image's arrays.  Default is the builder's.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If the path is invalid or already used, or if the number of dataset
            specs doesn't match the metadata.
        NotImplementedError
            If the Image has multiple multiscales.
        """
        _, specs = self._check_image(
            path, image, datasets, chunks=chunks, shards=shards
        )
        options = self._array_options(chunks, shards, compression)
        self._images[path] = (image, specs, options)
        return self

    def prepare(self) -> tuple[Path, dict[str, Any]]:
        """Create the Zarr hierarchy and return array handles.

        Checks the scene against all its images (those added with `add_image()` and
        any already written with `write_image()`), then creates the scene group and
        empty arrays for every added image.

        Returns
        -------
        tuple[Path, dict[str, Any]]
            A tuple of (root_path, arrays) where `arrays` maps composite keys
            like `"tile_0/0"` (image path / dataset path) to array objects. The
            array type depends on the configured writer (zarr.Array or
            tensorstore.TensorStore).

        Raises
        ------
        ValueError
            If no images have been added with `add_image()`, if the scene
            references an image (or coordinate system) that isn't there, or if
            its coordinate systems aren't all connected.
        FileExistsError
            If destination exists and `overwrite` is False.
        ImportError
            If no suitable writer backend is installed.
        """
        if not self._images:
            raise ValueError("No images added. Use add_image() before prepare().")
        added = {path: image for path, (image, *_) in self._images.items()}
        _check_scene(self._scene, {**self._written_images, **added})

        self._ensure_initialized()
        all_arrays: dict[str, Any] = {}
        for path, (image, specs, options) in self._images.items():
            _ensure_parent_groups(self._dest, path)
            _, arrays = _prepare_image(
                self._dest / path,
                image,
                specs,
                image_type=Image,
                writer=self._writer,
                overwrite=self._overwrite,
                **options,
            )
            for dataset_path, arr in arrays.items():
                all_arrays[f"{path}/{dataset_path}"] = arr
        return self._dest, all_arrays

    def __repr__(self) -> str:
        total_images = len(self._images) + len(self._written_images)
        return f"<{self.__class__.__name__}: {total_images} images>"

    # ------------------ Internal methods ------------------

    def _array_options(
        self,
        chunks: ShapeLike | Literal["auto"] | _Default | None,
        shards: ShapeLike | _Default | None,
        compression: CompressionName | _Default,
    ) -> dict[str, Any]:
        """Resolve per-image array settings, falling back to the builder's."""
        return {
            "chunks": self._chunks if chunks is _Default.BUILDER else chunks,
            "shards": self._shards if shards is _Default.BUILDER else shards,
            "compression": (
                self._compression if compression is _Default.BUILDER else compression
            ),
        }

    def _check_image(
        self,
        path: str,
        image: Any,
        datasets: Any,
        others: Iterable[str] | None = None,
        chunks: ShapeLike | Literal["auto"] | _Default | None = _Default.BUILDER,
        shards: ShapeLike | _Default | None = _Default.BUILDER,
    ) -> tuple[Any, Sequence[Any]]:
        """Check an image (and its path) before anything of it is written.

        `others` are the paths it mustn't clash with (default: those already added).
        """
        _check_image_path(path)
        if others is None:
            others = (*self._written_images, *self._images)
        for other in others:
            if other == path:
                raise ValueError(f"Image '{path}' was already added to the scene.")
            if other.startswith(f"{path}/") or path.startswith(f"{other}/"):
                raise ValueError(
                    f"Image '{path}' can't be nested in (or contain) image '{other}'."
                )
        options = self._array_options(chunks, shards, self._compression)
        multiscale, datasets_seq = _check_image_args(
            image,
            datasets,
            Image,
            f"Image '{path}': ",
            options["chunks"],
            options["shards"],
        )
        for item, dataset in zip(datasets_seq, multiscale.datasets):
            shape = item.shape if hasattr(item, "shape") else item[0]
            if len(shape) != multiscale.ndim:
                raise ValueError(
                    f"Image '{path}': dataset '{dataset.path}' has {len(shape)} "
                    f"dimensions, but the image has {multiscale.ndim} axes."
                )
        return multiscale, datasets_seq

    def _ensure_initialized(self) -> None:
        """Create the scene group (with the scene metadata) if not already done."""
        if self._initialized:
            return
        _get_create_func(self._writer)  # (raises for an unknown/missing backend)
        _create_zarr3_group(
            self._dest,
            self._scene,
            self._overwrite,
            extra_attributes=self._extra_attributes,
        )
        self._initialized = True


def _check_image_path(path: str) -> None:
    # (the path of each node below the scene group: Zarr node names must be
    # non-empty, not only periods, and not start with "__")
    # (and "zarr.json" would collide with a group's own metadata)
    parts = path.split("/")
    if any(
        not p or set(p) == {"."} or p.startswith("__") or p == "zarr.json"
        for p in parts
    ):
        raise ValueError(
            f"Invalid image path {path!r}: each part must be a valid Zarr node name "
            "(non-empty, not only periods, not starting with '__', not 'zarr.json')."
        )


def _check_scene(
    scene: Scene, images: Mapping[str, Image], complete: bool = True
) -> None:
    """Check a scene's references against its `images` (before writing them).

    Each `input`/`output` with a `path` must name a coordinate system of the image at
    that path.  If `complete`, every referenced image must be in `images` (or be a
    multiscale image inside one, such as one of its labels, which isn't checked), and
    the graph of coordinate systems must be connected.  Where both ends of a
    transformation are known, its dimensions must fit them.
    """
    defs = scene.scene
    systems = {cs.name: cs for cs in defs.coordinateSystems or ()}
    nodes: set[Node] = {(None, 0, name) for name in systems}
    edges: list[tuple[Node, Node]] = []
    entries: set[tuple[str, int]] = set()
    resolved_all = True
    for i, t in enumerate(defs.coordinateTransformations):
        ends: list[tuple[Node, CoordinateSystem]] = []
        for side in ("input", "output"):
            ref = getattr(t, side)
            loc = f"scene coordinateTransformations[{i}] {side}"
            if ref.path is None:
                ends.append(((None, 0, ref.name), systems[ref.name]))
                continue
            ref_path = posixpath.normpath(ref.path)
            if (image := images.get(ref_path)) is not None:
                if (found := _find_coordinate_system(image, ref.name)) is None:
                    names = [
                        cs.name
                        for ms in image.multiscales
                        for cs in ms.coordinateSystems
                    ]
                    raise ValueError(
                        f"{loc}: image '{ref_path}' has no coordinate system "
                        f"'{ref.name}' (it has {names})"
                    )
                ends.append(((ref_path, found[0], ref.name), found[1]))
                entries.add((ref_path, found[0]))
            elif any(ref_path.startswith(f"{path}/") for path in images):
                resolved_all = False  # (e.g. a label image of one of the images)
            elif complete:
                raise ValueError(
                    f"{loc}: refers to '{ref_path}', which is not one of the scene's "
                    f"images ({sorted(images)})"
                )
            else:
                resolved_all = False
        if len(ends) == 2:
            (a, cs_in), (b, cs_out) = ends
            edges.append((a, b))
            try:
                check_transform_ndim(t, cs_in.ndim, cs_out.ndim)
            except ValueError as e:
                raise ValueError(
                    f"scene coordinateTransformations[{i}] ({t.type}), from "
                    f"{_fmt(a)!r} to {_fmt(b)!r}: {e}"
                ) from None

    if not complete or not resolved_all:
        return
    for path, ms_idx in entries:
        ms_nodes, ms_edges = _multiscale_graph(
            path, ms_idx, images[path].multiscales[ms_idx]
        )
        nodes |= ms_nodes
        edges += ms_edges
    # The coordinate systems defined in the multiscales metadata and the `scene`
    # metadata combined with the coordinate transformations form a transformations
    # graph. ... The graph MUST be connected
    if len(groups := _components(nodes, edges)) > 1:
        raise ValueError(
            "The coordinate systems of the scene and its images are not all linked by "
            f"coordinate transformations: {groups}"
        )
