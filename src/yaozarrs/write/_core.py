"""Version-agnostic OME-Zarr writing machinery shared by `yaozarrs.write.v0X`.

Everything about *how* an OME-Zarr hierarchy is written (group/array creation,
chunk/shard resolution, backends, builders, plate-metadata auto-generation) is
identical across the Zarr v3-based spec versions (v0.5, v0.6). The only
version-specific pieces are *which* metadata model classes get built/accepted,
so those are injected via an [`OMEModels`][] bundle:

- module-level functions here take a `models=` argument, and
- the builder base classes read a `_models` class attribute that each version's
  concrete subclass sets.

The public, documented API lives in `yaozarrs.write.v05` / `yaozarrs.write.v06`:
thin wrappers (with version-specific docstrings and examples) around this module.
To make a version-specific fix, override/wrap in that version's module rather
than editing here.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel
from typing_extensions import Self

from yaozarrs._plate_common import Column, PlateWell, Row

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from os import PathLike

    import tensorstore
    import zarr
    from numpy.typing import DTypeLike
    from typing_extensions import Literal, TypeAlias

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

    # (Image, data) or (Image, specs) pairs. The Image type is version-specific,
    # so it's left as Any here; the version modules narrow it.
    ImageWithDatasets: TypeAlias = tuple[Any, ArrayOrPyramid]
    ImageWithShapeSpecs: TypeAlias = tuple[Any, ShapeAndDTypeOrPyramid]
    ImageWithAny: TypeAlias = ImageWithDatasets | ImageWithShapeSpecs

ImageT = TypeVar("ImageT")
LabelImageT = TypeVar("LabelImageT")
PlateT = TypeVar("PlateT")


@dataclass(frozen=True)
class OMEModels:
    """The version-specific metadata model classes a writer accepts/builds."""

    version: str  # e.g. "0.5", only used in error messages
    Image: type[Any]
    LabelImage: type[Any]
    LabelsGroup: type[Any]
    Plate: type[Any]
    PlateDef: type[Any]
    Well: type[Any]
    WellDef: type[Any]
    FieldOfView: type[Any]
    Bf2Raw: type[Any]
    Series: type[Any]

    def check(self, obj: Any, expected: type[Any], what: str = "") -> None:
        """Raise `TypeError` if `obj` is not an instance of `expected`.

        Guards against e.g. passing a `yaozarrs.v05.Image` to the v0.6 writer,
        which would otherwise silently write v0.5 metadata into a v0.6 store.
        """
        if not isinstance(obj, expected):
            exp = f"{expected.__module__.split('._')[0]}.{expected.__name__}"
            got = type(obj)
            raise TypeError(
                f"{what}expected a {exp} model for OME-Zarr v{self.version}, got "
                f"{got.__module__.split('._')[0]}.{got.__name__}."
            )


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
            (e.g., `zarr.Array` or `tensorstore.TensorStore`).
        """
        ...


# ######################## Module-level Functions ##############################
# (see the version modules for the documented public signatures)


def write_image(
    dest: str | PathLike,
    image: Any,
    datasets: ArrayOrPyramid,
    *,
    models: OMEModels,
    labels_builder: type[LabelsBuilderBase] | None = None,
    labels: Mapping[str, tuple[Any, ArrayOrPyramid]] | None = None,
    extra_attributes: dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write an Image group (and optional labels) with data."""
    models.check(image, models.Image)
    multiscale, datasets_seq = _validate_and_normalize_datasets(image, datasets)
    # validate labels up front, so a bad label doesn't leave a half-written image
    if labels:
        if labels_builder is None:  # pragma: no cover
            raise TypeError("labels_builder is required when writing labels")
        for label_name, (label_image, label_datasets) in labels.items():
            models.check(label_image, models.LabelImage, f"Label '{label_name}': ")
            _validate_and_normalize_datasets(
                label_image, label_datasets, f"Label '{label_name}': "
            )

    # Extract specs from arrays for prepare_image
    specs: list[ShapeAndDType] = [(arr.shape, arr.dtype) for arr in datasets_seq]

    # Create arrays using prepare_image
    dest_path, arrays = prepare_image(
        dest,
        image,
        specs,
        models=models,
        extra_attributes=extra_attributes,
        chunks=chunks,
        shards=shards,
        writer=writer,
        overwrite=overwrite,
        compression=compression,
    )

    # Write data to arrays
    for data_array, dataset_meta in zip(datasets_seq, multiscale.datasets):
        _write_to_array(arrays[dataset_meta.path], data_array, progress=progress)

    # Write labels if provided
    if labels and labels_builder is not None:
        builder = labels_builder(
            dest_path / "labels",
            writer=writer,
            chunks=chunks,
            shards=shards,
            overwrite=overwrite,
            compression=compression,
        )
        for label_name, (label_image, label_datasets) in labels.items():
            builder.write_label(
                label_name, label_image, label_datasets, progress=progress
            )

    return dest_path


def write_plate(
    dest: str | PathLike,
    images: Mapping[tuple[str, str, str], ImageWithDatasets],
    *,
    plate_builder: type[PlateBuilderBase],
    plate: Any | dict[str, Any] | None = None,
    extra_attributes: dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write a Plate group with data (via `plate_builder`)."""
    # Merge user-provided plate metadata with auto-generated
    plate_obj = _merge_plate_metadata(images, plate, plate_builder._models)

    # Use PlateBuilder to handle the writing
    builder = plate_builder(
        dest,
        plate=plate_obj,
        extra_attributes=extra_attributes,
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
        builder.write_well(row=row, col=col, images=fields_data, progress=progress)

    return builder.root_path


def write_bioformats2raw(
    dest: str | PathLike,
    images: Mapping[str, ImageWithDatasets],
    *,
    bf2raw_builder: type[Bf2RawBuilderBase],
    ome_xml: str | None = None,
    extra_attributes: dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write a bioformats2raw-layout collection (via `bf2raw_builder`)."""
    builder = bf2raw_builder(
        dest,
        ome_xml=ome_xml,
        extra_attributes=extra_attributes,
        writer=writer,
        chunks=chunks,
        shards=shards,
        overwrite=overwrite,
        compression=compression,
    )

    for series_name, (image_model, datasets) in images.items():
        builder.write_image(series_name, image_model, datasets, progress=progress)

    return builder.root_path


def prepare_image(
    dest: str | PathLike,
    image: Any,
    datasets: ShapeAndDTypeOrPyramid,
    *,
    models: OMEModels,
    extra_attributes: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
) -> tuple[Path, dict[str, Any]]:
    """Create an Image group with empty arrays, returning the array handles."""
    models.check(image, models.Image)
    if len(image.multiscales) != 1:
        raise NotImplementedError("Image must have exactly one multiscale")

    multiscale = image.multiscales[0]

    # Normalize to sequence: single (shape, dtype) tuple -> list
    datasets_seq: Sequence[ShapeAndDType]
    if _is_shape_and_dtype(datasets):
        datasets_seq = [datasets]
    else:
        datasets_seq = cast("Sequence[ShapeAndDType]", datasets)

    if len(datasets_seq) != len(multiscale.datasets):
        raise ValueError(
            f"Number of dataset specs ({len(datasets_seq)}) must match "
            f"number of datasets in metadata ({len(multiscale.datasets)})"
        )

    # Get create function
    create_func = _get_create_func(writer)

    # Create zarr group with Image metadata
    dest_path = Path(dest)
    _create_zarr3_group(dest_path, image, overwrite, extra_attributes=extra_attributes)

    dimension_names = [ax.name for ax in multiscale.axes]

    # FIXME: numpy is not listed in any of our extras...
    import numpy as np

    # Create arrays for each dataset
    arrays = {}
    for (shape, dtype_spec), dataset_meta in zip(datasets_seq, multiscale.datasets):
        # Convert dtype to np.dtype to ensure compatibility with all backends
        dtype = np.dtype(dtype_spec)
        resolved_chunks = _resolve_chunks(shape, dtype, chunks)
        arrays[dataset_meta.path] = create_func(
            path=dest_path / dataset_meta.path,
            shape=shape,
            dtype=dtype,
            chunks=resolved_chunks,
            shards=_resolve_shards(shape, resolved_chunks, shards),
            dimension_names=dimension_names,
            overwrite=overwrite,
            compression=compression,
        )

    return dest_path, arrays


# ######################## Builder Base Classes ################################
# Each version module subclasses these, setting `_models` and the (versioned)
# class docstring. Method docstrings here are shared by all versions.


class Bf2RawBuilderBase(Generic[ImageT]):
    """Shared implementation of the version-specific `Bf2RawBuilder`."""

    _models: ClassVar[OMEModels]

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
        self._dest = Path(dest)
        self._ome_xml = ome_xml
        self._extra_attributes = extra_attributes
        self._writer: ZarrWriter = writer
        self._chunks: ShapeLike | Literal["auto"] | None = chunks
        self._shards = shards
        self._overwrite = overwrite
        self._compression: CompressionName = compression
        self._indent = 2

        # For prepare-only workflow: {series_name: (image, dataset_specs)}
        self._series: dict[str, ImageWithShapeSpecs] = {}

        # For immediate write workflow
        self._initialized = False
        self._written_series: list[str] = []

    @property
    def root_path(self) -> Path:
        """Path to the root of the bioformats2raw hierarchy."""
        return self._dest

    def write_image(
        self,
        name: str,
        image: ImageT,
        datasets: ArrayOrPyramid,
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
        datasets : ArrayLike | Sequence[ArrayLike]
            Data array(s) for each resolution level. For a single dataset,
            pass the array directly without wrapping in a list.
        progress : bool, optional
            Show progress bar when writing dask arrays. Default is False.

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
        self._models.check(image, self._models.Image, f"Series '{name}': ")
        _validate_and_normalize_datasets(image, datasets, f"Series '{name}': ")

        # Initialize root structure if needed
        self._ensure_initialized()

        # Write the series using the existing write_image function
        write_image(
            self._dest / name,
            image,
            datasets,
            models=self._models,
            writer=self._writer,
            chunks=self._chunks,
            shards=self._shards,
            overwrite=self._overwrite,
            compression=self._compression,
            progress=progress,
        )

        # Only now register the series in OME/zarr.json, so that a failed write
        # doesn't leave the series list pointing at a group that doesn't exist.
        self._update_ome_series(name)

        return self

    def add_series(
        self,
        name: str,
        image: ImageT,
        datasets: ShapeAndDTypeOrPyramid,
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
        datasets : ShapeAndDType | Sequence[ShapeAndDType]
            Shape and dtype specification(s) for each resolution level, as
            `(shape, dtype)` tuples. For a single dataset, pass the tuple
            directly without wrapping in a list.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If a series with this name was already added or written, or if the
            number of dataset specs doesn't match the metadata.
        NotImplementedError
            If the Image has multiple multiscales.
        """
        self._validate_series_name(name)
        self._models.check(image, self._models.Image, f"Series '{name}': ")
        _, datasets_seq = _validate_and_normalize_datasets(
            image, datasets, f"Series '{name}': "
        )
        self._series[name] = (image, datasets_seq)
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
        bf2raw = self._models.Bf2Raw(bioformats2raw_layout=3)
        _create_zarr3_group(
            self._dest,
            bf2raw,
            self._overwrite,
            extra_attributes=self._extra_attributes,
        )

        # Create OME/zarr.json with series list
        ome_path = self._dest / "OME"
        series_model = self._models.Series(series=list(self._series))
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
                models=self._models,
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

    def __repr__(self) -> str:
        total_images = len(self._series) + len(self._written_series)
        return f"<{self.__class__.__name__}: {total_images} images>"

    # ------------------------ Internal Methods --------------------------

    def _validate_series_name(self, name: str) -> None:
        if name in self._written_series:
            raise ValueError(f"Series '{name}' already written via write_image().")
        if name in self._series:
            raise ValueError(f"Series '{name}' already added via add_series().")

    def _ensure_initialized(self) -> None:
        """Create root structure if not already done."""
        if self._initialized:
            return

        # Create root zarr.json with bioformats2raw.layout
        bf2raw = self._models.Bf2Raw(bioformats2raw_layout=3)
        _create_zarr3_group(
            self._dest,
            bf2raw,
            self._overwrite,
            extra_attributes=self._extra_attributes,
        )

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
        series_model = self._models.Series(series=self._written_series)
        zarr_json_path = self._dest / "OME" / "zarr.json"
        # Preserve existing extra attributes if present
        existing_extra: dict[str, Any] = {}
        if zarr_json_path.exists():
            existing = json.loads(zarr_json_path.read_text())
            existing_extra = {
                k: v for k, v in existing.get("attributes", {}).items() if k != "ome"
            }
        zarr_json = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": series_model.model_dump(mode="json", exclude_none=True),
                **existing_extra,
            },
        }
        zarr_json_path.write_text(json.dumps(zarr_json, indent=self._indent))


class PlateBuilderBase(Generic[ImageT, PlateT]):
    """Shared implementation of the version-specific `PlateBuilder`."""

    _models: ClassVar[OMEModels]

    def __init__(
        self,
        dest: str | PathLike,
        *,
        plate: PlateT | None = None,
        extra_attributes: dict[str, Any] | None = None,
        writer: ZarrWriter = "auto",
        chunks: ShapeLike | Literal["auto"] | None = "auto",
        shards: ShapeLike | None = None,
        overwrite: bool = False,
        compression: CompressionName = "blosc-zstd",
    ) -> None:
        if plate is not None:
            self._models.check(plate, self._models.Plate, "plate: ")
        self._dest = Path(dest)
        self._user_plate = plate  # Store user-provided plate (if any)
        self._extra_attributes = extra_attributes
        self._writer: ZarrWriter = writer
        self._chunks: ShapeLike | Literal["auto"] | None = chunks
        self._shards = shards
        self._overwrite = overwrite
        self._compression: CompressionName = compression

        # For prepare-only workflow: {well_path: {fov: (Image, specs)}}
        self._wells: dict[str, dict[str, ImageWithShapeSpecs]] = {}

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

    def write_well(
        self,
        row: str,
        col: str,
        images: Mapping[str, ImageWithDatasets],
        *,
        progress: bool = False,
    ) -> Self:
        """Write a well immediately with its `images` (fields of view) and data.

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
        images : Mapping[str, ImageWithDatasets]
            Mapping of `{fov -> (image_model, datasets)}` where:
            - fov: Field of view identifier like "0", "1", etc.
            - datasets can be:
              - Single array (for one dataset): `{"0": (image, data)}`
              - Sequence (for multiple datasets): `{"0": (image, [data1, data2])}`
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

        # Validate + normalize fields (convert single arrays to sequences) before
        # writing any metadata
        normalized_fields: dict[str, tuple[ImageT, Sequence[ArrayLike]]] = {}
        for fov, (image_model, datasets) in images.items():
            ctx = f"Well '{row}/{col}', field '{fov}': "
            self._models.check(image_model, self._models.Image, ctx)
            _, datasets_seq = _validate_and_normalize_datasets(
                image_model, datasets, ctx
            )
            normalized_fields[fov] = (image_model, datasets_seq)

        # Track this well's data before writing
        self._written_wells_data[(row, col)] = cast(
            "dict[str, ImageWithDatasets]", normalized_fields
        )

        # Update plate metadata with the new well
        self._update_plate_metadata()

        # Generate Well metadata for this well and create well subgroup
        well_group_path = self._dest / f"{row}/{col}"
        well_metadata = self._generate_well_metadata(list(images))
        _create_zarr3_group(well_group_path, well_metadata, self._overwrite)

        # Write each field of view
        for fov, (image_model, datasets_seq) in normalized_fields.items():
            field_path = well_group_path / fov
            write_image(
                field_path,
                image_model,
                datasets_seq,
                models=self._models,
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
        images: Mapping[str, ImageWithShapeSpecs],
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
        images : Mapping[str, ImageWithShapeSpecs]
            Mapping of `{fov -> (image_model, specs)}` where specs provide the
            dtype and shape for each resolution level:
            - Single level: `(image, (shape, dtype))`
            - Multiple levels: `(image, [(shape1, dtype1), (shape2, dtype2)])`

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If row/col combination was already added/written, or if a user-
            provided Plate doesn't include this well, or if field specs
            don't match Image metadata.
        NotImplementedError
            If any Image has multiple multiscales.
        """
        # Validate well hasn't been used
        self._validate_well_coordinates(row, col)

        # Validate and normalize all fields before accepting
        well_path = f"{row}/{col}"
        normalized_fields: dict[str, ImageWithShapeSpecs] = {}

        for fov, (image_model, specs) in images.items():
            ctx = f"Well '{well_path}', field '{fov}': "
            self._models.check(image_model, self._models.Image, ctx)
            _, specs_seq = _validate_and_normalize_datasets(image_model, specs, ctx)
            normalized_fields[fov] = (image_model, specs_seq)

        self._wells[well_path] = normalized_fields
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
        plate = _merge_plate_metadata(
            self._get_images_dict(), self._user_plate, self._models
        )

        # Create plate zarr.json
        _create_zarr3_group(
            self._dest,
            plate,
            self._overwrite,
            extra_attributes=self._extra_attributes,
        )

        # Create arrays for each well/field combination
        all_arrays: dict[str, Any] = {}

        for well_path, fields in self._wells.items():
            # Generate Well metadata and group
            well_metadata = self._generate_well_metadata(list(fields))
            well_group_path = self._dest / well_path
            _create_zarr3_group(well_group_path, well_metadata, self._overwrite)

            # Create arrays for each field
            for fov, (image_model, datasets) in fields.items():
                field_path = well_group_path / fov

                _field_path, field_arrays = prepare_image(
                    field_path,
                    image_model,
                    datasets,
                    models=self._models,
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

    def __repr__(self) -> str:
        total_wells = len(self._wells) + len(self._written_wells_data)
        return f"<{self.__class__.__name__}: {total_wells} wells>"

    # ------------------ Internal methods ------------------

    def _validate_well_coordinates(self, row: str, col: str) -> None:
        """Validate that well coordinates haven't been used yet."""
        # Check if already written or added
        well_coords = (row, col)
        if well_coords in self._written_wells_data:
            raise ValueError(f"Well ({row}, {col}) already written via write_well().")
        well_path = f"{row}/{col}"
        if well_path in self._wells:
            raise ValueError(f"Well ({row}, {col}) already added via add_well().")

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

    def _get_images_dict(self) -> dict[tuple[str, str, str], ImageWithAny]:
        """Convert internal well storage to images dict format.

        Combines both `_written_wells_data` (immediate write workflow) and
        `_wells` (prepare-only workflow) into a single images dict.
        """
        images_dict: dict[tuple[str, str, str], ImageWithAny] = {}
        # From immediate write workflow
        for (row, col), fields in self._written_wells_data.items():
            for fov, image_data in fields.items():
                images_dict[(row, col, fov)] = image_data
        # From prepare-only workflow
        for well_path, fields in self._wells.items():
            row, col = well_path.split("/")
            for fov, image_data in fields.items():
                images_dict[(row, col, fov)] = image_data
        return images_dict

    def _generate_current_plate_metadata(self) -> PlateT:
        """Generate plate metadata from currently written wells.

        If user provided a Plate, use that. Otherwise, auto-generate from
        written wells (similar to write_plate auto-generation).
        """
        if self._user_plate is not None:
            return self._user_plate
        return _merge_plate_metadata(
            self._get_images_dict(), self._user_plate, self._models
        )

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
                **(self._extra_attributes or {}),
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

    def _generate_well_metadata(self, field_names: list[str]) -> Any:
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
            self._models.FieldOfView(path=fov, acquisition=None)
            for fov in sorted(field_names)
        ]

        return self._models.Well(well=self._models.WellDef(images=images))


class LabelsBuilderBase(Generic[LabelImageT]):
    """Shared implementation of the version-specific `LabelsBuilder`."""

    _models: ClassVar[OMEModels]

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
        self._dest = Path(dest)
        self._writer: ZarrWriter = writer
        self._chunks: ShapeLike | Literal["auto"] | None = chunks
        self._shards = shards
        self._overwrite = overwrite
        self._compression: CompressionName = compression

        # For prepare-only workflow: {label_name: (LabelImage, specs)}
        self._labels: dict[str, tuple[LabelImageT, ShapeAndDTypeOrPyramid]] = {}

        # For immediate write workflow
        self._initialized = False
        self._written_labels: list[str] = []

        # Load existing labels from labels/zarr.json if it exists
        self._preexisting_labels: list[str] = self._load_existing_labels()

    @property
    def root_path(self) -> Path:
        """Path to the labels group."""
        return self._dest

    def write_label(
        self,
        name: str,
        label_image: LabelImageT,
        datasets: ArrayOrPyramid,
        *,
        progress: bool = False,
    ) -> Self:
        """Write a label immediately with its data.

        This method creates the label structure and writes data in one call.
        The labels group structure and LabelsGroup metadata are created/updated
        automatically. Use this for the "immediate write" workflow.

        Parameters
        ----------
        name : str
            Label name (becomes the subgroup path, e.g., "cells", "nuclei").
        label_image : LabelImage
            OME-Zarr LabelImage metadata model for this label.
        datasets : ArrayLike | Sequence[ArrayLike]
            Data array(s) for each resolution level. For a single dataset,
            pass the array directly without wrapping in a list.
        progress : bool, optional
            Show progress bar for dask arrays. Default is False.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If a label with this name was already written or added.
        NotImplementedError
            If the LabelImage has multiple multiscales.
        """
        self._validate_label_name(name)
        self._models.check(label_image, self._models.LabelImage, f"Label '{name}': ")

        # Initialize labels group structure if needed
        self._ensure_initialized()

        # Write the label using the existing write_image function
        # (LabelImage is a subclass of Image) BEFORE registering it in
        # labels/zarr.json: if this raises (bad dataset count, backend not
        # installed, FileExistsError, ...), the labels group must not end up
        # pointing at a name that was never actually written.
        write_image(
            self._dest / name,
            label_image,
            datasets,
            models=self._models,
            writer=self._writer,
            chunks=self._chunks,
            shards=self._shards,
            overwrite=self._overwrite,
            compression=self._compression,
            progress=progress,
        )

        # Only now update labels/zarr.json with this label.
        self._update_labels_group(name)

        return self

    def add_label(
        self,
        name: str,
        label_image: LabelImageT,
        datasets: ShapeAndDTypeOrPyramid,
    ) -> Self:
        """Add a label for the prepare-only workflow.

        Registers a label to be created when `prepare()` is called. Use this
        when you want to create the Zarr structure without writing data
        immediately. After calling `prepare()`, write data to the returned
        array handles.

        Parameters
        ----------
        name : str
            Label name (becomes the subgroup path, e.g., "cells", "nuclei").
        label_image : LabelImage
            OME-Zarr LabelImage metadata model for this label.
        datasets : ShapeAndDTypeOrPyramid
            Shape/dtype spec(s) for each resolution level:
            - Single level: `(shape, dtype)`
            - Multiple levels: `[(shape1, dtype1), (shape2, dtype2)]`

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If a label with this name was already added or written, or if the
            number of specs doesn't match the metadata.
        NotImplementedError
            If the LabelImage has multiple multiscales.
        """
        self._validate_label_name(name)
        self._models.check(label_image, self._models.LabelImage, f"Label '{name}': ")
        _, specs_seq = _validate_and_normalize_datasets(
            label_image, datasets, f"Label '{name}': "
        )
        self._labels[name] = (label_image, specs_seq)
        return self

    def prepare(self) -> tuple[Path, dict[str, Any]]:
        """Create the Zarr hierarchy and return array handles.

        Creates the complete labels group structure including LabelsGroup
        metadata, and empty arrays for all registered labels. Call this after
        registering all labels with `add_label()`.

        The returned arrays support numpy-style indexing for writing data:
        `arrays["label_name/dataset"][:] = data`.

        Returns
        -------
        tuple[Path, dict[str, Any]]
            A tuple of (root_path, arrays) where `arrays` maps composite keys
            like `"cells/0"` (label name / dataset path) to array objects. The
            array type depends on the configured writer (zarr.Array or
            tensorstore.TensorStore).

        Raises
        ------
        ValueError
            If no labels have been added with `add_label()`.
        FileExistsError
            If destination exists and `overwrite` is False.
        ImportError
            If no suitable writer backend is installed.
        """
        if not self._labels:  # pragma: no cover
            raise ValueError("No labels added. Use add_label() before prepare().")

        # Merge existing labels with new labels
        all_labels = list(self._preexisting_labels)
        all_labels.extend([new for new in self._labels if new not in all_labels])

        # Create or update labels/zarr.json with LabelsGroup metadata
        # If labels group already exists, just update the metadata file with new labels
        labels_group = self._models.LabelsGroup(labels=all_labels)
        if self._dest.exists() and (self._dest / "zarr.json").exists():
            _update_zarr3_group(self._dest, labels_group)
        else:
            # Create new group
            _create_zarr3_group(self._dest, labels_group, self._overwrite)

        # Create arrays for each label using prepare_image
        all_arrays: dict[str, Any] = {}
        for label_name, (label_image, datasets) in self._labels.items():
            _label_path, label_arrays = prepare_image(
                self._dest / label_name,
                label_image,
                datasets,
                models=self._models,
                chunks=self._chunks,
                shards=self._shards,
                writer=self._writer,
                overwrite=self._overwrite,
                compression=self._compression,
            )
            # Flatten into all_arrays with "label_name/dataset" keys
            for dataset_path, arr in label_arrays.items():
                all_arrays[f"{label_name}/{dataset_path}"] = arr

        return self._dest, all_arrays

    def __repr__(self) -> str:
        total_labels = len(self._labels) + len(self._written_labels)
        return f"<{self.__class__.__name__}: {total_labels} labels>"

    # ------------------ Internal methods ------------------

    def _load_existing_labels(self) -> list[str]:
        """Load existing labels from labels/zarr.json if it exists."""
        zarr_json_path = self._dest / "zarr.json"
        if not zarr_json_path.exists():
            return []

        try:
            with open(zarr_json_path) as f:
                data = json.load(f)
            labels = data.get("attributes", {}).get("ome", {}).get("labels", [])
            return labels if isinstance(labels, list) else []
        except (json.JSONDecodeError, KeyError):
            # If we can't parse it, assume no existing labels
            return []

    def _validate_label_name(self, name: str) -> None:
        if name in self._written_labels:  # pragma: no cover
            raise ValueError(f"Label '{name}' already written via write_label().")
        if name in self._labels:  # pragma: no cover
            raise ValueError(f"Label '{name}' already added via add_label().")

        # Check if label already exists in the labels group
        if name in self._preexisting_labels and not self._overwrite:
            raise ValueError(
                f"Label '{name}' already exists in the labels group. "
                f"Use overwrite=True to replace it."
            )

    def _ensure_initialized(self) -> None:
        """Create labels group directory if not already done.

        Note: labels/zarr.json is created/updated by _update_labels_group(),
        not here. This allows starting with zero labels.
        """
        if self._initialized:
            return

        # Create root directory
        self._dest.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def _update_labels_group(self, label_name: str) -> None:
        """Update labels/zarr.json with new label name.

        Similar to Bf2RawBuilder._update_ome_series(), this regenerates
        the LabelsGroup metadata from currently written labels and rewrites
        zarr.json. Merges with existing labels from the group.
        """
        if label_name in self._written_labels:
            # already added ... this is an internal method, don't need to raise
            return  # pragma: no cover

        self._written_labels.append(label_name)

        # Merge existing labels with newly written labels
        # If overwrite mode and label exists, it will be written to same path
        # Otherwise, we add new labels to the list
        all_labels = list(self._preexisting_labels)
        for new_label in self._written_labels:
            if new_label not in all_labels:
                all_labels.append(new_label)
            # If label exists and we're in overwrite mode, it's already in the list

        labels_group = self._models.LabelsGroup(labels=all_labels)
        zarr_json_path = self._dest / "zarr.json"
        # Preserve existing extra attributes if present
        existing_extra: dict[str, Any] = {}
        if zarr_json_path.exists():
            existing = json.loads(zarr_json_path.read_text())
            existing_extra = {
                k: v for k, v in existing.get("attributes", {}).items() if k != "ome"
            }
        zarr_json = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": labels_group.model_dump(mode="json", exclude_none=True),
                **existing_extra,
            },
        }
        zarr_json_path.write_text(json.dumps(zarr_json, indent=2))


# ##############################################################################
# ######################## Internal Helpers ####################################
# ##############################################################################


def _is_shape_and_dtype(obj: Any) -> TypeGuard[ShapeAndDType]:
    """Check if object is a (shape, dtype) tuple."""
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], tuple)  # shape is first element
    ):
        return True
    return False


def _validate_and_normalize_datasets(
    image: Any,
    datasets: ArrayOrPyramid | ShapeAndDTypeOrPyramid,
    context: str = "",
) -> tuple[Any, Sequence[ArrayLike]]:
    """Validate image has one multiscale and normalize datasets to a sequence.

    `datasets` can be either array-like data or (shape, dtype) specs.
    """
    if len(image.multiscales) != 1:
        raise NotImplementedError(f"{context}Image must have exactly one multiscale")

    multiscale = image.multiscales[0]
    if hasattr(datasets, "shape") and hasattr(datasets, "dtype"):
        datasets_seq = [datasets]
    elif _is_shape_and_dtype(datasets):
        datasets_seq = [datasets]
    else:
        datasets_seq = datasets

    if len(datasets_seq) != len(multiscale.datasets):
        raise ValueError(
            f"{context}Number of data arrays ({len(datasets_seq)}) must match "
            f"number of datasets in metadata ({len(multiscale.datasets)})"
        )

    return multiscale, datasets_seq


def _row_name_to_index(row_name: str) -> int:
    """Convert row name to index (A=0, B=1, ..., Z=25, AA=26, etc.)."""
    if not row_name or not row_name.isalpha() or not row_name.isupper():
        raise ValueError(  # pragma: no cover
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
    except ValueError:  # pragma: no cover
        raise ValueError(
            f"Column name must be numeric (1, 2, 10, etc.), got: {col_name}"
        ) from None


def _autogenerate_plate_metadata(
    fov_paths: Iterable[tuple[str, str, str]],
) -> dict[str, Any]:
    """Auto-generate plate metadata from images dict keys.

    Row indices follow the convention: A=0, B=1, ..., Z=25, AA=26, etc.
    Column indices follow: 1=0, 2=1, ..., 10=9, etc.

    All intermediate rows/columns are created (e.g., if you have rows A and D,
    rows B and C are also created, even if they have no wells).

    Parameters
    ----------
    fov_paths : Mapping[tuple[str, str, str], ImageWithAny]
        Images mapping with (row, col, fov) keys.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'rows', 'columns', 'wells' keys for PlateDef.
    """
    # Extract unique row and column names
    rows_set: set[str] = set()
    cols_set: set[int] = set()
    wells_set: set[tuple[str, str]] = set()
    for row, col, _fov in fov_paths:
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
    wells = [
        PlateWell(
            path=f"{row}/{col}",
            rowIndex=_row_name_to_index(row),
            columnIndex=_column_name_to_index(col),
        )
        for row, col in sorted(wells_set)
    ]

    return {"rows": rows, "columns": columns, "wells": wells}


def _merge_plate_metadata(
    images: Mapping[tuple[str, str, str], ImageWithAny],
    user_plate: Any | dict[str, Any] | None,
    models: OMEModels,
) -> Any:
    """Merge auto-generated and user-provided plate metadata.

    Parameters
    ----------
    images : Mapping[tuple[str, str, str], ImageWithAny]
        Images mapping used to auto-generate metadata.
    user_plate : Plate | dict[str, Any] | None
        User-provided plate metadata (takes precedence over auto-generated).
    models : OMEModels
        The version-specific model classes to build/check against.

    Returns
    -------
    Plate
        Final Plate object with merged metadata.

    Raises
    ------
    ValueError
        If user-provided metadata conflicts with images dict.
    """
    # Auto-generate base metadata from images
    auto_metadata = _autogenerate_plate_metadata(images)

    # If user provided a full Plate object, validate and return
    if isinstance(user_plate, BaseModel):
        models.check(user_plate, models.Plate, "plate: ")
        _validate_plate_matches_images(user_plate, images)
        return user_plate

    # If user provided a dict, merge with auto-generated
    merged = {**auto_metadata, **(user_plate or {})}

    # Construct Plate object from merged metadata
    plate = models.Plate(plate=models.PlateDef.model_validate(merged))

    # Validate that all images have valid coordinates
    _validate_plate_matches_images(plate, images)

    return plate


def _validate_plate_matches_images(
    plate: Any,
    images: Mapping[tuple[str, str, str], ImageWithAny],
) -> None:
    """Validate that plate metadata matches images dict.

    Parameters
    ----------
    plate : Plate
        Plate metadata to validate.
    images : Mapping[tuple[str, str, str], ImageWithAny]
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
        if row not in valid_rows:  # pragma: no cover
            raise ValueError(
                f"Image row '{row}' not found in plate rows: {sorted(valid_rows)}"
            )
        if col not in valid_cols:  # pragma: no cover
            raise ValueError(
                f"Image column '{col}' not found in plate columns: {sorted(valid_cols)}"
            )
        well_path = f"{row}/{col}"
        if well_path not in valid_wells:  # pragma: no cover
            raise ValueError(
                f"Image well '{well_path}' not found in plate wells: "
                f"{sorted(valid_wells)}"
            )


# ######################## Zarr Group Creation ##################################


def _create_zarr3_group(
    dest_path: Path,
    ome_model: BaseModel | None = None,
    overwrite: bool = False,
    indent: int = 2,
    extra_attributes: dict[str, Any] | None = None,
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
        if not zarr_json_path.exists():  # pragma: no cover
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
    if ome_model is not None or extra_attributes:
        attrs: dict[str, Any] = {}
        if ome_model is not None:
            attrs["ome"] = ome_model.model_dump(mode="json", exclude_none=True)
        if extra_attributes:
            attrs.update(extra_attributes)
        zarr_json["attributes"] = attrs
    zarr_json_path.write_text(json.dumps(zarr_json, indent=indent))


def _update_zarr3_group(
    dest_path: Path,
    ome_model: BaseModel,
    indent: int = 2,
    extra_attributes: dict[str, Any] | None = None,
) -> None:
    """Update the ome metadata in an existing zarr group."""
    zarr_json_path = dest_path / "zarr.json"
    if not zarr_json_path.exists():
        raise FileNotFoundError(f"Zarr group metadata not found at {zarr_json_path}")

    with open(zarr_json_path) as f:
        zarr_json = json.load(f)

    # Preserve existing extra attributes (non-ome keys)
    existing_attrs = zarr_json.get("attributes", {})
    attrs: dict[str, Any] = {k: v for k, v in existing_attrs.items() if k != "ome"}
    attrs["ome"] = ome_model.model_dump(mode="json", exclude_none=True)
    if extra_attributes:
        attrs.update(extra_attributes)
    zarr_json["attributes"] = attrs
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
        import numpy as np

        # Convert to np.dtype to ensure we have itemsize (handles types like np.uint16)
        dtype = np.dtype(dtype)
        return _calculate_auto_chunks(shape, dtype.itemsize)
    elif chunk_shape is None:
        return shape
    else:
        # Clamp to array shape
        return tuple(min(c, s) for c, s in zip(chunk_shape, shape))


def _resolve_shards(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shard_shape: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    """Clamp a requested shard shape against the array shape and chunk shape.

    Mirrors `_resolve_chunks`'s clamp-to-array-shape behavior for `shards`,
    which previously passed the user's `shards` through unclamped: for a
    multiscale pyramid, a fixed `shards=` tuple that fits the highest-res level
    can be larger than -- or not a multiple of -- the (already-shape-clamped)
    chunk size at a coarser level, producing an invalid Zarr v3 store (shard
    shape must be a multiple of chunk shape along every axis) partway through
    writing a pyramid, after the group metadata already claims that level
    exists.
    """
    if shard_shape is None:
        return None
    clamped = tuple(min(sh, s) for sh, s in zip(shard_shape, shape))
    # round each axis down to the nearest whole multiple of the (already
    # shape-clamped) chunk size, with a floor of one chunk per shard -- chunks
    # are already <= shape, so this can't exceed the array shape.
    return tuple(max(c, (sh // c) * c) if c else sh for sh, c in zip(clamped, chunks))


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
            # `da.store` writes block-by-block via the target's `__setitem__`,
            # which both `zarr.Array` and tensorstore's `TensorStore` support --
            # this keeps memory bounded to one dask chunk at a time. (Do NOT
            # `.compute()` the whole array first: that materializes the entire
            # array in memory before writing, defeating the point of passing a
            # dask array in the first place.)
            #
            # NB: use the default lock (`lock=True`, not `lock=False`). When
            # the dask array's chunk boundaries don't line up with the
            # storage's own chunk boundaries -- e.g. an "auto"-chunked target
            # array coarser than the dask chunks -- multiple worker threads
            # can end up writing into the *same* underlying storage chunk
            # concurrently. Without a lock those unsynchronized writes race
            # and silently corrupt data (verified: with `lock=False`, ~87% of
            # a small test array's elements ended up wrong; with the default
            # lock, zero).
            da.store(dask_data, array)  # ty: ignore[invalid-argument-type]

    else:
        if hasattr(array, "store"):  # zarr.Array
            array[:] = data
        else:  # tensorstore
            array[:].write(data).result()


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
                    f"zarr v3 or higher is required for OME-Zarr (v0.5+) writing, "
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
