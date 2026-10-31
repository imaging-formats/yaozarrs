"""Version-neutral write/prepare logic and builder base classes.

Each `yaozarrs.write.vNN` module subclasses the builders here, supplying its own
model classes (as class attributes) and its own public docstrings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeGuard, TypeVar, cast

from ._arrays import _get_create_func, _resolve_chunks, _write_to_array
from ._groups import _create_zarr3_group, _update_zarr3_group
from ._models import _check_model
from ._plate import _merge_plate_metadata

# The builders are generic over each version's Image/LabelImage models.  (Named like
# the models, so the rendered docs read the same as a version's own signatures.)
Image = TypeVar("Image")
LabelImage = TypeVar("LabelImage")

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from os import PathLike

    from pydantic import BaseModel
    from typing_extensions import Literal, Self, TypeAlias

    from ._arrays import (
        ArrayLike,
        ArrayOrPyramid,
        CompressionName,
        ShapeAndDType,
        ShapeAndDTypeOrPyramid,
        ShapeLike,
        ZarrWriter,
    )

    # (image, data) or (image, specs) pairs
    ImageWithDatasets: TypeAlias = tuple[Image, ArrayOrPyramid]
    ImageWithShapeSpecs: TypeAlias = tuple[Image, ShapeAndDTypeOrPyramid]
    ImageWithAny: TypeAlias = ImageWithDatasets[Any] | ImageWithShapeSpecs[Any]


def _is_shape_and_dtype(obj: Any) -> TypeGuard[ShapeAndDType]:
    """Check if object is a (shape, dtype) tuple."""
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], tuple)  # shape is first element
    ):
        return True
    return False


def _check_image_args(
    image: Any,
    datasets: ArrayOrPyramid | ShapeAndDTypeOrPyramid,
    image_type: type,
    context: str = "",
    chunks: Sequence[int] | Literal["auto"] | None = None,
    shards: Sequence[int] | None = None,
) -> tuple[Any, Sequence[ArrayLike]]:
    """Check an image and its datasets, before anything is written.

    Returns its multiscale, and `datasets` (array-like data or (shape, dtype) specs)
    as a sequence.  `chunks` and `shards`, if given, must have one entry per dimension
    of each dataset.
    """
    _check_model(image, image_type, context)
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

    for item, dataset in zip(cast("Sequence[Any]", datasets_seq), multiscale.datasets):
        shape = tuple(item.shape if hasattr(item, "shape") else item[0])
        for option, value in (("chunks", chunks), ("shards", shards)):
            if value not in (None, "auto") and len(value) != len(shape):
                raise ValueError(
                    f"{context}{option}={tuple(value)} must have one entry per "
                    f"dimension, but dataset '{dataset.path}' has {len(shape)} "
                    f"(shape {shape})."
                )

    return multiscale, datasets_seq


def _prepare_image(
    dest: Path,
    image: Any,
    datasets: ShapeAndDTypeOrPyramid,
    *,
    image_type: type,
    extra_attributes: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    compression: CompressionName = "blosc-zstd",
) -> tuple[Path, dict[str, Any]]:
    """Create the Image group and its (empty) arrays; see `vNN.prepare_image`."""
    multiscale, specs = _check_image_args(
        image, datasets, image_type, chunks=chunks, shards=shards
    )
    datasets_seq = cast("Sequence[ShapeAndDType]", specs)

    # Get create function
    create_func = _get_create_func(writer)

    # Create zarr group with Image metadata
    dest_path = Path(dest)
    _create_zarr3_group(dest_path, image, overwrite, extra_attributes=extra_attributes)

    # (every version's Multiscale has `axes`: in v0.6, those of the intrinsic
    # coordinate system, i.e. one per array dimension)
    dimension_names = [ax.name for ax in multiscale.axes]

    # Create arrays for each dataset
    arrays = {}
    for (shape, dtype_spec), dataset_meta in zip(datasets_seq, multiscale.datasets):
        # Convert dtype to np.dtype to ensure compatibility with all backends
        import numpy as np

        dtype = np.dtype(dtype_spec)
        arrays[dataset_meta.path] = create_func(
            path=dest_path / dataset_meta.path,
            shape=shape,
            dtype=dtype,
            chunks=_resolve_chunks(shape, dtype, chunks),
            shards=shards,
            dimension_names=dimension_names,
            overwrite=overwrite,
            compression=compression,
        )

    return dest_path, arrays


def _write_image(
    dest: Path,
    image: Any,
    datasets: ArrayOrPyramid,
    *,
    image_type: type,
    labels: Mapping[str, tuple[Any, ArrayOrPyramid]] | None = None,
    labels_builder_type: type[LabelsBuilderBase] | None = None,
    extra_attributes: dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write an Image group with data (and labels); see `vNN.write_image`."""
    multiscale, datasets_seq = _check_image_args(
        image, datasets, image_type, chunks=chunks, shards=shards
    )
    for label_name, (label_image, label_datasets) in (labels or {}).items():
        _check_image_args(
            label_image,
            label_datasets,
            image_type,
            f"Label '{label_name}': ",
            chunks,
            shards,
        )

    # Extract specs from arrays for prepare_image
    specs: list[ShapeAndDType] = [(arr.shape, arr.dtype) for arr in datasets_seq]

    # Create arrays using prepare_image
    dest_path, arrays = _prepare_image(
        dest,
        image,
        specs,
        image_type=image_type,
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
    if labels:
        if labels_builder_type is None:  # pragma: no cover
            raise TypeError("labels_builder_type is required to write labels")
        labels_builder = labels_builder_type(
            dest_path / "labels",
            writer=writer,
            chunks=chunks,
            shards=shards,
            overwrite=overwrite,
            compression=compression,
        )
        for label_name, (label_image, label_datasets) in labels.items():
            labels_builder.write_label(
                label_name, label_image, label_datasets, progress=progress
            )

    return dest_path


def _write_plate(
    builder_type: type[PlateBuilderBase],
    dest: str | PathLike,
    images: Mapping[tuple[str, str, str], ImageWithDatasets[Any]],
    *,
    plate: Any = None,
    extra_attributes: dict[str, Any] | None = None,
    writer: ZarrWriter = "auto",
    overwrite: bool = False,
    chunks: tuple[int, ...] | Literal["auto"] | None = "auto",
    shards: tuple[int, ...] | None = None,
    compression: CompressionName = "blosc-zstd",
    progress: bool = False,
) -> Path:
    """Write a Plate group with data; see `vNN.write_plate`."""
    # Merge user-provided plate metadata with auto-generated
    plate_obj = _merge_plate_metadata(images, plate, builder_type._plate_type)

    # Use PlateBuilder to handle the writing
    builder = builder_type(
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
    wells_data: dict[tuple[str, str], dict[str, ImageWithDatasets[Any]]] = {}
    for (row, col, fov), image_data in images.items():
        wells_data.setdefault((row, col), {})[fov] = image_data

    # (check every field first, so a bad one doesn't leave a partial plate behind)
    for (row, col), fields_data in wells_data.items():
        builder._check_fields(row, col, fields_data)

    # Write each well with all its fields
    for (row, col), fields_data in wells_data.items():
        builder.write_well(row=row, col=col, images=fields_data, progress=progress)

    return builder.root_path


def _write_bioformats2raw(
    builder_type: type[Bf2RawBuilderBase],
    dest: str | PathLike,
    images: Mapping[str, ImageWithDatasets[Any]],
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
    """Write a bioformats2raw group with data; see `vNN.write_bioformats2raw`."""
    # (check every series first, so a bad one doesn't leave a partial store behind)
    for series_name, (image_model, datasets) in images.items():
        _check_image_args(
            image_model,
            datasets,
            builder_type._image_type,
            f"Series '{series_name}': ",
            chunks,
            shards,
        )

    builder = builder_type(
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


class Bf2RawBuilderBase(Generic[Image]):
    """Builder for bioformats2raw layout hierarchies (see `vNN.Bf2RawBuilder`)."""

    _image_type: ClassVar[type[Any]]
    _bf2raw_type: ClassVar[type[Any]]
    _series_type: ClassVar[type[Any]]

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
        image: Image,
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
        _check_image_args(
            image,
            datasets,
            self._image_type,
            f"Series '{name}': ",
            self._chunks,
            self._shards,
        )

        # Initialize root structure if needed
        self._ensure_initialized()

        # Update OME/zarr.json with this series
        self._update_ome_series(name)

        # Write the series using the existing write_image function
        _write_image(
            self._dest / name,
            image,
            datasets,
            image_type=self._image_type,
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
        _, datasets_seq = _check_image_args(
            image,
            datasets,
            self._image_type,
            f"Series '{name}': ",
            self._chunks,
            self._shards,
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
        bf2raw = self._bf2raw_type(bioformats2raw_layout=3)
        _create_zarr3_group(
            self._dest,
            bf2raw,
            self._overwrite,
            extra_attributes=self._extra_attributes,
        )

        # Create OME/zarr.json with series list
        ome_path = self._dest / "OME"
        series_model = self._series_type(series=list(self._series))
        _create_zarr3_group(ome_path, series_model, self._overwrite)

        # Write METADATA.ome.xml if provided
        if self._ome_xml is not None:
            (ome_path / "METADATA.ome.xml").write_text(self._ome_xml)

        # Create arrays for each series using prepare_image
        all_arrays: dict[str, Any] = {}
        for series_name, (image_model, dataset_specs) in self._series.items():
            _root_path, series_arrays = _prepare_image(
                self._dest / series_name,
                image_model,
                dataset_specs,
                image_type=self._image_type,
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
        bf2raw = self._bf2raw_type(bioformats2raw_layout=3)
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
        series_model = self._series_type(series=self._written_series)
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


class PlateBuilderBase(Generic[Image]):
    """Builder for Plate hierarchies (see `vNN.PlateBuilder`)."""

    _image_type: ClassVar[type[Any]]
    _plate_type: ClassVar[type[Any]]
    _well_type: ClassVar[type[Any]]
    _well_def_type: ClassVar[type[Any]]
    _field_of_view_type: ClassVar[type[Any]]

    def __init__(
        self,
        dest: str | PathLike,
        *,
        plate: Any | None = None,
        extra_attributes: dict[str, Any] | None = None,
        writer: ZarrWriter = "auto",
        chunks: ShapeLike | Literal["auto"] | None = "auto",
        shards: ShapeLike | None = None,
        overwrite: bool = False,
        compression: CompressionName = "blosc-zstd",
    ) -> None:
        self._dest = Path(dest)
        if plate is not None:
            _check_model(plate, self._plate_type, "plate: ")
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
        images: Mapping[str, ImageWithDatasets[Image]],
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

        # Check and normalize fields (convert single arrays to sequences)
        normalized_fields = self._check_fields(row, col, images)

        # Initialize plate structure if needed (after the checks, so nothing is
        # created for a well that would be rejected)
        self._ensure_initialized()

        # Track this well's data before writing
        self._written_wells_data[(row, col)] = cast(
            "dict[str, ImageWithDatasets]", normalized_fields
        )

        # Update plate metadata with the new well
        self._update_plate_metadata()

        # Create the well subgroup (and the row group above it)
        well_group_path = self._create_well_group(f"{row}/{col}", list(images))

        # Write each field of view
        for fov, (image_model, datasets_seq) in normalized_fields.items():
            field_path = well_group_path / fov
            _write_image(
                field_path,
                image_model,
                datasets_seq,
                image_type=self._image_type,
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
        images: Mapping[str, ImageWithShapeSpecs[Image]],
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

        # Check and normalize all fields before accepting
        well_path = f"{row}/{col}"
        normalized_fields = cast(
            "dict[str, ImageWithShapeSpecs[Image]]",
            self._check_fields(row, col, images),
        )

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
            self._get_images_dict(), self._user_plate, self._plate_type
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
            well_group_path = self._create_well_group(well_path, list(fields))

            # Create arrays for each field
            for fov, (image_model, datasets) in fields.items():
                field_path = well_group_path / fov

                _field_path, field_arrays = _prepare_image(
                    field_path,
                    image_model,
                    datasets,
                    image_type=self._image_type,
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

    def _check_fields(
        self, row: str, col: str, images: Mapping[str, ImageWithAny]
    ) -> dict[str, tuple[Image, Sequence[Any]]]:
        """Check a well's fields before anything is written; normalize datasets."""
        normalized: dict[str, tuple[Image, Sequence[Any]]] = {}
        for fov, (image_model, datasets) in images.items():
            _, datasets_seq = _check_image_args(
                image_model,
                datasets,
                self._image_type,
                f"Well '{row}/{col}', field '{fov}': ",
                self._chunks,
                self._shards,
            )
            normalized[fov] = (image_model, datasets_seq)
        # (the well metadata validates the field names)
        self._generate_well_metadata(list(images))
        return normalized

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

    def _generate_current_plate_metadata(self) -> Any:
        """Generate plate metadata from currently written wells.

        If user provided a Plate, use that. Otherwise, auto-generate from
        written wells (similar to write_plate auto-generation).
        """
        if self._user_plate is not None:
            return self._user_plate
        return _merge_plate_metadata(
            self._get_images_dict(), self._user_plate, self._plate_type
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

    def _create_well_group(self, well_path: str, field_names: list[str]) -> Path:
        """Create the group for well ``well_path`` ("A/1") and the row group above it.

        The row group ("A/") is created first if it is not already there; Zarr v3 has
        no implicit groups, so it needs a ``zarr.json`` of its own for a reader to
        resolve plate -> row -> well. Returns the path of the well group.
        """
        row_path = self._dest / well_path.split("/")[0]
        row_path.mkdir(parents=True, exist_ok=True)
        row_json = row_path / "zarr.json"
        if not row_json.exists():
            # Not _create_zarr3_group: that treats an existing path as something to
            # replace, and the row directory already exists once any sibling well
            # does. A row has no metadata to overwrite, so writing it is idempotent.
            row_json.write_text(json.dumps({"zarr_format": 3, "node_type": "group"}))

        well_group_path = self._dest / well_path
        well_metadata = self._generate_well_metadata(field_names)
        _create_zarr3_group(well_group_path, well_metadata, self._overwrite)
        return well_group_path

    def _generate_well_metadata(self, field_names: list[str]) -> BaseModel:
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
            self._field_of_view_type(path=fov, acquisition=None)
            for fov in sorted(field_names)
        ]

        return self._well_type(well=self._well_def_type(images=images))


class LabelsBuilderBase(Generic[LabelImage]):
    """Builder for labels groups within an Image (see `vNN.LabelsBuilder`)."""

    _image_type: ClassVar[type[Any]]
    _labels_group_type: ClassVar[type[Any]]

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
        self._labels: dict[str, tuple[LabelImage, ShapeAndDTypeOrPyramid]] = {}

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
        label_image: LabelImage,
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
        _check_image_args(
            label_image,
            datasets,
            self._image_type,
            f"Label '{name}': ",
            self._chunks,
            self._shards,
        )

        # Initialize labels group structure if needed
        self._ensure_initialized()

        # Update labels/zarr.json with this label
        self._update_labels_group(name)

        # Write the label using the existing write_image function
        # (LabelImage is a subclass of Image)
        _write_image(
            self._dest / name,
            label_image,
            datasets,
            image_type=self._image_type,
            writer=self._writer,
            chunks=self._chunks,
            shards=self._shards,
            overwrite=self._overwrite,
            compression=self._compression,
            progress=progress,
        )

        return self

    def add_label(
        self,
        name: str,
        label_image: LabelImage,
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
        _, specs_seq = _check_image_args(
            label_image,
            datasets,
            self._image_type,
            f"Label '{name}': ",
            self._chunks,
            self._shards,
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
        labels_group = self._labels_group_type(labels=all_labels)
        if self._dest.exists() and (self._dest / "zarr.json").exists():
            _update_zarr3_group(self._dest, labels_group)
        else:
            # Create new group
            _create_zarr3_group(self._dest, labels_group, self._overwrite)

        # Create arrays for each label using prepare_image
        all_arrays: dict[str, Any] = {}
        for label_name, (label_image, datasets) in self._labels.items():
            _label_path, label_arrays = _prepare_image(
                self._dest / label_name,
                label_image,
                datasets,
                image_type=self._image_type,
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

        labels_group = self._labels_group_type(labels=all_labels)
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
