"""Version-neutral write/prepare logic and builder base classes.

Each `yaozarrs.write.vNN` module subclasses the builders here, supplying its own
model classes (as class attributes) and its own public docstrings.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeGuard, TypeVar, cast

from ._arrays import get_create_func, get_open_func, resolve_chunks, write_to_array
from ._groups import (
    atomic_write,
    create_zarr3_group,
    ensure_parent_groups,
    is_empty_dir,
    rewrite_zarr3_group,
    update_zarr3_group,
    write_group_json,
)
from ._models import check_model
from ._plate import (
    autogenerate_plate_metadata,
    extend_plate_metadata,
    merge_plate_metadata,
)

# The builders are generic over each version's Image/LabelImage models.  (Named like
# the models, so the rendered docs read the same as a version's own signatures.)
Image = TypeVar("Image")
LabelImage = TypeVar("LabelImage")

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from os import PathLike
    from typing import Literal, Self, TypeAlias

    from pydantic import BaseModel

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


class Default(Enum):
    """Sentinel for a per-image setting that defaults to the builder's."""

    BUILDER = "builder"


# Marks a bioformats2raw collection being created, until its root zarr.json is written
# along with its first series (so that an interrupted first write can be resumed, or
# overwritten, unlike an unrelated directory).
_CREATING_MARKER = ".yaozarrs-creating"

_ADDED_VIA = {
    "write_image": "written via write_image()",
    "add_series": "added via add_series()",
    "prepare_series": "created via prepare_series()",
}

# One lock per store (by its real path), shared by all builders of this process
_STORE_LOCKS: dict[str, threading.RLock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


def _store_lock(dest: Path) -> threading.RLock:
    key = os.path.normcase(os.path.realpath(dest))
    with _STORE_LOCKS_GUARD:
        return _STORE_LOCKS.setdefault(key, threading.RLock())


def _discard(path: Path) -> None:
    """Delete `path`, what was created of a series that failed."""
    shutil.rmtree(path, ignore_errors=True)


def _numbered_series(root: Path) -> list[str]:
    """The groups numbered `0, 1, ...` (up to the first gap) below `root`."""
    names: list[str] = []
    while (root / str(len(names)) / "zarr.json").exists():
        names.append(str(len(names)))
    return names


def _claim_dir(path: Path) -> None:
    """Create the directory `path` (and its parents); fail if it already exists.

    Unlike checking for it and then creating it, only one of several concurrent
    callers can succeed, so success means the directory is the caller's own.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(path)


def _remove_tree(path: Path) -> None:
    """Delete the directory `path`, making it disappear at once.

    It's renamed aside first (an atomic operation), then deleted.  If it can't be
    renamed, its `zarr.json` is deleted first, so it stops being a Zarr group before
    anything else of it is deleted.
    """
    trash = path.with_name(f".{path.name}.{uuid.uuid4().hex}.trash")
    try:
        os.rename(path, trash)
    except OSError:
        (path / "zarr.json").unlink(missing_ok=True)
        shutil.rmtree(path)
    else:
        # (a leftover trash directory is no longer part of anything)
        shutil.rmtree(trash, ignore_errors=True)


class _SeriesNames:
    """Casefolded series names and their parent groups, to find clashing names."""

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._names: dict[str, str] = {}  # (casefolded name -> name)
        self._parents: dict[str, str] = {}  # (casefolded group -> a name below it)
        for name in names:
            self.add(name)

    def add(self, name: str) -> None:
        folded = name.casefold()
        self._names[folded] = name
        for parent in _parent_paths(folded):
            self._parents.setdefault(parent, name)

    def clash(self, name: str) -> tuple[str, bool] | None:
        """The name `name` clashes with (if any), and whether it's the same path."""
        folded = name.casefold()
        if (other := self._names.get(folded)) is not None:
            return other, True
        if (other := self._parents.get(folded)) is not None:
            return other, False
        for parent in _parent_paths(folded):
            if (other := self._names.get(parent)) is not None:
                return other, False
        return None


def _parent_paths(path: str) -> list[str]:
    """E.g. `["a", "a/b"]` for `"a/b/c"`."""
    parts = path.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


def check_image_path(path: str) -> None:
    # (each part is the name of a node below the parent group: Zarr node names must be
    # non-empty, not only periods, and not start with "__"; and "zarr.json" would
    # collide with a group's own metadata)
    if any(
        not p or set(p) == {"."} or p.startswith("__") or p == "zarr.json"
        for p in path.split("/")
    ):
        raise ValueError(
            f"Invalid image path {path!r}: each part must be a valid Zarr node name "
            "(non-empty, not only periods, not starting with '__', not 'zarr.json')."
        )


def _is_shape_and_dtype(obj: Any) -> TypeGuard[ShapeAndDType]:
    """Check if object is a (shape, dtype) tuple."""
    return isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], tuple)


def check_image_args(
    image: Any,
    datasets: ArrayOrPyramid | ShapeAndDTypeOrPyramid,
    image_type: type,
    context: str = "",
    chunks: Sequence[int] | Literal["auto"] | None = None,
    shards: Sequence[int] | None = None,
) -> tuple[Any, Sequence[ArrayLike]]:
    """Check an image and its datasets, before anything is written.

    Returns its multiscale, and `datasets` (array-like data or (shape, dtype) specs)
    as a sequence.  Each dataset must have one dimension per axis, and `chunks` and
    `shards`, if given, one entry per dimension.
    """
    check_model(image, image_type, context)
    if len(image.multiscales) != 1:
        raise NotImplementedError(f"{context}Image must have exactly one multiscale")

    multiscale = image.multiscales[0]
    is_array = hasattr(datasets, "shape") and hasattr(datasets, "dtype")
    if is_array or _is_shape_and_dtype(datasets):
        datasets_seq = [datasets]
    else:
        datasets_seq = datasets

    if len(datasets_seq) != len(multiscale.datasets):
        raise ValueError(
            f"{context}Number of data arrays ({len(datasets_seq)}) must match "
            f"number of datasets in metadata ({len(multiscale.datasets)})"
        )

    ndim = len(multiscale.axes)
    for item, dataset in zip(
        cast("Sequence[Any]", datasets_seq), multiscale.datasets, strict=False
    ):
        shape = tuple(item.shape if hasattr(item, "shape") else item[0])
        if len(shape) != ndim:
            raise ValueError(
                f"{context}dataset '{dataset.path}' has {len(shape)} dimensions "
                f"{shape}, but the image has {ndim} axes."
            )
        for option, value in (("chunks", chunks), ("shards", shards)):
            if value not in (None, "auto") and len(value) != len(shape):
                raise ValueError(
                    f"{context}{option}={tuple(value)} must have one entry per "
                    f"dimension, but dataset '{dataset.path}' has {len(shape)} "
                    f"(shape {shape})."
                )

    return multiscale, datasets_seq


def check_label_dtypes(datasets_seq: Sequence[Any], context: str) -> None:
    # The pixels of the label images MUST be integer data types
    import numpy as np

    for item in datasets_seq:
        dtype = item.dtype if hasattr(item, "dtype") else item[1]
        if np.dtype(dtype).kind not in "iu":
            raise ValueError(
                f"{context}label images MUST have an integer data type, "
                f"got {np.dtype(dtype)}."
            )


def prepare_image(
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
    multiscale, specs = check_image_args(
        image, datasets, image_type, chunks=chunks, shards=shards
    )
    datasets_seq = cast("Sequence[ShapeAndDType]", specs)
    create_func = get_create_func(writer)
    dest_path = Path(dest)
    create_zarr3_group(dest_path, image, overwrite, extra_attributes=extra_attributes)

    # (every version's Multiscale has `axes`: in v0.6, those of the intrinsic
    # coordinate system, i.e. one per array dimension)
    dimension_names = [ax.name for ax in multiscale.axes]

    import numpy as np

    arrays = {}
    for (shape, dtype_spec), dataset_meta in zip(
        datasets_seq, multiscale.datasets, strict=False
    ):
        dtype = np.dtype(dtype_spec)  # (an np.dtype, which every backend accepts)
        arrays[dataset_meta.path] = create_func(
            path=dest_path / dataset_meta.path,
            shape=shape,
            dtype=dtype,
            chunks=resolve_chunks(shape, dtype, chunks),
            shards=shards,
            dimension_names=dimension_names,
            overwrite=overwrite,
            compression=compression,
        )

    return dest_path, arrays


def write_image(
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
    multiscale, datasets_seq = check_image_args(
        image, datasets, image_type, chunks=chunks, shards=shards
    )
    for label_name, (label_image, label_datasets) in (labels or {}).items():
        context = f"Label '{label_name}': "
        _, label_seq = check_image_args(
            label_image, label_datasets, image_type, context, chunks, shards
        )
        check_label_dtypes(label_seq, context)

    specs: list[ShapeAndDType] = [(arr.shape, arr.dtype) for arr in datasets_seq]
    dest_path, arrays = prepare_image(
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

    for data_array, dataset_meta in zip(
        datasets_seq, multiscale.datasets, strict=False
    ):
        write_to_array(arrays[dataset_meta.path], data_array, progress=progress)

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


def write_plate(
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
    plate_obj = merge_plate_metadata(images, plate, builder_type._plate_type)
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

    for (row, col), fields_data in wells_data.items():
        builder.write_well(row=row, col=col, images=fields_data, progress=progress)

    return builder.root_path


def write_bioformats2raw(
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
        check_image_args(
            image_model,
            datasets,
            builder_type._image_type,
            f"Series '{series_name}': ",
            chunks,
            shards,
        )
    # (unlike a builder's, a one-shot write doesn't add to an existing collection)
    dest_path = Path(dest)
    if not overwrite and dest_path.exists() and not is_empty_dir(dest_path):
        raise FileExistsError(
            f"Zarr group already exists at {dest_path}. "
            "Use overwrite=True to replace it."
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


class BuilderBase:
    """The array settings of a builder, and writing/preparing its images with them."""

    _image_type: ClassVar[type[Any]]

    def __init__(
        self,
        dest: str | PathLike,
        *,
        writer: ZarrWriter,
        chunks: ShapeLike | Literal["auto"] | None,
        shards: ShapeLike | None,
        overwrite: bool,
        compression: CompressionName,
    ) -> None:
        self._dest = Path(dest)
        self._writer: ZarrWriter = writer
        self._chunks: ShapeLike | Literal["auto"] | None = chunks
        self._shards = shards
        self._overwrite = overwrite
        self._compression: CompressionName = compression
        self._initialized = False

    def _array_options(
        self,
        chunks: ShapeLike | Literal["auto"] | Default | None = Default.BUILDER,
        shards: ShapeLike | Default | None = Default.BUILDER,
        compression: CompressionName | Default = Default.BUILDER,
    ) -> dict[str, Any]:
        """Resolve per-image array settings, falling back to the builder's."""
        return {
            "chunks": self._chunks if chunks is Default.BUILDER else chunks,
            "shards": self._shards if shards is Default.BUILDER else shards,
            "compression": (
                self._compression if compression is Default.BUILDER else compression
            ),
        }

    def _check_datasets(self, image: Any, datasets: Any, context: str) -> Sequence[Any]:
        """`check_image_args`, with the builder's settings; return the datasets."""
        _, datasets_seq = check_image_args(
            image, datasets, self._image_type, context, self._chunks, self._shards
        )
        return datasets_seq

    def _write(
        self,
        image: Any,
        datasets: Any,
        *path: str,
        progress: bool,
        labels: Mapping[str, tuple[Any, ArrayOrPyramid]] | None = None,
        labels_builder_type: type[LabelsBuilderBase] | None = None,
        chunks: ShapeLike | Literal["auto"] | Default | None = Default.BUILDER,
        shards: ShapeLike | Default | None = Default.BUILDER,
        compression: CompressionName | Default = Default.BUILDER,
    ) -> None:
        """Write `image` and its data (and `labels`) at `path`, below the root."""
        write_image(
            self._dest.joinpath(*path),
            image,
            datasets,
            image_type=self._image_type,
            labels=labels,
            labels_builder_type=labels_builder_type,
            writer=self._writer,
            overwrite=self._overwrite,
            progress=progress,
            **self._array_options(chunks, shards, compression),
        )

    def _prepare(
        self,
        image: Any,
        specs: Any,
        *path: str,
        chunks: ShapeLike | Literal["auto"] | Default | None = Default.BUILDER,
        shards: ShapeLike | Default | None = Default.BUILDER,
        compression: CompressionName | Default = Default.BUILDER,
    ) -> dict[str, Any]:
        """Create `image` (no data) at `path`; return its arrays, by path below root."""
        _, arrays = prepare_image(
            self._dest.joinpath(*path),
            image,
            specs,
            image_type=self._image_type,
            writer=self._writer,
            overwrite=self._overwrite,
            **self._array_options(chunks, shards, compression),
        )
        prefix = "/".join(path)
        return {f"{prefix}/{dataset_path}": arr for dataset_path, arr in arrays.items()}

    def _ensure_initialized(self) -> None:
        """Create the root directory, if not already done.

        (Its zarr.json, which lists the wells or labels, is rewritten as each one is
        written.)
        """
        if self._initialized:
            return
        self._dest.mkdir(parents=True, exist_ok=True)
        self._initialized = True


class Bf2RawBuilderBase(BuilderBase, Generic[Image]):
    """Builder for bioformats2raw layout hierarchies (see `vNN.Bf2RawBuilder`).

    Every series is created (its group and arrays, and data if given) before it is
    listed in `OME/zarr.json`, and the root `zarr.json` is written with the first
    series, last.  So the store is a valid collection after every call, even an
    interrupted one.
    """

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
        super().__init__(
            dest,
            writer=writer,
            chunks=chunks,
            shards=shards,
            overwrite=overwrite,
            compression=compression,
        )
        # (written along with the next series created)
        self._ome_xml = ome_xml
        self._extra_attributes = extra_attributes
        self._extra_pending = extra_attributes is not None
        # (with `overwrite`, the store is replaced when the first series is created)
        self._wipe_pending = overwrite
        self._lock = _store_lock(self._dest)
        # (set if the existing collection has numbered series, but no `series` list)
        self._numbered = False
        # The collection's series, as last read from (or written to) OME/zarr.json
        self._series: list[str] = [] if overwrite else self._load_existing()
        # The series added with this builder: {name: the method that added it}
        self._added: dict[str, str] = {}
        # (the names of both, to check new ones against)
        self._names = _SeriesNames(self._series)
        # For prepare-only workflow: {name: (image, dataset specs, array options)}
        self._queued: dict[str, tuple[Any, Sequence[Any], dict[str, Any]]] = {}

    @property
    def root_path(self) -> Path:
        """Path to the root of the bioformats2raw hierarchy."""
        return self._dest

    @property
    def series(self) -> tuple[str, ...]:
        """The series of the collection, in `OME/series` order.

        Those already in the store when the builder was created, and those created
        since (but not those only added with `add_series()`, until `prepare()`).
        """
        return tuple(self._series)

    def prepare_series(
        self,
        name: str,
        image: Image,
        datasets: ShapeAndDTypeOrPyramid,
        *,
        chunks: ShapeLike | Literal["auto"] | Default | None = Default.BUILDER,
        shards: ShapeLike | Default | None = Default.BUILDER,
        compression: CompressionName | Default = Default.BUILDER,
    ) -> tuple[Path, dict[str, Any]]:
        """Create a series now, with empty arrays, and add it to the collection.

        The series' group and arrays are created first, and only then is the series
        listed in `OME/zarr.json` (which creates the collection, if this is its first
        series), so the collection stays valid even if this is interrupted.

        Write data into the returned arrays (or into the arrays from `arrays()`, or
        opened with your own backend) at any time, in any order.

        Parameters
        ----------
        name : str
            Path of the series' image group below the root, e.g. "0", or
            "group_1/array_1_0" (intermediate groups are created).
        image : Image
            OME-Zarr Image metadata model for this series.
        datasets : ShapeAndDType | Sequence[ShapeAndDType]
            Shape and dtype specification(s) for each resolution level, as
            `(shape, dtype)` tuples. For a single dataset, pass the tuple
            directly without wrapping in a list.
        chunks : tuple[int, ...] | "auto" | None, optional
            Chunk shape for this series' arrays.  Default is the builder's.
        shards : tuple[int, ...] | None, optional
            Shard shape for this series' arrays.  Default is the builder's.
        compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
            Compression codec for this series' arrays.  Default is the builder's.

        Returns
        -------
        tuple[Path, dict[str, Any]]
            The path of the series' group, and its arrays by dataset path (e.g.
            "0").  The array type depends on the configured writer (zarr.Array or
            tensorstore.TensorStore).

        Raises
        ------
        ValueError
            If `name` is not a valid path, is (or is inside) "OME", is nested in (or
            contains) another series, or differs only in case from one; if it was
            already added with this builder; or if the dataset specs don't match the
            metadata.
        FileExistsError
            If the collection already has a series `name`, or something that isn't
            one of its series is at its path (e.g. left by an interrupted write).
        ImportError
            If no suitable writer backend is installed (before anything is written).
        """
        specs, options = self._check_series(
            name, image, datasets, chunks, shards, compression
        )
        self._reserve(name, "prepare_series")
        arrays = self._create({name: (image, specs, options)})
        return self._dest / name, arrays[name]

    def write_image(
        self,
        name: str,
        image: Image,
        datasets: ArrayOrPyramid,
        *,
        chunks: ShapeLike | Literal["auto"] | Default | None = Default.BUILDER,
        shards: ShapeLike | Default | None = Default.BUILDER,
        compression: CompressionName | Default = Default.BUILDER,
        progress: bool = False,
    ) -> Self:
        """Write a series immediately with its data.

        Like `prepare_series()`, but the data is written too, before the series is
        listed in `OME/zarr.json`: so the series only appears in the collection
        with all of its data.

        Parameters
        ----------
        name : str
            Path of the series' image group below the root, e.g. "0", or
            "group_1/array_1_0" (intermediate groups are created).
        image : Image
            OME-Zarr Image metadata model for this series.
        datasets : ArrayLike | Sequence[ArrayLike]
            Data array(s) for each resolution level. For a single dataset,
            pass the array directly without wrapping in a list.
        chunks : tuple[int, ...] | "auto" | None, optional
            Chunk shape for this series' arrays.  Default is the builder's.
        shards : tuple[int, ...] | None, optional
            Shard shape for this series' arrays.  Default is the builder's.
        compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
            Compression codec for this series' arrays.  Default is the builder's.
        progress : bool, optional
            Show progress bar when writing dask arrays. Default is False.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If `name` is invalid or was already added with this builder, or if the
            datasets don't match the metadata (see `prepare_series()`).
        FileExistsError
            If the collection already has a series `name`, or something else is at
            its path.
        NotImplementedError
            If the Image has multiple multiscales.
        """
        datasets_seq, options = self._check_series(
            name, image, datasets, chunks, shards, compression
        )
        specs = [(arr.shape, arr.dtype) for arr in datasets_seq]
        self._reserve(name, "write_image")
        series = {name: (image, specs, options)}
        self._create(series, data={name: datasets_seq}, progress=progress)
        return self

    def add_series(
        self,
        name: str,
        image: Image,
        datasets: ShapeAndDTypeOrPyramid,
        *,
        chunks: ShapeLike | Literal["auto"] | Default | None = Default.BUILDER,
        shards: ShapeLike | Default | None = Default.BUILDER,
        compression: CompressionName | Default = Default.BUILDER,
    ) -> Self:
        """Add a series for the prepare-only workflow.

        Registers a series to be created when `prepare()` is next called.  (To create
        a series right away, use `prepare_series()`.)  After calling `prepare()`,
        write data to the returned array handles.

        Parameters
        ----------
        name : str
            Path of the series' image group below the root, e.g. "0", or
            "group_1/array_1_0" (intermediate groups are created).
        image : Image
            OME-Zarr Image metadata model for this series.
        datasets : ShapeAndDType | Sequence[ShapeAndDType]
            Shape and dtype specification(s) for each resolution level, as
            `(shape, dtype)` tuples. For a single dataset, pass the tuple
            directly without wrapping in a list.
        chunks : tuple[int, ...] | "auto" | None, optional
            Chunk shape for this series' arrays.  Default is the builder's.
        shards : tuple[int, ...] | None, optional
            Shard shape for this series' arrays.  Default is the builder's.
        compression : "blosc-zstd" | "blosc-lz4" | "zstd" | "none", optional
            Compression codec for this series' arrays.  Default is the builder's.

        Returns
        -------
        Self
            The builder instance (for method chaining).

        Raises
        ------
        ValueError
            If `name` is invalid or was already added with this builder, or if the
            number of dataset specs doesn't match the metadata.
        FileExistsError
            If the collection already has a series `name`, or something else is at
            its path.
        NotImplementedError
            If the Image has multiple multiscales.
        """
        specs, options = self._check_series(
            name, image, datasets, chunks, shards, compression
        )
        self._reserve(name, "add_series")
        with self._lock:
            self._queued[name] = (image, specs, options)
        return self

    def prepare(self) -> tuple[Path, dict[str, Any]]:
        """Create the series added with `add_series()`, and return their arrays.

        Creates the series added since the last call (with empty arrays), then lists
        them all in `OME/zarr.json`, creating the collection if needed.  It can be
        called again after adding more series.  If creating any of them fails,
        nothing is added to the collection, and they stay added for the next call
        (except one whose path is taken, which is dropped).

        The returned arrays support numpy-style indexing for writing data:
        `arrays["series/dataset"][:] = data`.

        Returns
        -------
        tuple[Path, dict[str, Any]]
            A tuple of (root_path, arrays) where `arrays` maps composite keys
            like `"0/0"` (series name / dataset path) to the arrays of the series
            created by this call. The array type depends on the configured writer
            (zarr.Array or tensorstore.TensorStore).

        Raises
        ------
        ValueError
            If no series have been added with `add_series()` since the last call.
        FileExistsError
            If something is already at the path of one of the series (which is then
            dropped: delete what's there and add it again, or use another name).
        ImportError
            If no suitable writer backend is installed.
        """
        with self._lock:
            queued, self._queued = self._queued, {}
        if not queued:
            raise ValueError("No series added. Use add_series() before prepare().")
        try:
            arrays = self._create(queued, release=False)
        except BaseException:
            with self._lock:
                # (not those listed after all, nor those dropped)
                retry = {
                    name: series
                    for name, series in queued.items()
                    if name in self._added and name not in self._series
                }
                self._queued = {**retry, **self._queued}
            raise
        all_arrays = {
            f"{name}/{dataset_path}": arr
            for name, series_arrays in arrays.items()
            for dataset_path, arr in series_arrays.items()
        }
        return self._dest, all_arrays

    def arrays(self, name: str) -> dict[str, Any]:
        """Open the arrays of series `name` for writing, by dataset path.

        This works for any series of the collection, including those created by an
        earlier builder.

        Raises
        ------
        KeyError
            If the collection has no series `name`.
        TypeError
            If the builder has a custom `writer`, which can't reopen arrays: open them
            with your own backend instead (at `root_path / name / dataset_path`).
        """
        if name not in self._series:
            with self._lock:
                if not self._wipe_pending:  # (else what's there is about to go)
                    self._set_series(self._read_listing() or self._series)
            if name not in self._series:
                raise KeyError(f"The collection has no series {name!r}.")
        if (open_func := get_open_func(get_create_func(self._writer))) is None:
            raise TypeError(
                "Arrays created by a custom writer can't be reopened by yaozarrs: open "
                f"them with your own backend, at {self._dest / name}/<dataset path>."
            )
        doc = json.loads((self._dest / name / "zarr.json").read_bytes())
        multiscales = doc["attributes"]["ome"]["multiscales"]
        paths = dict.fromkeys(ds["path"] for ms in multiscales for ds in ms["datasets"])
        return {path: open_func(self._dest / name / path) for path in paths}

    def write_ome_xml(self, ome_xml: str) -> None:
        """(Re)write `OME/METADATA.ome.xml`, atomically.

        Its `Image` elements must be in the order of the collection's series.  If the
        collection doesn't exist yet, it is written along with the first series.
        """
        with self._lock:
            self._ome_xml = ome_xml
            if not self._wipe_pending and (self._dest / "zarr.json").exists():
                self._write_ome_xml()

    def __repr__(self) -> str:
        total_images = len({*self._series, *self._added})
        return f"<{self.__class__.__name__}: {total_images} images>"

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]  # (locks can't be pickled: see __setstate__)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = _store_lock(self._dest)

    # ------------------ Internal methods ------------------

    def _check_series(
        self,
        name: str,
        image: Any,
        datasets: Any,
        chunks: ShapeLike | Literal["auto"] | Default | None,
        shards: ShapeLike | Default | None,
        compression: CompressionName | Default,
    ) -> tuple[Sequence[Any], dict[str, Any]]:
        """Check a series before anything is written; return datasets and options."""
        check_image_path(name)
        if name.split("/")[0].casefold() == "ome":
            raise ValueError(
                f"Invalid series name {name!r}: 'OME' is the group of the collection's "
                "metadata."
            )
        options = self._array_options(chunks, shards, compression)
        _, datasets_seq = check_image_args(
            image,
            datasets,
            self._image_type,
            f"Series '{name}': ",
            options["chunks"],
            options["shards"],
        )
        get_create_func(self._writer)  # (raises for an unknown/missing backend)
        return datasets_seq, options

    def _reserve(self, name: str, method: str) -> None:
        """Reserve `name` for a series added with `method`, if it's free."""
        with self._lock:
            if (previous := self._added.get(name)) is not None:
                raise ValueError(f"Series '{name}' already {_ADDED_VIA[previous]}.")
            self._check_clash(name)
            if not self._wipe_pending and (self._dest / name).exists():
                # (perhaps another builder has listed it since this one last looked)
                self._set_series(self._read_listing() or self._series)
                self._check_clash(name)
                raise FileExistsError(_not_a_series_message(self._dest / name))
            self._added[name] = method
            self._names.add(name)

    def _check_clash(self, name: str) -> None:
        """Raise if `name` clashes with a series of the collection or this builder."""
        if (clash := self._names.clash(name)) is None:
            return
        other, same_path = clash
        if not same_path:
            raise ValueError(
                f"Series '{name}' can't be nested in (or contain) series '{other}'."
            )
        if other in self._added:
            raise ValueError(
                f"Series '{name}' and '{other}' differ only in case (so they'd be "
                "the same directory on some file systems)."
            )
        raise FileExistsError(f"Series '{other}' already exists in {self._dest}.")

    def _create(
        self,
        series: Mapping[str, tuple[Any, Sequence[Any], dict[str, Any]]],
        *,
        data: Mapping[str, Sequence[Any]] | None = None,
        progress: bool = False,
        release: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Create the (reserved) `series`, then list them in the collection.

        Returns their arrays, by series name and dataset path.  On failure, the series
        created by this call are deleted (and, if `release`, their names freed).
        """
        created: list[str] = []
        try:
            with self._lock:
                self._begin()
            arrays: dict[str, dict[str, Any]] = {}
            for name, (image, specs, options) in series.items():
                path = self._dest / name
                ensure_parent_groups(self._dest, name)
                try:
                    _claim_dir(path)
                except FileExistsError:
                    with self._lock:
                        self._added.pop(name, None)  # (it would fail again: drop it)
                    raise FileExistsError(_not_a_series_message(path)) from None
                created.append(name)
                _, arrays[name] = prepare_image(
                    path,
                    image,
                    specs,
                    image_type=self._image_type,
                    writer=self._writer,
                    overwrite=False,
                    **options,
                )
                if data is not None:
                    for arr, values in zip(
                        arrays[name].values(), data[name], strict=True
                    ):
                        write_to_array(arr, values, progress=progress)
            with self._lock:
                self._commit(list(series))
        except BaseException:
            with self._lock:
                for name in series:
                    if name in self._series:
                        continue  # (it was listed: it's part of the collection now)
                    if name in created:
                        _discard(self._dest / name)
                    if release:
                        self._added.pop(name, None)
                self._names = _SeriesNames([*self._series, *self._added])
            raise
        return arrays

    def _begin(self) -> None:
        """Ready the store for new series: replace (if `overwrite`), start, or convert.

        (Called with the lock held, before a series is created.)
        """
        dest = self._dest
        if self._wipe_pending:
            if dest.exists() and not is_empty_dir(dest):
                # (be cautious: don't delete anything that doesn't look like a group)
                markers = ("zarr.json", ".zgroup", _CREATING_MARKER, "OME/zarr.json")
                if not any((dest / marker).exists() for marker in markers):
                    raise FileExistsError(
                        f"Destination {dest} exists, but is not a Zarr group. "
                        "Refusing to overwrite.  Please delete manually."
                    )
                _remove_tree(dest)
            self._wipe_pending = False
            self._set_series([])
        if not (dest / "zarr.json").exists():
            # (marks the directory as a collection in the making, until its root
            # zarr.json is written along with the first series)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / _CREATING_MARKER).touch()
        elif self._numbered:
            # Without a `series` list, the numbered groups are the collection, so a new
            # one would be part of it before it's complete: list them explicitly first.
            if self._read_listing() is None and (numbered := _numbered_series(dest)):
                self._write_listing(numbered)
                self._set_series(numbered)
            self._numbered = False

    def _commit(self, names: Sequence[str]) -> None:
        """List the (created) series `names` in OME/zarr.json, after any already there.

        A new collection's root zarr.json is written last: until then, `dest` isn't a
        Zarr group.  (Called with the lock held.)
        """
        dest = self._dest
        listed = self._read_listing() or []
        known = set(listed)
        listing = [*listed, *(name for name in names if name not in known)]
        (dest / "OME").mkdir(exist_ok=True)
        if self._ome_xml is not None:
            self._write_ome_xml()
        self._write_listing(listing)
        self._set_series(listing)

        root = dest / "zarr.json"
        if not root.exists():
            bf2raw = self._bf2raw_type(bioformats2raw_layout=3)
            ome = bf2raw.model_dump(mode="json", exclude_none=True)
            attrs = {"ome": ome, **(self._extra_attributes or {})}
            write_group_json(dest, attrs, indent=2)
        elif self._extra_pending:
            existing = json.loads(root.read_bytes()).get("attributes", {})
            extra = {
                k: v for k, v in (self._extra_attributes or {}).items() if k != "ome"
            }
            write_group_json(dest, {**existing, **extra}, indent=2)
        self._extra_pending = False
        (dest / _CREATING_MARKER).unlink(missing_ok=True)

    def _set_series(self, series: list[str]) -> None:
        """Set the collection's series, as read from (or written to) OME/zarr.json."""
        if series[: len(self._series)] == self._series:  # (the usual case: appended)
            for name in series[len(self._series) :]:
                self._names.add(name)
        else:
            self._names = _SeriesNames([*series, *self._added])
        self._series = series

    def _write_listing(self, series: list[str]) -> None:
        rewrite_zarr3_group(self._dest / "OME", self._series_type(series=series))

    def _write_ome_xml(self) -> None:
        (ome := self._dest / "OME").mkdir(parents=True, exist_ok=True)
        atomic_write(ome / "METADATA.ome.xml", cast("str", self._ome_xml))
        self._ome_xml = None

    def _read_listing(self) -> list[str] | None:
        """The `series` in OME/zarr.json, or None if there are none."""
        zarr_json = self._dest / "OME" / "zarr.json"
        if not zarr_json.is_file():
            return None
        try:
            ome = json.loads(zarr_json.read_bytes()).get("attributes", {}).get("ome")
        except (ValueError, AttributeError):
            raise FileExistsError(
                f"{zarr_json} isn't a valid zarr.json; refusing to overwrite it."
            ) from None
        series = ome.get("series") if isinstance(ome, dict) else None
        if series is None:
            return None
        if not isinstance(series, list) or not all(isinstance(s, str) for s in series):
            raise FileExistsError(
                f"{zarr_json} doesn't hold a valid `series` list; refusing to "
                "overwrite it."
            )
        return series

    def _load_existing(self) -> list[str]:
        """Return the series of the collection already at `dest` (if any)."""
        dest = self._dest
        if not dest.exists() or is_empty_dir(dest):
            return []
        if (dest / "zarr.json").is_file():
            self._check_existing_root()
            if (listing := self._read_listing()) is not None:
                return listing
            self._numbered = True
            return _numbered_series(dest)
        if (dest / _CREATING_MARKER).exists():
            # (a collection whose first series wasn't finished: carry on with it)
            if (listing := self._read_listing()) is None:
                return []
            ome = json.loads((dest / "OME" / "zarr.json").read_bytes())
            try:
                self._series_type.model_validate(ome["attributes"]["ome"])
            except ValueError:
                version = self._bf2raw_type.model_fields["version"].default
                raise FileExistsError(
                    f"{dest} holds an unfinished collection that isn't v{version}.  "
                    "Delete it first."
                ) from None
            return listing
        raise FileExistsError(
            f"{dest} exists, but is not a Zarr v3 group: refusing to add to it."
        )

    def _check_existing_root(self) -> None:
        """Check that the group at `dest` is a collection of this version."""
        version = self._bf2raw_type.model_fields["version"].default
        try:
            doc = json.loads((self._dest / "zarr.json").read_bytes())
            ome = doc["attributes"]["ome"]
            if doc["zarr_format"] != 3 or doc["node_type"] != "group" or "plate" in ome:
                raise ValueError
            self._bf2raw_type.model_validate(ome)
        except (ValueError, KeyError, TypeError):
            raise FileExistsError(
                f"{self._dest} already holds a Zarr group that isn't a v{version} "
                "bioformats2raw collection.  Use overwrite=True to replace it "
                "(deleting everything in it)."
            ) from None


def _not_a_series_message(path: Path) -> str:
    return (
        f"{path} exists, but isn't a series of the collection (perhaps left by an "
        "interrupted write).  Delete it, or use another name."
    )


class PlateBuilderBase(BuilderBase, Generic[Image]):
    """Builder for Plate hierarchies (see `vNN.PlateBuilder`)."""

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
        super().__init__(
            dest,
            writer=writer,
            chunks=chunks,
            shards=shards,
            overwrite=overwrite,
            compression=compression,
        )
        if plate is not None:
            check_model(plate, self._plate_type, "plate: ")
        self._user_plate = plate
        self._extra_attributes = extra_attributes
        # (a plate already at `dest` is added to, not replaced)
        self._existing_plate = self._load_existing_plate()

        # For prepare-only workflow: {well_path: {fov: (Image, specs)}}
        self._wells: dict[str, dict[str, ImageWithShapeSpecs]] = {}
        # For immediate write workflow
        self._plate_written = False
        # {(row, col): {fov: (Image, datasets)}}
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
        self._validate_well_coordinates(row, col)
        normalized_fields = self._check_fields(row, col, images)
        # (after the checks, so nothing is created for a well that would be rejected)
        self._ensure_initialized()

        self._written_wells_data[(row, col)] = cast(
            "dict[str, ImageWithDatasets]", normalized_fields
        )
        self._update_plate_metadata()

        well_path = f"{row}/{col}"
        self._create_well_group(well_path, list(images))
        for fov, (image_model, datasets_seq) in normalized_fields.items():
            self._write(image_model, datasets_seq, well_path, fov, progress=progress)
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
        self._validate_well_coordinates(row, col)
        normalized_fields = cast(
            "dict[str, ImageWithShapeSpecs[Image]]",
            self._check_fields(row, col, images),
        )
        self._wells[f"{row}/{col}"] = normalized_fields
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

        plate = merge_plate_metadata(
            self._get_images_dict(), self._user_plate, self._plate_type
        )
        create_zarr3_group(
            self._dest, plate, self._overwrite, extra_attributes=self._extra_attributes
        )

        all_arrays: dict[str, Any] = {}
        for well_path, fields in self._wells.items():
            self._create_well_group(well_path, list(fields))
            for fov, (image_model, datasets) in fields.items():
                all_arrays.update(self._prepare(image_model, datasets, well_path, fov))
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
            context = f"Well '{row}/{col}', field '{fov}': "
            datasets_seq = self._check_datasets(image_model, datasets, context)
            normalized[fov] = (image_model, datasets_seq)
        # (the well metadata validates the field names)
        self._generate_well_metadata(list(images))
        return normalized

    def _validate_well_coordinates(self, row: str, col: str) -> None:
        """Validate that well coordinates haven't been used yet."""
        if (row, col) in self._written_wells_data:
            raise ValueError(f"Well ({row}, {col}) already written via write_well().")
        well_path = f"{row}/{col}"
        if well_path in self._wells:
            raise ValueError(f"Well ({row}, {col}) already added via add_well().")
        if self._existing_plate is not None and well_path in {
            well.path for well in self._existing_plate.plate.wells
        }:
            raise FileExistsError(
                f"Well ({row}, {col}) already exists in {self._dest}."
            )
        if self._user_plate is not None:
            valid_well_paths = [well.path for well in self._user_plate.plate.wells]
            if well_path not in valid_well_paths:
                raise ValueError(
                    f"Well path '{well_path}' not found in plate metadata. "
                    f"Valid wells are: {valid_well_paths}"
                )

    def _load_existing_plate(self) -> BaseModel | None:
        """Return the Plate already at `dest`, if there is one (and no `overwrite`)."""
        zarr_json = self._dest / "zarr.json"
        if self._overwrite or not zarr_json.exists():
            return None
        ome = json.loads(zarr_json.read_bytes()).get("attributes", {}).get("ome", {})
        return self._plate_type.model_validate(ome) if "plate" in ome else None

    def _get_images_dict(self) -> dict[tuple[str, str, str], ImageWithAny]:
        """Return the written and added wells' fields, keyed by `(row, col, fov)`."""
        images_dict: dict[tuple[str, str, str], ImageWithAny] = {}
        for (row, col), fields in self._written_wells_data.items():
            for fov, image_data in fields.items():
                images_dict[(row, col, fov)] = image_data
        for well_path, fields in self._wells.items():
            row, col = well_path.split("/")
            for fov, image_data in fields.items():
                images_dict[(row, col, fov)] = image_data
        return images_dict

    def _generate_current_plate_metadata(self) -> Any:
        """Return the given Plate, or one generated from the wells written so far.

        (A plate already at `dest` is extended with those wells.)
        """
        if self._user_plate is not None:
            return self._user_plate
        if self._existing_plate is not None:
            new = autogenerate_plate_metadata(self._get_images_dict())
            return extend_plate_metadata(self._existing_plate, new)
        return merge_plate_metadata(self._get_images_dict(), None, self._plate_type)

    def _update_plate_metadata(self) -> None:
        """(Re)write the plate's zarr.json, with the wells written so far."""
        # (a given Plate doesn't change: rewriting it after every well is O(n^2))
        if self._user_plate is not None and self._plate_written:
            return
        plate = self._generate_current_plate_metadata()
        ome = plate.model_dump(mode="json", exclude_none=True)
        attributes = {"ome": ome, **(self._extra_attributes or {})}
        write_group_json(self._dest, attributes, indent=2)
        self._plate_written = True

    def _create_well_group(self, well_path: str, field_names: list[str]) -> None:
        """Create the group for well `well_path` ("A/1"), and the row group above it."""
        ensure_parent_groups(self._dest, well_path)
        well_metadata = self._generate_well_metadata(field_names)
        create_zarr3_group(self._dest / well_path, well_metadata, self._overwrite)

    def _generate_well_metadata(self, field_names: list[str]) -> BaseModel:
        """Generate Well metadata listing the (sorted) `field_names`."""
        images = [
            self._field_of_view_type(path=fov, acquisition=None)
            for fov in sorted(field_names)
        ]

        return self._well_type(well=self._well_def_type(images=images))


class LabelsBuilderBase(BuilderBase, Generic[LabelImage]):
    """Builder for labels groups within an Image (see `vNN.LabelsBuilder`)."""

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
        super().__init__(
            dest,
            writer=writer,
            chunks=chunks,
            shards=shards,
            overwrite=overwrite,
            compression=compression,
        )
        # For prepare-only workflow: {label_name: (LabelImage, specs)}
        self._labels: dict[str, tuple[LabelImage, ShapeAndDTypeOrPyramid]] = {}
        # For immediate write workflow
        self._written_labels: list[str] = []
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
        context = f"Label '{name}': "
        datasets_seq = self._check_datasets(label_image, datasets, context)
        check_label_dtypes(datasets_seq, context)
        self._ensure_initialized()
        self._update_labels_group(name)
        # (`_image_type` is the Image model, of which LabelImage is a subclass)
        self._write(label_image, datasets, name, progress=progress)
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
        context = f"Label '{name}': "
        specs_seq = self._check_datasets(label_image, datasets, context)
        check_label_dtypes(specs_seq, context)
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

        labels_group = self._labels_group_type(labels=self._all_labels(self._labels))
        # (an existing labels group is added to, not replaced)
        if (self._dest / "zarr.json").exists():
            update_zarr3_group(self._dest, labels_group)
        else:
            create_zarr3_group(self._dest, labels_group, self._overwrite)

        all_arrays: dict[str, Any] = {}
        for label_name, (label_image, datasets) in self._labels.items():
            all_arrays.update(self._prepare(label_image, datasets, label_name))
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
        if name in self._preexisting_labels and not self._overwrite:
            raise ValueError(
                f"Label '{name}' already exists in the labels group. "
                f"Use overwrite=True to replace it."
            )

    def _all_labels(self, new: Iterable[str]) -> list[str]:
        """The labels already in the group, then those of `new` that aren't."""
        existing = self._preexisting_labels
        return [*existing, *(label for label in new if label not in existing)]

    def _update_labels_group(self, label_name: str) -> None:
        """Add `label_name` to the labels listed in labels/zarr.json."""
        if label_name in self._written_labels:
            # already added ... this is an internal method, don't need to raise
            return  # pragma: no cover

        self._written_labels.append(label_name)
        labels = self._all_labels(self._written_labels)
        rewrite_zarr3_group(self._dest, self._labels_group_type(labels=labels))
