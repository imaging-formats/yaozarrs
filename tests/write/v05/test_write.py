"""Tests for OME-Zarr v0.5 write functionality."""

from __future__ import annotations

import doctest
import importlib.metadata
import importlib.util
import json
import math
from typing import TYPE_CHECKING

import pytest

import yaozarrs
from yaozarrs import DimSpec, v05
from yaozarrs.write.v05 import (
    Bf2RawBuilder,
    PlateBuilder,
    _write,
    prepare_image,
    write_bioformats2raw,
    write_image,
    write_plate,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from yaozarrs.write.v05._write import CompressionName, ZarrWriter


try:
    import numpy as np
except ImportError:
    pytest.skip("numpy not available", allow_module_level=True)

WRITERS: list[ZarrWriter] = []
if importlib.util.find_spec("zarr") is not None:
    zarr_version_str = importlib.metadata.version("zarr")
    if int(zarr_version_str.split(".")[0]) >= 3:
        WRITERS.append("zarr")
if importlib.util.find_spec("tensorstore") is not None:
    WRITERS.append("tensorstore")

if not WRITERS:
    pytest.skip(
        "No supported Zarr writer (zarrs, or tensorstore) found",
        allow_module_level=True,
    )

np.random.seed(42)


def _make_image(name: str, dims_scale: Mapping[str, float]) -> v05.Image:
    """Create a simple v05.Image model for testing."""
    dims = [DimSpec(name=n, scale=s) for n, s in dims_scale.items()]
    return v05.Image(multiscales=[v05.Multiscale.from_dims(dims, name=name)])


def _make_multiscale_image(
    name: str = "test", n_levels: int = 3
) -> tuple[list[np.ndarray], v05.Image]:
    """Create multiscale pyramid data and matching Image model."""
    shape = (2, 128, 128)  # CYX
    dims = [DimSpec(name=n, scale=s) for n, s in [("c", 1.0), ("y", 0.5), ("x", 0.5)]]
    data = [
        np.random.rand(shape[0], shape[1] // 2**i, shape[2] // 2**i).astype("float32")
        for i in range(n_levels)
    ]
    ms = v05.Multiscale.from_dims(dims, name=name, n_levels=n_levels)
    return data, v05.Image(multiscales=[ms])


def _make_plate(
    n_rows: int = 2, n_cols: int = 2
) -> tuple[v05.Plate, dict[tuple[str, str, str], tuple[v05.Image, list[np.ndarray]]]]:
    """Create a simple plate for testing."""
    row_names = [chr(ord("A") + i) for i in range(n_rows)]
    col_names = [f"{i + 1:02d}" for i in range(n_cols)]

    wells = [
        v05.PlateWell(path=f"{r}/{c}", rowIndex=ri, columnIndex=ci)
        for ri, r in enumerate(row_names)
        for ci, c in enumerate(col_names)
    ]
    plate = v05.Plate(
        plate=v05.PlateDef(
            columns=[v05.Column(name=name) for name in col_names],
            rows=[v05.Row(name=name) for name in row_names],
            wells=wells,
        )
    )
    images = {
        (r, c, "0"): (
            _make_image(f"{r}/{c}/0", {"c": 1.0, "y": 0.5, "x": 0.5}),
            [np.random.rand(2, 64, 64).astype("float32")],
        )
        for r in row_names
        for c in col_names
    }
    return plate, images


# =============================================================================
# write_image tests
# =============================================================================


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    ("ndim", "dims_scale", "shape"),
    [
        (2, {"y": 0.5, "x": 0.5}, (64, 64)),
        (3, {"c": 1.0, "y": 0.5, "x": 0.5}, (2, 64, 64)),
        (4, {"t": 100.0, "c": 1.0, "y": 0.5, "x": 0.5}, (5, 2, 32, 32)),
    ],
    ids=["2D", "3D", "4D"],
)
def test_write_image_dimensions(
    tmp_path: Path, writer: ZarrWriter, ndim: int, dims_scale: dict, shape: tuple
) -> None:
    """Test write_image with different dimensions."""
    dest = tmp_path / f"test{ndim}d.zarr"
    data = np.random.rand(*shape).astype("float32")
    result = write_image(
        dest, _make_image(f"test{ndim}d", dims_scale), datasets=[data], writer=writer
    )
    assert result == dest and dest.exists()
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_multiscale(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with multiple resolution levels."""
    dest = tmp_path / "multiscale.zarr"
    datasets, image = _make_multiscale_image("pyramid", n_levels=3)
    write_image(dest, image, datasets=datasets, writer=writer)
    for level in range(3):
        assert (dest / str(level)).exists()
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    ("chunks", "data_shape", "expected"),
    [
        ((1, 32, 32), (2, 64, 64), [1, 32, 32]),
        ("auto", (128, 256, 256), [64, 128, 128]),
        (None, (2, 32, 32), [2, 32, 32]),
    ],
    ids=["custom", "auto", "none"],
)
def test_write_image_chunks(
    tmp_path: Path, writer: ZarrWriter, chunks, data_shape: tuple, expected: list
) -> None:
    """Test write_image with different chunk configurations."""
    dest = tmp_path / "chunks.zarr"
    data = np.random.rand(*data_shape).astype("float32")
    write_image(
        dest,
        _make_image("chunks_test", {"c": 1.0, "y": 0.5, "x": 0.5}),
        datasets=[data],
        chunks=chunks,
        writer=writer,
    )
    arr_meta = json.loads((dest / "0" / "zarr.json").read_bytes())
    assert arr_meta["chunk_grid"]["configuration"]["chunk_shape"] == expected
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_metadata_correct(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test that written metadata matches input Image model."""
    dest = tmp_path / "metadata.zarr"
    data = np.random.rand(2, 64, 64).astype("float32")
    write_image(
        dest,
        _make_image("metadata_test", {"c": 1.0, "y": 0.5, "x": 0.5}),
        datasets=[data],
        writer=writer,
    )
    meta = json.loads((dest / "zarr.json").read_bytes())
    assert meta["attributes"]["ome"]["multiscales"][0]["name"] == "metadata_test"
    yaozarrs.validate_zarr_store(dest)


def test_write_image_mismatch_datasets_error(tmp_path: Path) -> None:
    """Test that mismatched number of datasets raises error."""
    # Image model with 2 datasets but only 1 array provided
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                axes=[
                    v05.ChannelAxis(name="c"),
                    v05.SpaceAxis(name="y"),
                    v05.SpaceAxis(name="x"),
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1.0, 0.5, 0.5])
                        ],
                    ),
                    v05.Dataset(
                        path="1",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1.0, 1.0, 1.0])
                        ],
                    ),
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="Number of data arrays"):
        write_image(
            tmp_path / "mismatch.zarr",
            image,
            datasets=[np.random.rand(2, 64, 64).astype("float32")],
        )


def test_write_image_invalid_writer(tmp_path: Path) -> None:
    """Test that invalid writer raises error."""
    with pytest.raises(ValueError, match="Unknown writer"):
        write_image(  # ty: ignore[no-matching-overload]
            tmp_path / "invalid.zarr",
            _make_image("invalid", {"c": 1.0, "y": 0.5, "x": 0.5}),
            datasets=[np.random.rand(2, 64, 64).astype("float32")],
            writer="invalid",  # type: ignore
        )


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_with_omero(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with omero metadata."""
    dest = tmp_path / "omero.zarr"
    image = _make_image("rgb_image", {"c": 1.0, "y": 0.5, "x": 0.5}).model_copy(
        update={
            "omero": v05.Omero(
                channels=[
                    v05.OmeroChannel(
                        window=v05.OmeroWindow(start=0, min=0, end=255, max=255),
                        color="FF0000",
                        label="Red",
                        active=True,
                    ),
                    v05.OmeroChannel(
                        window=v05.OmeroWindow(start=0, min=0, end=255, max=255),
                        color="00FF00",
                        label="Green",
                        active=True,
                    ),
                    v05.OmeroChannel(
                        window=v05.OmeroWindow(start=0, min=0, end=255, max=255),
                        color="0000FF",
                        label="Blue",
                        active=True,
                    ),
                ],
                rdefs=v05.OmeroRenderingDefs(model="color"),
            )
        }
    )
    write_image(
        dest,
        image,
        datasets=[np.random.rand(3, 64, 64).astype("float32")],
        writer=writer,
    )
    meta = json.loads((dest / "zarr.json").read_bytes())
    assert len(meta["attributes"]["ome"]["omero"]["channels"]) == 3
    assert meta["attributes"]["ome"]["omero"]["channels"][0]["color"] == "FF0000"
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32", "float64"])
def test_write_image_different_dtypes(
    tmp_path: Path, writer: ZarrWriter, dtype
) -> None:
    """Test write_image with various data types."""
    dest = tmp_path / f"dtype_{dtype}.zarr"
    data = np.random.rand(2, 32, 32).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        data = (data * 255).astype(dtype)
    write_image(
        dest,
        _make_image(f"dtype_{dtype}", {"c": 1.0, "y": 0.5, "x": 0.5}),
        datasets=[data],
        writer=writer,
    )
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    ("compression", "expected_codec", "expected_config"),
    [
        ("blosc-zstd", "blosc", {"cname": "zstd", "clevel": 3, "shuffle": "shuffle"}),
        ("blosc-lz4", "blosc", {"cname": "lz4", "clevel": 5, "shuffle": "shuffle"}),
        ("zstd", "zstd", {"level": 3, "checksum": False}),
        ("none", None, None),
    ],
)
def test_write_image_compression_options(
    tmp_path: Path,
    writer: ZarrWriter,
    compression: CompressionName,
    expected_codec: str | None,
    expected_config: dict | None,
) -> None:
    """Test that each writer backend correctly honors each compression option."""
    dest = tmp_path / f"{writer}_{compression}.zarr"
    write_image(
        dest,
        _make_image(f"{compression}_test", {"c": 1.0, "y": 0.5, "x": 0.5}),
        datasets=[np.random.rand(2, 32, 32).astype("float32")],
        writer=writer,
        compression=compression,  # type: ignore
    )
    arr_meta = json.loads((dest / "0" / "zarr.json").read_bytes())
    codecs = arr_meta.get("codecs", [])
    assert codecs[0]["name"] == "bytes"
    if expected_codec:
        assert codecs[1]["name"] == expected_codec
        for key, val in expected_config.items():  # type: ignore
            assert codecs[1]["configuration"][key] == val
    else:
        assert len(codecs) == 1
    yaozarrs.validate_zarr_store(dest)


# =============================================================================
# write_bioformats2raw tests
# =============================================================================


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw_single_series(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_bioformats2raw with a single series."""
    dest = tmp_path / "bf2raw.zarr"
    data = np.random.rand(2, 64, 64).astype("float32")
    write_bioformats2raw(
        dest,
        {"0": (_make_image("series_0", {"c": 1.0, "y": 0.5, "x": 0.5}), [data])},
        writer=writer,
    )
    root_meta = json.loads((dest / "zarr.json").read_bytes())
    assert root_meta["attributes"]["ome"]["bioformats2raw.layout"] == 3
    ome_meta = json.loads((dest / "OME" / "zarr.json").read_bytes())
    assert ome_meta["attributes"]["ome"]["series"] == ["0"]
    yaozarrs.validate_zarr_store(str(dest / "0"))


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw_multiple_series(
    tmp_path: Path, writer: ZarrWriter
) -> None:
    """Test write_bioformats2raw with multiple series."""
    dest = tmp_path / "multi_series.zarr"
    images = {
        str(i): (
            _make_image(f"series_{i}", {"c": 1.0, "y": 0.5, "x": 0.5}),
            [np.random.rand(2, 32, 32).astype("float32")],
        )
        for i in range(3)
    }
    write_bioformats2raw(dest, images, writer=writer)
    ome_meta = json.loads((dest / "OME" / "zarr.json").read_bytes())
    assert ome_meta["attributes"]["ome"]["series"] == ["0", "1", "2"]
    for i in range(3):
        yaozarrs.validate_zarr_store(str(dest / str(i)))


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw_with_ome_xml(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_bioformats2raw with OME-XML metadata."""
    dest = tmp_path / "with_xml.zarr"
    ome_xml = '<?xml version="1.0"?><OME xmlns="test">test content</OME>'
    write_bioformats2raw(
        dest,
        {
            "0": (
                _make_image("with_xml", {"c": 1.0, "y": 0.5, "x": 0.5}),
                [np.zeros((2, 64, 64))],
            )
        },
        ome_xml=ome_xml,
        writer=writer,
    )
    assert (dest / "OME" / "METADATA.ome.xml").read_text() == ome_xml


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw_multiscale_series(
    tmp_path: Path, writer: ZarrWriter
) -> None:
    """Test write_bioformats2raw with multiscale pyramid series."""
    dest = tmp_path / "pyramid_series.zarr"
    datasets, image = _make_multiscale_image("pyramid_series", n_levels=2)
    write_bioformats2raw(dest, {"0": (image, datasets)}, writer=writer)
    assert (dest / "0" / "0").exists() and (dest / "0" / "1").exists()
    yaozarrs.validate_zarr_store(str(dest / "0"))


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    ("first_method", "second_method", "error_match"),
    [
        ("write_image", "add_series", "already written via write_image"),
        ("add_series", "write_image", "already added via add_series"),
    ],
)
def test_bf2raw_builder_conflict_detection(
    tmp_path: Path,
    writer: ZarrWriter,
    first_method: str,
    second_method: str,
    error_match: str,
) -> None:
    """Test that Bf2RawBuilder detects conflicts between write_image and add_series."""
    dest = tmp_path / "conflict.zarr"
    data = np.random.rand(2, 32, 32).astype("float32")
    image = _make_image("test", {"c": 1.0, "y": 0.5, "x": 0.5})
    builder = Bf2RawBuilder(dest, writer=writer)
    getattr(builder, first_method)("0", image, [data])
    with pytest.raises(ValueError, match=error_match):
        getattr(builder, second_method)("0", image, [data])


# =============================================================================
# write_plate and PlateBuilder tests
# =============================================================================


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    ("n_rows", "n_cols"),
    [(1, 1), (2, 2), (4, 6)],
    ids=["1x1", "2x2", "4x6"],
)
def test_write_plate_grid_sizes(
    tmp_path: Path, writer: ZarrWriter, n_rows: int, n_cols: int
) -> None:
    """Test write_plate with various plate grid sizes."""
    dest = tmp_path / f"plate_{n_rows}x{n_cols}.zarr"
    plate, images = _make_plate(n_rows=n_rows, n_cols=n_cols)
    result = write_plate(dest, images, plate=plate, writer=writer)
    assert result == dest
    plate_meta = json.loads((dest / "zarr.json").read_bytes())
    assert len(plate_meta["attributes"]["ome"]["plate"]["wells"]) == n_rows * n_cols
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate_multi_field(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_plate with multiple fields per well."""
    dest = tmp_path / "plate_multi_field.zarr"

    # Create simple 1x1 plate with 2 fields
    plate = v05.Plate(
        plate=v05.PlateDef(
            columns=[v05.Column(name="01")],
            rows=[v05.Row(name="A")],
            wells=[v05.PlateWell(path="A/01", rowIndex=0, columnIndex=0)],
        )
    )

    images = {}
    for fov in ["0", "1"]:
        data = np.random.rand(2, 32, 32).astype("float32")
        image = _make_image(f"field_{fov}", {"c": 1.0, "y": 0.5, "x": 0.5})
        images[("A", "01", fov)] = (image, [data])

    result = write_plate(dest, images, plate=plate, writer=writer)
    assert result == dest

    # Check both fields exist
    well_path = dest / "A" / "01"
    assert (well_path / "0").exists()
    assert (well_path / "1").exists()

    # Check well metadata has both images
    well_meta = json.loads((well_path / "zarr.json").read_bytes())
    assert len(well_meta["attributes"]["ome"]["well"]["images"]) == 2

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_plate_builder_immediate_write(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test PlateBuilder immediate write workflow."""
    dest = tmp_path / "builder_immediate.zarr"
    plate, images_mapping = _make_plate(n_rows=1, n_cols=2)
    builder = PlateBuilder(dest, plate=plate, writer=writer)
    # Group images by well and write
    wells_data: dict[tuple[str, str], dict[str, tuple[v05.Image, list]]] = {}
    for (row, col, fov), (image, datasets) in images_mapping.items():
        wells_data.setdefault((row, col), {})[fov] = (image, datasets)
    for (row, col), fields in wells_data.items():
        assert builder.write_well(row=row, col=col, images=fields) is builder
    assert repr(builder) == "<PlateBuilder: 2 wells>"
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_plate_builder_prepare_only(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test PlateBuilder prepare-only workflow."""

    dest = tmp_path / "builder_prepare.zarr"
    plate, images_mapping = _make_plate(n_rows=1, n_cols=1)

    builder = PlateBuilder(dest, plate=plate, writer=writer)

    # Add wells - convert arrays to specs (shape, dtype) for add_well
    for (row, col, fov), (image, datasets) in images_mapping.items():
        specs = [(arr.shape, arr.dtype) for arr in datasets]
        result = builder.add_well(row=row, col=col, images={fov: (image, specs)})
        assert result is builder

    # Prepare
    path, arrays = builder.prepare()
    assert path == dest
    assert "A/01/0/0" in arrays

    # Write data to arrays
    for _key, arr in arrays.items():
        # Get shape from array and write matching data
        data = np.random.rand(*arr.shape).astype("float32")
        arr[:] = data

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_plate_builder_well_metadata_generation(
    tmp_path: Path, writer: ZarrWriter
) -> None:
    """Test that PlateBuilder correctly generates well metadata."""
    dest = tmp_path / "well_metadata.zarr"
    plate, _ = _make_plate(n_rows=1, n_cols=1)
    builder = PlateBuilder(dest, plate=plate, writer=writer)
    # Create well with multiple fields in non-sorted order
    fields = {
        fov: (
            _make_image(f"field_{fov}", {"c": 1.0, "y": 0.5, "x": 0.5}),
            [np.random.rand(2, 32, 32).astype("float32")],
        )
        for fov in ["2", "0", "1"]
    }
    builder.write_well(row="A", col="01", images=fields)
    well_meta = json.loads((dest / "A" / "01" / "zarr.json").read_bytes())
    images_meta = well_meta["attributes"]["ome"]["well"]["images"]
    assert [img["path"] for img in images_meta] == ["0", "1", "2"]


@pytest.mark.parametrize(
    ("setup", "action", "error_match"),
    [
        # invalid well path
        (
            lambda b, img, data: None,
            lambda b, img, data: b.write_well(
                row="B", col="01", images={"0": (img, [data])}
            ),
            "not found in plate metadata",
        ),
        # duplicate write_well
        (
            lambda b, img, data: b.write_well(
                row="A", col="01", images={"0": (img, [data])}
            ),
            lambda b, img, data: b.write_well(
                row="A", col="01", images={"0": (img, [data])}
            ),
            "already written via write_well",
        ),
        # add_well then write_well
        (
            lambda b, img, data: b.add_well(
                row="A", col="01", images={"0": (img, [data])}
            ),
            lambda b, img, data: b.write_well(
                row="A", col="01", images={"0": (img, [data])}
            ),
            "already added via add_well",
        ),
        # prepare with no wells
        (
            lambda b, img, data: None,
            lambda b, img, data: b.prepare(),
            "No wells added",
        ),
        # dataset count mismatch
        (
            lambda b, img, data: None,
            lambda b, img, data: b.add_well(
                row="A", col="01", images={"0": (img, [data, np.zeros((2, 16, 16))])}
            ),
            "must match",
        ),
    ],
    ids=[
        "invalid_well",
        "duplicate_write",
        "add_then_write",
        "prepare_empty",
        "dataset_mismatch",
    ],
)
def test_plate_builder_errors(tmp_path: Path, setup, action, error_match: str) -> None:
    """Test PlateBuilder error cases."""
    plate, _ = _make_plate(n_rows=1, n_cols=1)
    builder = PlateBuilder(tmp_path / "error.zarr", plate=plate)
    data = np.random.rand(2, 32, 32).astype("float32")
    image = _make_image("test", {"c": 1.0, "y": 0.5, "x": 0.5})
    setup(builder, image, data)
    with pytest.raises((ValueError, NotImplementedError), match=error_match):
        action(builder, image, data)


def test_plate_builder_multiscale_error(tmp_path: Path) -> None:
    """Test that PlateBuilder raises error for multiple multiscales."""
    plate, _ = _make_plate(n_rows=1, n_cols=1)
    builder = PlateBuilder(tmp_path / "multiscale_error.zarr", plate=plate)
    # Create image with 2 multiscales (not supported)
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="multiscale_0",
                axes=[
                    v05.ChannelAxis(name="c"),
                    v05.SpaceAxis(name="y"),
                    v05.SpaceAxis(name="x"),
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1.0, 0.5, 0.5])
                        ],
                    )
                ],
            ),
            v05.Multiscale(
                name="multiscale_1",
                axes=[
                    v05.ChannelAxis(name="c"),
                    v05.SpaceAxis(name="y"),
                    v05.SpaceAxis(name="x"),
                ],
                datasets=[
                    v05.Dataset(
                        path="1",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1.0, 1.0, 1.0])
                        ],
                    )
                ],
            ),
        ]
    )
    with pytest.raises(NotImplementedError, match="exactly one multiscale"):
        builder.add_well(
            row="A", col="01", images={"0": (image, [((2, 32, 32), np.float32)])}
        )


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate_with_different_chunks(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_plate with custom chunk settings."""
    dest = tmp_path / "plate_chunks.zarr"
    plate, images = _make_plate(n_rows=1, n_cols=1)
    write_plate(dest, images, plate=plate, chunks=(1, 32, 32), writer=writer)
    arr_meta = json.loads((dest / "A" / "01" / "0" / "0" / "zarr.json").read_bytes())
    assert arr_meta["chunk_grid"]["configuration"]["chunk_shape"] == [1, 32, 32]
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("compression", ["blosc-zstd", "blosc-lz4", "zstd", "none"])
def test_write_plate_compression(
    tmp_path: Path, writer: ZarrWriter, compression: CompressionName
) -> None:
    """Test write_plate with different compression options."""
    dest = tmp_path / f"plate_{compression}.zarr"
    plate, images = _make_plate(n_rows=1, n_cols=1)
    write_plate(dest, images, plate=plate, compression=compression, writer=writer)  # type: ignore
    arr_meta = json.loads((dest / "A" / "01" / "0" / "0" / "zarr.json").read_bytes())
    codecs = arr_meta.get("codecs", [])
    assert len(codecs) == (1 if compression == "none" else 2)
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate_overwrite(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_plate with overwrite=True."""
    dest = tmp_path / "plate_overwrite.zarr"
    plate, images = _make_plate(n_rows=1, n_cols=1)
    write_plate(dest, images, plate=plate, writer=writer)
    with pytest.raises(FileExistsError):
        write_plate(dest, images, plate=plate, writer=writer, overwrite=False)
    write_plate(dest, images, plate=plate, writer=writer, overwrite=True)
    yaozarrs.validate_zarr_store(dest)


# =============================================================================
# prepare_image streaming tests
# =============================================================================


@pytest.mark.parametrize("writer", WRITERS)
def test_prepare_image_streaming_frames(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test prepare_image with frame-by-frame streaming writes (microscope pattern)."""
    dest = tmp_path / "streaming_5d.zarr"
    (nt, nc, nz, *_) = shape = (2, 3, 4, 16, 16)  # TCZYX
    image = _make_image(
        "streaming_test", {"t": 100.0, "c": 1.0, "z": 1.0, "y": 0.5, "x": 0.5}
    )
    _, arrays = prepare_image(dest, image, (shape, "uint16"), writer=writer)
    arr = arrays["0"]

    # Simulate microscope acquiring frames one at a time
    reference = np.zeros(shape, dtype="uint16")
    frame_count = 0
    futures = []
    for idx in np.ndindex(nt, nc, nz):
        frame = np.full((16, 16), frame_count, dtype="uint16")
        if writer == "tensorstore":
            # exercise the async write path for tensorstore
            futures.append(arr[idx].write(frame))  # type: ignore
        else:
            arr[idx] = frame
        reference[idx] = frame
        frame_count += 1

    for fut in futures:
        fut.result()

    assert frame_count == math.prod(shape[:-2])
    np.testing.assert_array_equal(arr, reference)
    yaozarrs.validate_zarr_store(dest)


# =============================================================================
# Doctests
# =============================================================================

finder = doctest.DocTestFinder()


@pytest.mark.skipif("zarr" not in WRITERS, reason="zarr writer not available")
@pytest.mark.parametrize(
    "case",
    (test for test in finder.find(_write) if test.examples),
    ids=lambda t: t.name.split(".")[-1],
)
def test_write_doctests_parametrized(
    tmp_path: Path,
    case: doctest.DocTest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.REPORTING_FLAGS
    )

    # put all paths inside the test tmp_path
    monkeypatch.setattr(_write, "Path", lambda p: tmp_path / p)
    runner.run(case)
    if runner.failures > 0:
        captured = capsys.readouterr().out.split("******************")[-1]
        pytest.fail(f"Doctest {case.name} failed:\n\n{captured}")

    for result in tmp_path.glob("*.zarr"):
        if result.is_dir() and (result / "zarr.json").exists():
            yaozarrs.validate_zarr_store(result)
