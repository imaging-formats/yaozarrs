"""Tests for OME-Zarr v0.5 write functionality."""

from __future__ import annotations

import doctest
import importlib.metadata
import importlib.util
import json
from typing import TYPE_CHECKING

import pytest

import yaozarrs
from yaozarrs import v05
from yaozarrs.write.v05 import _write, write_bioformats2raw, write_image

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
    zarr_major_version = int(zarr_version_str.split(".")[0])
    if zarr_major_version >= 3:
        WRITERS.append("zarr")
if importlib.util.find_spec("tensorstore") is not None:
    WRITERS.append("tensorstore")

if not WRITERS:
    pytest.skip(
        "No supported Zarr writer (zarrs, or tensorstore) found",
        allow_module_level=True,
    )

np.random.seed(42)


def make_simple_image(name: str, dims_scale: Mapping[str, float]) -> v05.Image:
    """Create a simple v05.Image model for testing.

    Parameters
    ----------
    name : str
        Name of the image/multiscale.
    dims_scale : Mapping[str, float]
        Mapping of dimension names to scale values.
        Supported dimensions: 'x', 'y', 'z' (spatial), 'c' (channel), 't' (time).
    """
    axes = []
    scale = []

    for dim_name, scale_value in dims_scale.items():
        if dim_name in ("x", "y", "z"):
            axes.append(v05.SpaceAxis(name=dim_name, unit="micrometer"))
        elif dim_name == "c":
            axes.append(v05.ChannelAxis(name=dim_name))
        elif dim_name == "t":
            axes.append(v05.TimeAxis(name=dim_name, unit="millisecond"))
        else:
            raise ValueError(f"Unsupported dimension: {dim_name}")
        scale.append(scale_value)

    return v05.Image(
        multiscales=[
            v05.Multiscale(
                name=name,
                axes=axes,
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=scale)
                        ],
                    )
                ],
            )
        ]
    )


def make_multiscale_image(
    name: str = "test", n_levels: int = 3
) -> tuple[list[np.ndarray], v05.Image]:
    """Create multiscale pyramid data and matching Image model."""
    shape = (2, 128, 128)  # CYX
    datasets_data = []
    datasets_meta = []

    for level in range(n_levels):
        scale_factor = 2**level
        level_shape = (shape[0], shape[1] // scale_factor, shape[2] // scale_factor)
        data = np.random.rand(*level_shape).astype(np.float32)
        datasets_data.append(data)
        datasets_meta.append(
            v05.Dataset(
                path=str(level),
                coordinateTransformations=[
                    v05.ScaleTransformation(
                        scale=[1.0, 0.5 * scale_factor, 0.5 * scale_factor]
                    )
                ],
            )
        )

    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name=name,
                axes=[
                    v05.ChannelAxis(name="c"),
                    v05.SpaceAxis(name="y", unit="micrometer"),
                    v05.SpaceAxis(name="x", unit="micrometer"),
                ],
                datasets=datasets_meta,
            )
        ]
    )

    return datasets_data, image


# =============================================================================
# write_image tests
# =============================================================================


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_write_image_dimensions(tmp_path: Path, writer: ZarrWriter, ndim: int) -> None:
    """Test write_image with different dimensions (2D, 3D, 4D)."""
    dims_scales = {
        2: {"y": 0.5, "x": 0.5},
        3: {"c": 1.0, "y": 0.5, "x": 0.5},
        4: {"t": 100.0, "c": 1.0, "y": 0.5, "x": 0.5},
    }
    shapes = {2: (64, 64), 3: (2, 64, 64), 4: (5, 2, 32, 32)}
    dest = tmp_path / f"test{ndim}d.zarr"
    data = np.random.rand(*shapes[ndim]).astype(np.float32)
    image = make_simple_image(f"test{ndim}d", dims_scales[ndim])

    result = write_image(dest, image, datasets=[data], writer=writer)

    assert result == dest
    assert dest.exists()
    if ndim == 3:  # Only check these for basic case
        assert (dest / "zarr.json").exists()
        assert (dest / "0").exists()

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_multiscale(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with multiple resolution levels."""
    dest = tmp_path / "multiscale.zarr"
    n_levels = 3
    datasets, image = make_multiscale_image("pyramid", n_levels=n_levels)

    result = write_image(dest, image, datasets=datasets, writer=writer)
    assert result == dest

    for level in range(n_levels):
        assert (dest / str(level)).exists()

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    ("chunks", "data_shape"),
    [
        ((1, 32, 32), (2, 64, 64)),  # custom chunks
        ("auto", (128, 256, 256)),  # auto chunks
        (None, (2, 32, 32)),  # no chunks
    ],
    ids=["custom", "auto", "none"],
)
def test_write_image_chunks(
    tmp_path: Path, writer: ZarrWriter, chunks, data_shape: tuple[int, ...]
) -> None:
    """Test write_image with different chunk configurations."""
    dest = tmp_path / f"chunks_{chunks}.zarr"
    data = np.random.rand(*data_shape).astype(np.float32)
    image = make_simple_image("chunks_test", {"c": 1.0, "y": 0.5, "x": 0.5})
    write_image(dest, image, datasets=[data], chunks=chunks, writer=writer)

    yaozarrs.validate_zarr_store(dest)

    arr_meta = json.loads((dest / "0" / "zarr.json").read_bytes())
    chunk_shape = arr_meta["chunk_grid"]["configuration"]["chunk_shape"]
    if chunks == "auto":
        assert chunk_shape == [64, 128, 128]
    elif chunks is not None:
        assert chunk_shape == list(chunks)
    else:
        assert chunk_shape == list(data.shape)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_metadata_correct(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test that written metadata matches input Image model."""
    dest = tmp_path / "metadata.zarr"
    data = np.random.rand(2, 64, 64).astype(np.float32)
    image = make_simple_image("metadata_test", {"c": 1.0, "y": 0.5, "x": 0.5})

    write_image(dest, image, datasets=[data], writer=writer)

    meta = json.loads((dest / "zarr.json").read_bytes())
    ome = meta["attributes"]["ome"]
    assert ome["multiscales"][0]["name"] == "metadata_test"

    yaozarrs.validate_zarr_store(dest)


def test_write_image_mismatch_datasets_error(tmp_path: Path) -> None:
    """Test that mismatched number of datasets raises error."""
    dest = tmp_path / "mismatch.zarr"
    data = [np.random.rand(2, 64, 64).astype(np.float32)]  # 1 array

    # Image model with 2 datasets
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
        write_image(dest, image, datasets=data)


# =============================================================================
# write_bioformats2raw tests
# =============================================================================


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw_single_series(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_bioformats2raw with a single series."""
    dest = tmp_path / "bf2raw.zarr"
    data = np.random.rand(2, 64, 64).astype(np.float32)
    image = make_simple_image("series_0", {"c": 1.0, "y": 0.5, "x": 0.5})

    images = {"0": (image, [data])}

    result = write_bioformats2raw(dest, images, writer=writer)

    assert result == dest
    assert dest.exists()

    # Check root has bioformats2raw.layout
    root_meta = json.loads((dest / "zarr.json").read_bytes())
    assert root_meta["attributes"]["ome"]["bioformats2raw.layout"] == 3

    # Check OME directory with series metadata
    ome_path = dest / "OME"
    assert ome_path.exists()
    ome_meta = json.loads((ome_path / "zarr.json").read_bytes())
    assert ome_meta["attributes"]["ome"]["series"] == ["0"]

    # Check image exists at 0/
    assert (dest / "0").exists()
    yaozarrs.validate_zarr_store(str(dest / "0"))


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw_multiple_series(
    tmp_path: Path, writer: ZarrWriter
) -> None:
    """Test write_bioformats2raw with multiple series."""
    dest = tmp_path / "multi_series.zarr"

    images = {}
    for i in range(3):
        data = np.random.rand(2, 32, 32).astype(np.float32)
        image = make_simple_image(f"series_{i}", {"c": 1.0, "y": 0.5, "x": 0.5})
        images[str(i)] = (image, [data])

    result = write_bioformats2raw(dest, images, writer=writer)

    assert result == dest

    # Check root
    root_meta = json.loads((dest / "zarr.json").read_bytes())
    assert root_meta["attributes"]["ome"]["bioformats2raw.layout"] == 3

    # Check OME directory
    ome_meta = json.loads((dest / "OME" / "zarr.json").read_bytes())
    assert ome_meta["attributes"]["ome"]["series"] == ["0", "1", "2"]

    # Check each series
    for i in range(3):
        assert (dest / str(i)).exists()
        yaozarrs.validate_zarr_store(str(dest / str(i)))


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw_with_ome_xml(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_bioformats2raw with OME-XML metadata."""
    dest = tmp_path / "with_xml.zarr"
    data = np.random.rand(2, 64, 64).astype(np.float32)
    image = make_simple_image("with_xml", {"c": 1.0, "y": 0.5, "x": 0.5})

    ome_xml = '<?xml version="1.0"?><OME xmlns="test">test content</OME>'
    images = {"0": (image, [data])}

    write_bioformats2raw(dest, images, ome_xml=ome_xml, writer=writer)

    # Check METADATA.ome.xml was written
    xml_path = dest / "OME" / "METADATA.ome.xml"
    assert xml_path.exists()
    assert xml_path.read_text() == ome_xml

    yaozarrs.validate_zarr_store(str(dest / "0"))


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw_multiscale_series(
    tmp_path: Path, writer: ZarrWriter
) -> None:
    """Test write_bioformats2raw with multiscale pyramid series."""
    dest = tmp_path / "pyramid_series.zarr"

    datasets, image = make_multiscale_image("pyramid_series", n_levels=2)
    images = {"0": (image, datasets)}

    write_bioformats2raw(dest, images, writer=writer)

    # Check pyramid levels exist
    assert (dest / "0" / "0").exists()
    assert (dest / "0" / "1").exists()

    yaozarrs.validate_zarr_store(str(dest / "0"))


@pytest.mark.parametrize("writer", WRITERS)
def test_bf2raw_builder_conflict_detection(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test that Bf2RawBuilder detects conflicts between write_image and add_series."""
    from yaozarrs.write.v05 import Bf2RawBuilder

    dest = tmp_path / "conflict.zarr"
    data = np.random.rand(2, 32, 32).astype(np.float32)
    image = make_simple_image("test", {"c": 1.0, "y": 0.5, "x": 0.5})

    # Test 1: write_image then add_series with same name
    builder = Bf2RawBuilder(dest, writer=writer)
    builder.write_image("0", image, [data])

    with pytest.raises(ValueError, match="already written via write_image"):
        builder.add_series("0", image, [data])

    # Test 2: add_series then write_image with same name
    dest2 = tmp_path / "conflict2.zarr"
    builder2 = Bf2RawBuilder(dest2, writer=writer)
    builder2.add_series("0", image, [data])

    with pytest.raises(ValueError, match="already added via add_series"):
        builder2.write_image("0", image, [data])


# =============================================================================
# Edge cases and error handling
# =============================================================================


def test_write_image_invalid_writer(tmp_path: Path) -> None:
    """Test that invalid writer raises error."""
    dest = tmp_path / "invalid.zarr"
    data = np.random.rand(2, 64, 64).astype(np.float32)
    image = make_simple_image("invalid", {"c": 1.0, "y": 0.5, "x": 0.5})

    with pytest.raises(ValueError, match="Unknown writer"):
        write_image(dest, image, datasets=[data], writer="invalid")  # type: ignore


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_with_omero(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with omero metadata."""
    dest = tmp_path / "omero.zarr"
    data = np.random.rand(3, 64, 64).astype(np.float32)

    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="rgb_image",
                axes=[
                    v05.ChannelAxis(name="c"),
                    v05.SpaceAxis(name="y", unit="micrometer"),
                    v05.SpaceAxis(name="x", unit="micrometer"),
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1.0, 0.5, 0.5])
                        ],
                    )
                ],
            )
        ],
        omero=v05.Omero(
            channels=[
                v05.OmeroChannel(color="FF0000", label="Red", active=True),
                v05.OmeroChannel(color="00FF00", label="Green", active=True),
                v05.OmeroChannel(color="0000FF", label="Blue", active=True),
            ],
            rdefs=v05.OmeroRenderingDefs(model="color"),
        ),
    )

    write_image(dest, image, datasets=[data], writer=writer)

    # Check omero metadata was written
    meta = json.loads((dest / "zarr.json").read_bytes())
    ome = meta["attributes"]["ome"]
    assert "omero" in ome
    assert len(ome["omero"]["channels"]) == 3
    assert ome["omero"]["channels"][0]["color"] == "FF0000"

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_write_image_different_dtypes(
    tmp_path: Path, writer: ZarrWriter, dtype
) -> None:
    """Test write_image with various data types."""
    dest = tmp_path / f"dtype_{dtype.__name__}.zarr"
    data = np.random.rand(2, 32, 32).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        data = (data * 255).astype(dtype)
    image = make_simple_image(f"dtype_{dtype.__name__}", {"c": 1.0, "y": 0.5, "x": 0.5})

    write_image(dest, image, datasets=[data], writer=writer)

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    "compression",
    ["blosc-zstd", "blosc-lz4", "zstd", "none"],
)
def test_write_image_compression_options(
    tmp_path: Path, writer: ZarrWriter, compression: CompressionName
) -> None:
    """Test that each writer backend correctly honors each compression option."""
    dest = tmp_path / f"{writer}_{compression}.zarr"
    data = np.random.rand(2, 32, 32).astype(np.float32)
    image = make_simple_image(f"{compression}_test", {"c": 1.0, "y": 0.5, "x": 0.5})

    write_image(
        dest,
        image,
        datasets=[data],
        writer=writer,
        compression=compression,  # type: ignore
    )

    # Read the array metadata
    arr_meta = json.loads((dest / "0" / "zarr.json").read_bytes())
    codecs = arr_meta.get("codecs", [])

    # All codecs should start with bytes serializer
    assert len(codecs) >= 1
    assert codecs[0]["name"] == "bytes"
    assert codecs[0]["configuration"]["endian"] == "little"

    # Check compression-specific codecs
    if compression == "blosc-zstd":
        assert len(codecs) == 2
        assert codecs[1]["name"] == "blosc"
        config = codecs[1]["configuration"]
        assert config["cname"] == "zstd"
        assert config["clevel"] == 3
        assert config["shuffle"] == "shuffle"

    elif compression == "blosc-lz4":
        assert len(codecs) == 2
        assert codecs[1]["name"] == "blosc"
        config = codecs[1]["configuration"]
        assert config["cname"] == "lz4"
        assert config["clevel"] == 5
        assert config["shuffle"] == "shuffle"

    elif compression == "zstd":
        assert len(codecs) == 2
        assert codecs[1]["name"] == "zstd"
        config = codecs[1]["configuration"]
        assert config["level"] == 3
        assert config["checksum"] is False

    elif compression == "none":
        # Only bytes codec, no compression
        assert len(codecs) == 1

    yaozarrs.validate_zarr_store(dest)


# =============================================================================
# write_plate and PlateBuilder tests
# =============================================================================


def make_simple_plate(
    n_rows: int = 2, n_cols: int = 2
) -> tuple[v05.Plate, dict[tuple[str, str, str], tuple[v05.Image, list[np.ndarray]]]]:
    """Create a simple plate for testing.

    Parameters
    ----------
    n_rows : int
        Number of rows (default 2, will use names A, B, ...)
    n_cols : int
        Number of columns (default 2, will use names 01, 02, ...)

    Returns
    -------
    tuple[Plate, dict]
        Plate metadata and images mapping
    """
    # Create plate structure
    row_names = [chr(ord("A") + i) for i in range(n_rows)]
    col_names = [f"{i + 1:02d}" for i in range(n_cols)]

    wells = []
    for row_idx, row_name in enumerate(row_names):
        for col_idx, col_name in enumerate(col_names):
            wells.append(
                v05.PlateWell(
                    path=f"{row_name}/{col_name}",
                    rowIndex=row_idx,
                    columnIndex=col_idx,
                )
            )

    plate = v05.Plate(
        plate=v05.PlateDef(
            columns=[v05.Column(name=name) for name in col_names],
            rows=[v05.Row(name=name) for name in row_names],
            wells=wells,
        )
    )

    # Create image data
    images = {}
    for row_name in row_names:
        for col_name in col_names:
            data = np.random.rand(2, 64, 64).astype(np.float32)
            image = make_simple_image(
                f"{row_name}/{col_name}/0", {"c": 1.0, "y": 0.5, "x": 0.5}
            )
            images[(row_name, col_name, "0")] = (image, [data])

    return plate, images


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate_simple_2x2(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_plate with a simple 2x2 plate."""
    from yaozarrs.write.v05 import write_plate

    dest = tmp_path / "plate_2x2.zarr"
    plate, images = make_simple_plate(n_rows=2, n_cols=2)

    result = write_plate(dest, images, plate=plate, writer=writer)

    assert result == dest
    assert dest.exists()

    # Check plate zarr.json
    plate_meta = json.loads((dest / "zarr.json").read_bytes())
    assert plate_meta["attributes"]["ome"]["version"] == "0.5"
    assert len(plate_meta["attributes"]["ome"]["plate"]["wells"]) == 4

    # Check well structure
    for row in ["A", "B"]:
        for col in ["01", "02"]:
            well_path = dest / row / col
            assert well_path.exists()
            assert (well_path / "zarr.json").exists()

            # Check well metadata
            well_meta = json.loads((well_path / "zarr.json").read_bytes())
            assert well_meta["attributes"]["ome"]["version"] == "0.5"
            assert len(well_meta["attributes"]["ome"]["well"]["images"]) == 1

            # Check field image
            assert (well_path / "0" / "zarr.json").exists()
            assert (well_path / "0" / "0").exists()

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate_multi_field(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_plate with multiple fields per well."""
    from yaozarrs.write.v05 import write_plate

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
        data = np.random.rand(2, 32, 32).astype(np.float32)
        image = make_simple_image(f"field_{fov}", {"c": 1.0, "y": 0.5, "x": 0.5})
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
    assert well_meta["attributes"]["ome"]["well"]["images"][0]["path"] == "0"
    assert well_meta["attributes"]["ome"]["well"]["images"][1]["path"] == "1"

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate_single_well(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_plate with a single well (1x1 plate)."""
    from yaozarrs.write.v05 import write_plate

    dest = tmp_path / "plate_1x1.zarr"
    plate, images = make_simple_plate(n_rows=1, n_cols=1)

    result = write_plate(dest, images, plate=plate, writer=writer)
    assert result == dest

    assert (dest / "A" / "01" / "0" / "zarr.json").exists()
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_plate_builder_immediate_write(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test PlateBuilder immediate write workflow."""
    from yaozarrs.write.v05 import PlateBuilder

    dest = tmp_path / "builder_immediate.zarr"
    plate, images_mapping = make_simple_plate(n_rows=1, n_cols=2)

    builder = PlateBuilder(dest, plate=plate, writer=writer)

    # Group images by well
    wells_data: dict[tuple[str, str], dict[str, tuple[v05.Image, list]]] = {}
    for (row, col, fov), (image, datasets) in images_mapping.items():
        if (row, col) not in wells_data:
            wells_data[(row, col)] = {}
        wells_data[(row, col)][fov] = (image, datasets)

    # Write each well
    for (row, col), fields in wells_data.items():
        result = builder.write_well(row=row, col=col, fields=fields)
        assert result is builder  # Check method chaining

    assert repr(builder) == "<PlateBuilder: 2 wells>"
    assert builder.root_path == dest

    # Check structure
    assert (dest / "A" / "01" / "0" / "zarr.json").exists()
    assert (dest / "A" / "02" / "0" / "zarr.json").exists()

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_plate_builder_prepare_only(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test PlateBuilder prepare-only workflow."""
    from yaozarrs.write.v05 import PlateBuilder

    dest = tmp_path / "builder_prepare.zarr"
    plate, images_mapping = make_simple_plate(n_rows=1, n_cols=1)

    builder = PlateBuilder(dest, plate=plate, writer=writer)

    # Add wells
    for (row, col, fov), (image, datasets) in images_mapping.items():
        result = builder.add_well(row=row, col=col, fields={fov: (image, datasets)})
        assert result is builder

    # Prepare
    path, arrays = builder.prepare()
    assert path == dest
    assert "A/01/0/0" in arrays

    # Write data to arrays
    for _key, arr in arrays.items():
        # Get shape from array and write matching data
        data = np.random.rand(*arr.shape).astype(np.float32)
        arr[:] = data

    yaozarrs.validate_zarr_store(dest)


def test_plate_builder_invalid_well_path(tmp_path: Path) -> None:
    """Test that PlateBuilder raises error for invalid well path."""
    from yaozarrs.write.v05 import PlateBuilder

    dest = tmp_path / "invalid_well.zarr"
    plate, _ = make_simple_plate(n_rows=1, n_cols=1)

    builder = PlateBuilder(dest, plate=plate, writer="zarr")
    data = np.random.rand(2, 32, 32).astype(np.float32)
    image = make_simple_image("test", {"c": 1.0, "y": 0.5, "x": 0.5})

    # Try to write well that doesn't exist in plate metadata
    with pytest.raises(ValueError, match="not found in plate metadata"):
        builder.write_well(row="B", col="01", fields={"0": (image, [data])})


def test_plate_builder_duplicate_well_write_write(tmp_path: Path) -> None:
    """Test that PlateBuilder raises error when writing same well twice."""
    from yaozarrs.write.v05 import PlateBuilder

    dest = tmp_path / "duplicate_well.zarr"
    plate, _ = make_simple_plate(n_rows=1, n_cols=1)

    builder = PlateBuilder(dest, plate=plate, writer="zarr")
    data = np.random.rand(2, 32, 32).astype(np.float32)
    image = make_simple_image("test", {"c": 1.0, "y": 0.5, "x": 0.5})

    # Write well first time
    builder.write_well(row="A", col="01", fields={"0": (image, [data])})

    # Try to write same well again
    with pytest.raises(ValueError, match="already written via write_well"):
        builder.write_well(row="A", col="01", fields={"0": (image, [data])})


def test_plate_builder_duplicate_well_add_write(tmp_path: Path) -> None:
    """Test that PlateBuilder raises error when adding then writing same well."""
    from yaozarrs.write.v05 import PlateBuilder

    dest = tmp_path / "duplicate_add_write.zarr"
    plate, _ = make_simple_plate(n_rows=1, n_cols=1)

    builder = PlateBuilder(dest, plate=plate, writer="zarr")
    data = np.random.rand(2, 32, 32).astype(np.float32)
    image = make_simple_image("test", {"c": 1.0, "y": 0.5, "x": 0.5})

    # Add well first
    builder.add_well(row="A", col="01", fields={"0": (image, [data])})

    # Try to write same well
    with pytest.raises(ValueError, match="already added via add_well"):
        builder.write_well(row="A", col="01", fields={"0": (image, [data])})


def test_plate_builder_prepare_empty(tmp_path: Path) -> None:
    """Test that PlateBuilder.prepare() raises error when no wells added."""
    from yaozarrs.write.v05 import PlateBuilder

    dest = tmp_path / "empty_prepare.zarr"
    plate, _ = make_simple_plate(n_rows=1, n_cols=1)

    builder = PlateBuilder(dest, plate=plate, writer="zarr")

    with pytest.raises(ValueError, match="No wells added"):
        builder.prepare()


def test_plate_builder_dataset_count_mismatch(tmp_path: Path) -> None:
    """Test that PlateBuilder raises error for dataset count mismatch."""
    from yaozarrs.write.v05 import PlateBuilder

    dest = tmp_path / "dataset_mismatch.zarr"
    plate, _ = make_simple_plate(n_rows=1, n_cols=1)

    builder = PlateBuilder(dest, plate=plate, writer="zarr")
    image = make_simple_image("test", {"c": 1.0, "y": 0.5, "x": 0.5})

    # Image expects 1 dataset but we provide 2
    data1 = np.random.rand(2, 32, 32).astype(np.float32)
    data2 = np.random.rand(2, 16, 16).astype(np.float32)

    with pytest.raises(ValueError, match="must match"):
        builder.add_well(row="A", col="01", fields={"0": (image, [data1, data2])})


def test_plate_builder_multiscale_error(tmp_path: Path) -> None:
    """Test that PlateBuilder raises error for multiple multiscales."""
    from yaozarrs.write.v05 import PlateBuilder

    dest = tmp_path / "multiscale_error.zarr"
    plate, _ = make_simple_plate(n_rows=1, n_cols=1)

    builder = PlateBuilder(dest, plate=plate, writer="zarr")
    data = np.random.rand(2, 32, 32).astype(np.float32)

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
        builder.add_well(row="A", col="01", fields={"0": (image, [data])})


@pytest.mark.parametrize("writer", WRITERS)
def test_plate_builder_well_metadata_generation(
    tmp_path: Path, writer: ZarrWriter
) -> None:
    """Test that PlateBuilder correctly generates well metadata."""
    from yaozarrs.write.v05 import PlateBuilder

    dest = tmp_path / "well_metadata.zarr"
    plate, _ = make_simple_plate(n_rows=1, n_cols=1)

    builder = PlateBuilder(dest, plate=plate, writer=writer)

    # Create well with multiple fields in non-sorted order
    fields = {}
    for fov in ["2", "0", "1"]:
        data = np.random.rand(2, 32, 32).astype(np.float32)
        image = make_simple_image(f"field_{fov}", {"c": 1.0, "y": 0.5, "x": 0.5})
        fields[fov] = (image, [data])

    builder.write_well(row="A", col="01", fields=fields)

    # Check well metadata - fields should be sorted
    well_meta = json.loads((dest / "A" / "01" / "zarr.json").read_bytes())
    images_meta = well_meta["attributes"]["ome"]["well"]["images"]
    assert len(images_meta) == 3
    assert images_meta[0]["path"] == "0"
    assert images_meta[1]["path"] == "1"
    assert images_meta[2]["path"] == "2"
    # acquisition=None is excluded from JSON by pydantic's exclude_none=True
    for img in images_meta:
        assert img.get("acquisition") is None


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate_with_different_chunks(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_plate with custom chunk settings."""
    from yaozarrs.write.v05 import write_plate

    dest = tmp_path / "plate_chunks.zarr"
    plate, images = make_simple_plate(n_rows=1, n_cols=1)

    write_plate(dest, images, plate=plate, chunks=(1, 32, 32), writer=writer)

    # Check chunk shape in array metadata
    arr_meta = json.loads((dest / "A" / "01" / "0" / "0" / "zarr.json").read_bytes())
    chunk_shape = arr_meta["chunk_grid"]["configuration"]["chunk_shape"]
    assert chunk_shape == [1, 32, 32]

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize(
    "compression",
    ["blosc-zstd", "blosc-lz4", "zstd", "none"],
)
def test_write_plate_compression(
    tmp_path: Path, writer: ZarrWriter, compression: CompressionName
) -> None:
    """Test write_plate with different compression options."""
    from yaozarrs.write.v05 import write_plate

    dest = tmp_path / f"plate_{compression}.zarr"
    plate, images = make_simple_plate(n_rows=1, n_cols=1)

    write_plate(dest, images, plate=plate, compression=compression, writer=writer)  # type: ignore

    # Verify compression was applied
    arr_meta = json.loads((dest / "A" / "01" / "0" / "0" / "zarr.json").read_bytes())
    codecs = arr_meta.get("codecs", [])

    if compression == "none":
        assert len(codecs) == 1  # Only bytes codec
    else:
        assert len(codecs) >= 2  # Bytes + compression codec

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate_overwrite(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_plate with overwrite=True."""
    from yaozarrs.write.v05 import write_plate

    dest = tmp_path / "plate_overwrite.zarr"
    plate, images = make_simple_plate(n_rows=1, n_cols=1)

    # Write first time
    write_plate(dest, images, plate=plate, writer=writer)
    first_write_time = (dest / "zarr.json").stat().st_mtime

    # Write again without overwrite - should raise error
    with pytest.raises(FileExistsError):
        write_plate(dest, images, plate=plate, writer=writer, overwrite=False)

    # Write again with overwrite - should succeed
    write_plate(dest, images, plate=plate, writer=writer, overwrite=True)
    second_write_time = (dest / "zarr.json").stat().st_mtime

    # Verify file was updated
    assert second_write_time >= first_write_time

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_plate_large_grid(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_plate with a larger plate grid (4x6)."""
    from yaozarrs.write.v05 import write_plate

    dest = tmp_path / "plate_4x6.zarr"
    plate, images = make_simple_plate(n_rows=4, n_cols=6)

    result = write_plate(dest, images, plate=plate, writer=writer)
    assert result == dest

    # Verify all 24 wells exist
    plate_meta = json.loads((dest / "zarr.json").read_bytes())
    assert len(plate_meta["attributes"]["ome"]["plate"]["wells"]) == 24

    # Spot check a few wells
    assert (dest / "A" / "01" / "0" / "zarr.json").exists()
    assert (dest / "D" / "06" / "0" / "zarr.json").exists()

    yaozarrs.validate_zarr_store(dest)


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
