"""Tests for OME-Zarr v0.5 write functionality."""

from __future__ import annotations

import doctest
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import yaozarrs
from yaozarrs import v05
from yaozarrs.v05 import _write

if TYPE_CHECKING:
    from typing import Literal

    ZarrWriter = Literal["zarr", "zarrs", "tensorstore", "auto"]

try:
    import numpy as np
except ImportError:
    pytest.skip("NumPy not available", allow_module_level=True)

WRITERS: list[ZarrWriter] = []
if importlib.util.find_spec("zarr") is not None:
    WRITERS.append("zarr")
if importlib.util.find_spec("zarrs") is not None:
    WRITERS.append("zarrs")
if importlib.util.find_spec("tensorstore") is not None:
    WRITERS.append("tensorstore")

if not WRITERS:
    pytest.skip(
        "No supported Zarr writer (zarr, zarrs, or tensorstore) found",
        allow_module_level=True,
    )


def make_simple_image(name: str = "test", ndim: int = 3) -> v05.Image:
    """Create a simple v05.Image model for testing."""
    if ndim == 2:
        axes = [
            v05.SpaceAxis(name="y", unit="micrometer"),
            v05.SpaceAxis(name="x", unit="micrometer"),
        ]
        scale = [0.5, 0.5]
    elif ndim == 3:
        axes = [
            v05.ChannelAxis(name="c"),
            v05.SpaceAxis(name="y", unit="micrometer"),
            v05.SpaceAxis(name="x", unit="micrometer"),
        ]
        scale = [1.0, 0.5, 0.5]
    elif ndim == 4:
        axes = [
            v05.TimeAxis(name="t", unit="millisecond"),
            v05.ChannelAxis(name="c"),
            v05.SpaceAxis(name="y", unit="micrometer"),
            v05.SpaceAxis(name="x", unit="micrometer"),
        ]
        scale = [100.0, 1.0, 0.5, 0.5]
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")

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
def test_write_image_basic(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test basic write_image with a single resolution level."""
    dest = tmp_path / "test.zarr"
    data = np.random.rand(2, 64, 64).astype(np.float32)
    image = make_simple_image("basic_test", ndim=3)

    result = v05.write_image(dest, [data], image, writer=writer)

    assert result == dest
    assert dest.exists()
    assert (dest / "zarr.json").exists()
    assert (dest / "0").exists()

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_2d(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with 2D data."""
    dest = tmp_path / "test2d.zarr"
    data = np.random.rand(64, 64).astype(np.float32)
    image = make_simple_image("test2d", ndim=2)

    result = v05.write_image(dest, [data], image, writer=writer)

    assert result == dest
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_4d(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with 4D data (TCYX)."""
    dest = tmp_path / "test4d.zarr"
    data = np.random.rand(5, 2, 32, 32).astype(np.float32)
    image = make_simple_image("test4d", ndim=4)

    result = v05.write_image(dest, [data], image, writer=writer)

    assert result == dest
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_multiscale(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with multiple resolution levels."""
    dest = tmp_path / "multiscale.zarr"
    datasets, image = make_multiscale_image("pyramid", n_levels=3)

    result = v05.write_image(dest, datasets, image, writer=writer)

    assert result == dest
    assert (dest / "0").exists()
    assert (dest / "1").exists()
    assert (dest / "2").exists()

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_custom_chunks(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with custom chunk shape."""
    dest = tmp_path / "chunked.zarr"
    data = np.random.rand(2, 64, 64).astype(np.float32)
    image = make_simple_image("chunked", ndim=3)
    custom_chunks = (1, 32, 32)

    v05.write_image(dest, [data], image, chunks=custom_chunks, writer=writer)

    yaozarrs.validate_zarr_store(dest)

    # Verify chunks in metadata
    with open(dest / "0" / "zarr.json") as fh:
        arr_meta = json.load(fh)
    chunk_shape = arr_meta["chunk_grid"]["configuration"]["chunk_shape"]
    assert chunk_shape == list(custom_chunks)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_auto_chunks(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with auto chunk calculation."""
    dest = tmp_path / "auto_chunked.zarr"
    data = np.random.rand(2, 256, 256).astype(np.float32)
    image = make_simple_image("auto_chunked", ndim=3)

    v05.write_image(dest, [data], image, chunks="auto", writer=writer)

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_no_chunks(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with chunks=None (single chunk)."""
    dest = tmp_path / "single_chunk.zarr"
    data = np.random.rand(2, 32, 32).astype(np.float32)
    image = make_simple_image("single_chunk", ndim=3)

    v05.write_image(dest, [data], image, chunks=None, writer=writer)

    yaozarrs.validate_zarr_store(dest)

    # Verify chunks match full array shape
    with open(dest / "0" / "zarr.json") as fh:
        arr_meta = json.load(fh)
    chunk_shape = arr_meta["chunk_grid"]["configuration"]["chunk_shape"]
    assert chunk_shape == list(data.shape)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_metadata_correct(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test that written metadata matches input Image model."""
    dest = tmp_path / "metadata.zarr"
    data = np.random.rand(2, 64, 64).astype(np.float32)
    image = make_simple_image("metadata_test", ndim=3)

    v05.write_image(dest, [data], image, writer=writer)

    with open(dest / "zarr.json") as fh:
        meta = json.load(fh)

    ome = meta["attributes"]["ome"]
    assert ome["multiscales"][0]["name"] == "metadata_test"

    axes = ome["multiscales"][0]["axes"]
    assert len(axes) == 3
    assert axes[0]["name"] == "c"
    assert axes[0]["type"] == "channel"
    assert axes[1]["name"] == "y"
    assert axes[1]["type"] == "space"
    assert axes[2]["name"] == "x"
    assert axes[2]["type"] == "space"

    transforms = ome["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    assert transforms[0]["type"] == "scale"
    assert transforms[0]["scale"] == [1.0, 0.5, 0.5]

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
        v05.write_image(dest, data, image)


# =============================================================================
# write_bioformats2raw tests
# =============================================================================


@pytest.mark.parametrize("writer", WRITERS)
def test_write_bioformats2raw_single_series(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_bioformats2raw with a single series."""
    dest = tmp_path / "bf2raw.zarr"
    data = np.random.rand(2, 64, 64).astype(np.float32)
    image = make_simple_image("series_0", ndim=3)

    images = {"0": ([data], image)}

    result = v05.write_bioformats2raw(dest, images, writer=writer)

    assert result == dest
    assert dest.exists()

    # Check root has bioformats2raw.layout
    with open(dest / "zarr.json") as fh:
        root_meta = json.load(fh)
    assert root_meta["attributes"]["ome"]["bioformats2raw.layout"] == 3

    # Check OME directory with series metadata
    ome_path = dest / "OME"
    assert ome_path.exists()
    with open(ome_path / "zarr.json") as fh:
        ome_meta = json.load(fh)
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
        image = make_simple_image(f"series_{i}", ndim=3)
        images[str(i)] = ([data], image)

    result = v05.write_bioformats2raw(dest, images, writer=writer)

    assert result == dest

    # Check root
    with open(dest / "zarr.json") as fh:
        root_meta = json.load(fh)
    assert root_meta["attributes"]["ome"]["bioformats2raw.layout"] == 3

    # Check OME directory
    with open(dest / "OME" / "zarr.json") as fh:
        ome_meta = json.load(fh)
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
    image = make_simple_image("with_xml", ndim=3)

    ome_xml = '<?xml version="1.0"?><OME xmlns="test">test content</OME>'
    images = {"0": ([data], image)}

    v05.write_bioformats2raw(dest, images, ome_xml=ome_xml, writer=writer)

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
    images = {"0": (datasets, image)}

    v05.write_bioformats2raw(dest, images, writer=writer)

    # Check pyramid levels exist
    assert (dest / "0" / "0").exists()
    assert (dest / "0" / "1").exists()

    yaozarrs.validate_zarr_store(str(dest / "0"))


# =============================================================================
# Edge cases and error handling
# =============================================================================


def test_write_image_invalid_writer(tmp_path: Path) -> None:
    """Test that invalid writer raises error."""
    dest = tmp_path / "invalid.zarr"
    data = np.random.rand(2, 64, 64).astype(np.float32)
    image = make_simple_image("invalid", ndim=3)

    with pytest.raises(ValueError, match="Unknown writer"):
        v05.write_image(dest, [data], image, writer="invalid")  # type: ignore


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

    v05.write_image(dest, [data], image, writer=writer)

    # Check omero metadata was written
    with open(dest / "zarr.json") as fh:
        meta = json.load(fh)

    ome = meta["attributes"]["ome"]
    assert "omero" in ome
    assert len(ome["omero"]["channels"]) == 3
    assert ome["omero"]["channels"][0]["color"] == "FF0000"

    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize("writer", WRITERS)
def test_write_image_different_dtypes(tmp_path: Path, writer: ZarrWriter) -> None:
    """Test write_image with various data types."""
    for dtype in [np.uint8, np.uint16, np.float32, np.float64]:
        dest = tmp_path / f"dtype_{dtype.__name__}.zarr"
        data = np.random.rand(2, 32, 32).astype(dtype)
        if np.issubdtype(dtype, np.integer):
            data = (data * 255).astype(dtype)
        image = make_simple_image(f"dtype_{dtype.__name__}", ndim=3)

        v05.write_image(dest, [data], image, writer=writer)

        yaozarrs.validate_zarr_store(dest)


def test_write_doctests(tmp_path: Path) -> None:
    """Run doctests from the write module."""
    # Run doctests with extraglobs
    results = doctest.testmod(
        _write,
        globs={"tmpdir": str(tmp_path), "Path": Path},
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )

    # Check if any tests failed
    if results.failed > 0:
        pytest.fail(
            f"Doctest failed: {results.failed} failures of {results.attempted} tests"
        )
