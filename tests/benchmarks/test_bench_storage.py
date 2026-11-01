"""Benchmarks for validating complete (local) OME-Zarr stores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from yaozarrs import open_group, validate_ome_uri, validate_zarr_store

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytest_codspeed import BenchmarkFixture

    StoreFactory = Callable[[str], Path]

pytest.importorskip("fsspec")

# A representative selection of the test data stores
STORES = [
    "v05/6001240_labels.zarr",  # image with labels
    "v06/image_with_labels.ome.zarr",
    "v06/plate.ome.zarr",
    "v06/bf2raw.ome.zarr",
    "v06/scene.ome.zarr",
]


@pytest.mark.parametrize("store", STORES)
def test_validate_zarr_store(
    benchmark: BenchmarkFixture, ome_store: StoreFactory, store: str
) -> None:
    """Validate both metadata and structure of a whole hierarchy."""
    path = str(ome_store(store))
    benchmark(validate_zarr_store, path)


# (validate_ome_uri is mostly validate_ome_json on the root metadata)
def test_validate_ome_uri(benchmark: BenchmarkFixture, ome_store: StoreFactory) -> None:
    """Load and validate the root group metadata only."""
    path = str(ome_store("v06/plate.ome.zarr"))
    benchmark(validate_ome_uri, path)


@pytest.mark.parametrize("store", ["v05/6001240_labels.zarr", "v06/plate.ome.zarr"])
def test_open_group(
    benchmark: BenchmarkFixture, ome_store: StoreFactory, store: str
) -> None:
    """Open a store and read its OME metadata."""
    path = str(ome_store(store))

    def open_and_read() -> Any:
        return open_group(path).ome_metadata()

    benchmark(open_and_read)
