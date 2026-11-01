"""Benchmarks for validating OME-Zarr metadata documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from yaozarrs import v05, validate_ome_json, validate_ome_object

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

DATA = Path(__file__).parent.parent / "data"

# Representative metadata documents of every version and kind (relative to DATA)
DOCUMENTS = [
    "v04/2551.zarr/.zattrs",  # plate
    "v04/2551.zarr/A/2/2/.zattrs",  # image
    "v05/76-45.ome.zarr/zarr.json",  # large plate (384 wells)
    "v05/9822152.zarr/zarr.json",  # image, 11 resolution levels
    "v05/6001240_labels.zarr/labels/0/zarr.json",  # label image
    "v05/EM04534_04_dU_SBF_8bits_aligned_z4-136_7-7-50nm_v3.zarr/zarr.json",  # bf2raw
    "v06/image_transforms.ome.zarr/zarr.json",  # several coordinate systems
    "v06/plate.ome.zarr/zarr.json",
    "v06/scene.ome.zarr/zarr.json",
]
# The largest document of each version
LARGEST = [
    "v04/2551.zarr/.zattrs",
    "v05/76-45.ome.zarr/zarr.json",
    "v06/scene.ome.zarr/zarr.json",
]
TEXTS = {doc: (DATA / doc).read_text() for doc in DOCUMENTS}


@pytest.mark.parametrize("doc", DOCUMENTS)
def test_validate_ome_json(benchmark: BenchmarkFixture, doc: str) -> None:
    """Validate a JSON document, detecting its version and kind."""
    benchmark(validate_ome_json, TEXTS[doc])


@pytest.mark.parametrize("doc", LARGEST)
def test_validate_ome_object(benchmark: BenchmarkFixture, doc: str) -> None:
    """Validate an already-parsed python object, detecting its version and kind."""
    obj = json.loads(TEXTS[doc])
    benchmark(validate_ome_object, obj)


def test_validate_ome_json_with_cls(benchmark: BenchmarkFixture) -> None:
    """Validate a JSON document against an explicit model."""
    text = TEXTS["v05/76-45.ome.zarr/zarr.json"]
    benchmark(validate_ome_json, text, v05.OMEZarrGroupJSON)


@pytest.mark.parametrize("doc", LARGEST)
def test_model_dump_json(benchmark: BenchmarkFixture, doc: str) -> None:
    """Serialize a validated model back to JSON."""
    obj = validate_ome_json(TEXTS[doc])
    benchmark(obj.model_dump_json, exclude_unset=True)
