"""Benchmarks for constructing OME-Zarr models and converting between versions."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from yaozarrs import (
    ConversionWarning,
    DimSpec,
    convert_version,
    v04,
    v05,
    v06,
    validate_ome_json,
)

DATA = Path(__file__).parent.parent / "data"

DIMS = [
    DimSpec(name="t", size=10, scale=2.0, unit="second"),
    DimSpec(name="c", size=3),
    DimSpec(name="z", size=64, scale=0.5, unit="micrometer"),
    DimSpec(name="y", size=1024, scale=0.1, unit="micrometer"),
    DimSpec(name="x", size=1024, scale=0.1, unit="micrometer"),
]


@pytest.mark.parametrize("module", [v04, v05, v06], ids=["v04", "v05", "v06"])
def test_multiscale_from_dims(benchmark, module) -> None:
    """Build a 5D image with a 5-level pyramid from dimension specs."""

    def build():
        return module.Image(multiscales=[module.Multiscale.from_dims(DIMS, n_levels=5)])

    benchmark(build)


def test_construct_v05_group(benchmark) -> None:
    """Build a v0.5 zarr.json document from explicit models (as in the README)."""

    def build() -> str:
        scale = v05.Multiscale(
            name="scale0",
            axes=[
                v05.SpaceAxis(name="y", type="space"),
                v05.SpaceAxis(name="x", type="space"),
            ],
            datasets=[
                v05.Dataset(
                    path=str(i),
                    coordinateTransformations=[
                        v05.ScaleTransformation(scale=[2.0**i, 2.0**i])
                    ],
                )
                for i in range(5)
            ],
        )
        img = v05.Image(multiscales=[scale])
        group = v05.OMEZarrGroupJSON(attributes={"ome": img})
        return group.model_dump_json(exclude_unset=True)

    benchmark(build)


CONVERSIONS = [
    ("v04/2551.zarr/A/2/2/.zattrs", "0.6"),
    ("v04/2551.zarr/.zattrs", "0.5"),
    ("v05/9822152.zarr/zarr.json", "0.6"),
    ("v05/9822152.zarr/zarr.json", "0.4"),
    ("v05/76-45.ome.zarr/zarr.json", "0.6"),
    ("v06/image_transforms.ome.zarr/zarr.json", "0.5"),
    ("v06/plate.ome.zarr/zarr.json", "0.4"),
]


@pytest.mark.parametrize(
    ("doc", "version"),
    CONVERSIONS,
    ids=[f"{d.split('/')[0]}-{d.split('/')[1]}-to-{v}" for d, v in CONVERSIONS],
)
def test_convert_version(benchmark, doc: str, version: Any) -> None:
    """Convert a validated model to another version of the spec."""
    obj = validate_ome_json((DATA / doc).read_text())

    def convert():
        # metadata that the target can't express is dropped with a warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConversionWarning)
            return convert_version(obj, version)

    benchmark(convert)
