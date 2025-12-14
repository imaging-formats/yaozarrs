# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yaozarrs[write-zarr,write-tensorstore]",
# ]
#
# [tool.uv.sources]
# yaozarrs = { path = "../", editable = true }
# ///
"""Benchmark different zarr writers for OME-Zarr v0.5."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yaozarrs.v05 import (
    ChannelAxis,
    Dataset,
    Image,
    Multiscale,
    ScaleTransformation,
    SpaceAxis,
    TimeAxis,
)
from yaozarrs.write.v05 import write_image

if TYPE_CHECKING:
    from yaozarrs.write.v05._write import ZarrWriter

# Test configuration
SHAPE = (10, 3, 25, 1024, 1024)  # TCZYX
DTYPE = np.uint16
ITERATIONS = 6 if len(sys.argv) < 2 else int(sys.argv[1])
WARMUP = 1  # Number of warmup iterations to discard

# Available writers
WRITERS: list[ZarrWriter] = ["zarr", "tensorstore"]

print(f"Testing writers: {WRITERS}")
print(f"Shape: {SHAPE}")
print(f"Dtype: {DTYPE}")
print(f"Size: {np.prod(SHAPE) * np.dtype(DTYPE).itemsize / 1024**3:.2f} GB")
print(f"Iterations: {ITERATIONS}")
print()

# Pre-allocate data in RAM
print("Allocating data in RAM...")
data = np.random.randint(0, 65535, size=SHAPE, dtype=DTYPE)
print(f"Data allocated: {data.nbytes / 1024**3:.2f} GB")
print()

# Create metadata
image = Image(
    multiscales=[
        Multiscale(
            axes=[
                TimeAxis(name="t"),
                ChannelAxis(name="c"),
                SpaceAxis(name="z", unit="micrometer"),
                SpaceAxis(name="y", unit="micrometer"),
                SpaceAxis(name="x", unit="micrometer"),
            ],
            datasets=[
                Dataset(
                    path="0",
                    coordinateTransformations=[
                        ScaleTransformation(scale=[1.0, 1.0, 1.0, 1.0, 1.0])
                    ],
                )
            ],
            name="test",
        )
    ]
)

# Benchmark each writer
results = {}

for writer in WRITERS:
    times = []

    # Warmup iterations
    for i in range(WARMUP):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / f"test_{writer}.zarr"
            write_image(dest, image, [data], writer=writer)
            print(f"{writer} warmup {i + 1}/{WARMUP}")

    # Actual benchmark iterations
    for i in range(ITERATIONS):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / f"test_{writer}.zarr"

            # Time the write operation
            start = time.perf_counter()
            write_image(dest, image, [data], writer=writer)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            print(f"{writer} iteration {i + 1}/{ITERATIONS}: {elapsed:.3f}s")

    results[writer] = {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }
    print()

# Print summary
print("=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"{'Writer':<15} {'Mean (s)':<12} {'Std (s)':<12} {'MB/s':<12}")
print("-" * 60)

data_size_mb = data.nbytes / 1024**2

for writer, stats in sorted(results.items(), key=lambda x: x[1]["mean"]):
    throughput = data_size_mb / stats["mean"]
    print(
        f"{writer:<15} {stats['mean']:>10.3f}  "
        f"{stats['std']:>10.3f}  {throughput:>10.1f}"
    )

print("=" * 60)
