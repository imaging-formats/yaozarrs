# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yaozarrs[write-zarr]",
#     "numpy",
# ]
#
# [tool.uv.sources]
# yaozarrs = { path = "../", editable = true }
# ///
"""Simple example of writing an OME-Zarr v0.5 image with data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yaozarrs import v05
from yaozarrs.write.v05 import write_image

# Create sample data (CZYX format)
data = np.random.randint(0, 255, size=(2, 10, 512, 512), dtype=np.uint8)
print(f"Created data with shape: {data.shape} ({data.dtype})")

# Create OME-Zarr Image metadata
image = v05.Image(
    multiscales=[
        v05.Multiscale(
            name="Sample Image",
            axes=[
                v05.ChannelAxis(name="channel", type="channel"),
                v05.SpaceAxis(name="z", type="space", unit="micrometer"),
                v05.SpaceAxis(name="y", type="space", unit="micrometer"),
                v05.SpaceAxis(name="x", type="space", unit="micrometer"),
            ],
            datasets=[
                v05.Dataset(
                    path="0",
                    coordinateTransformations=[
                        v05.ScaleTransformation(scale=[1.0, 1.0, 0.5, 0.5])
                    ],
                )
            ],
        )
    ],
)

dest = Path("example_image.ome.zarr")

# Write the image with data
written_path = write_image(
    dest,
    image,
    data,
    chunks=(1, 1, 256, 256),  # Chunk along spatial dimensions
    compression="blosc-zstd",
    overwrite=True,
)

print(f"\n✅ Image written to: {written_path}")
print("   Files created:")
for f in sorted(written_path.rglob("*")):
    if f.is_file():
        print(f"   - {f.relative_to(written_path)}")
