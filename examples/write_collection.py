# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yaozarrs[write-tensorstore]",
#     "numpy",
# ]
#
# [tool.uv.sources]
# yaozarrs = { path = "../", editable = true }
# ///
"""Complex example using Bf2RawBuilder for multi-series collections."""

from __future__ import annotations

import numpy as np

from yaozarrs import v05
from yaozarrs.write.v05 import Bf2RawBuilder


def make_image(name: str) -> v05.Image:
    """Create a simple Image metadata object."""
    return v05.Image(
        multiscales=[
            v05.Multiscale(
                name=name,
                axes=[
                    v05.SpaceAxis(name="y", type="space", unit="micrometer"),
                    v05.SpaceAxis(name="x", type="space", unit="micrometer"),
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1.0, 1.0])
                        ],
                    )
                ],
            )
        ]
    )


# Initialize builder
builder = Bf2RawBuilder("collection.ome.zarr", overwrite=True)

# Write each series immediately with data
img1 = np.random.randint(0, 100, size=(256, 256), dtype=np.uint16)
builder.write_image("0", make_image("First Image"), img1)

img2 = np.random.randint(0, 100, size=(128, 128), dtype=np.uint16)
builder.write_image("1", make_image("Second Image"), img2)

img3 = np.random.randint(0, 100, size=(64, 64), dtype=np.uint16)
builder.write_image("2", make_image("Third Image"), img3)

print(f"✅ Wrote collection to {builder.root_path}")
