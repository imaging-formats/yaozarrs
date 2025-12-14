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
builder2 = Bf2RawBuilder(
    "prepared_collection.ome.zarr", chunks=(128, 128), overwrite=True
)

# Register all series first (without data)
shape = (512, 512)
dtype = "uint8"
builder2.add_series("0", make_image("Series A"), (shape, dtype))
builder2.add_series("1", make_image("Series B"), (shape, dtype))
builder2.add_series("2", make_image("Series C"), (shape, dtype))

# Prepare creates the structure with empty arrays
path, arrays = builder2.prepare()

# Now fill in the data yourself
arrays["0/0"][:] = np.random.randint(0, 255, size=shape, dtype=dtype)
arrays["1/0"][:] = np.random.randint(0, 255, size=shape, dtype=dtype)
arrays["2/0"][:] = np.random.randint(0, 255, size=shape, dtype=dtype)

print(f"✅ Prepared and filled collection at: {path}")
