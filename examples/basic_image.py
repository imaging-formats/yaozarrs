# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yaozarrs",
#     "rich",
# ]
#
# [tool.uv.sources]
# yaozarrs = { path = "../", editable = true }
# ///
"""Basic example of creating OME-Zarr v0.5 image metadata with yaozarrs."""

from yaozarrs import v05

try:
    from rich import print
except ImportError:
    pass

image = v05.Image(
    multiscales=[
        v05.Multiscale(
            name="My Image",
            axes=[
                v05.Axis(name="time", type="time", unit="second"),
                v05.Axis(name="channel", type="channel"),
                v05.Axis(name="z", type="space", unit="micrometer"),
                v05.Axis(name="y", type="space", unit="micrometer"),
                v05.Axis(name="x", type="space", unit="micrometer"),
            ],
            datasets=[
                v05.Dataset(
                    path="0",
                    coordinateTransformations=[
                        v05.ScaleTransformation(scale=[1.0, 1.0, 1.0, 0.5, 0.5])
                    ],
                )
            ],
        )
    ],
)

print(image.model_dump_json(indent=2, exclude_unset=True))
