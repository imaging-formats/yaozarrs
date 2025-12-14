# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yaozarrs",
#     "rich",
# ]
#
# [tool.uv.sources]
# yaozarrs = { path = "../", editable = true }
# ///
"""Basic example of creating OME-Zarr v0.5 plate metadata with yaozarrs."""

from yaozarrs import v05

try:
    from rich import print
except ImportError:
    pass

plate = v05.Plate(
    plate=v05.PlateDef(
        name="My Plate",
        rows=[
            v05.Row(name="A"),
            v05.Row(name="B"),
            v05.Row(name="C"),
        ],
        columns=[
            v05.Column(name="1"),
            v05.Column(name="2"),
            v05.Column(name="3"),
        ],
        wells=[
            v05.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            v05.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
            v05.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
            v05.PlateWell(path="B/2", rowIndex=1, columnIndex=1),
            v05.PlateWell(path="C/3", rowIndex=2, columnIndex=2),
        ],
        acquisitions=[
            v05.Acquisition(
                id=0,
                name="Initial Scan",
                description="First acquisition run",
            )
        ],
        field_count=2,
    ),
)

print(plate.model_dump_json(indent=2, exclude_unset=True))
