"""A zarr.json document found in any ome-zarr v0.5 group.

https://ngff.openmicroscopy.org/0.5/

Here are ALL the possible zarr.json documents you might encounter with OME metadata:

  1. Image Group

  Array Discovery: Self-contained, resolution levels in datasets paths
  my_image/
  ├── zarr.json          # ← Contains THIS metadata
  ├── 0/                 # Resolution level 0 Array
  ├── 1/                 # Resolution level 1 Array
  └── labels/            # Optional labels (see Labels Group below)
      └── ...

  {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
      "ome": {
        "version": "0.5",
        "multiscales": [
          {
            "name": "example_image",
            "axes": [
              {"name": "t", "type": "time", "unit": "millisecond"},
              {"name": "c", "type": "channel"},
              {"name": "z", "type": "space", "unit": "micrometer"},
              {"name": "y", "type": "space", "unit": "micrometer"},
              {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": [
              {
                "path": "0",
                "coordinateTransformations": [
                  {"type": "scale", "scale": [1.0, 1.0, 0.5, 0.5, 0.5]}
                ]
              }
            ]
          }
        ],
        "omero": {
          "channels": [
            {
              "label": "DAPI",
              "color": "0000FF",
              "window": {"start": 0, "end": 255}
            }
          ]
        }
      }
    }
  }

  2. Plate Group

  Array Discovery: Wells listed explicitly in wells array
  my_plate/
  ├── zarr.json               # ← Contains plate metadata
  ├── A/                      # Row A
  │   ├── 1/                  # Column 1
  │   │   ├── zarr.json       # Well Group metadata (see below)
  │   │   ├── 0/              # Field 0
  │   │   │   ├── 0/          # Resolution level 0 Array
  │   │   │   ├── ...
  │   │   │   └── zarr.json   # multiscales metadata for field 0 image
  │   │   └── 1/              # Field 1
  │   └── 2/
  └── B/
      └── 1/

  {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
      "ome": {
        "version": "0.5",
        // "bioformats2raw.layout": 3, // MAY be present if came from bioformats2raw
        "plate": {
          "name": "experiment_001",
          "columns": [
            {"name": "1"},
            {"name": "2"}
          ],
          "rows": [
            {"name": "A"},
            {"name": "B"}
          ],
          "wells": [
            {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
            {"path": "B/1", "rowIndex": 1, "columnIndex": 0}
          ],
          "acquisitions": [
            {"id": 1, "name": "initial_scan"}
          ]
        }
      }
    }
  }

  3. Well Group

  Array Discovery: Images listed explicitly in images array
  my_plate/A/1/
  ├── zarr.json          # ← Contains THIS metadata
  ├── 0/                 # Field/image 0
  │   ├── zarr.json      # Image metadata
  │   ├── 0/             # Resolution level
  │   └── 1/
  └── 1/                 # Field/image 1
      └── ...

  {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
      "ome": {
        "version": "0.5",
        "well": {
          "images": [
            {
              "path": "0",
              "acquisition": 1
            },
            {
              "path": "1",
              "acquisition": 1
            }
          ]
        }
      }
    }
  }

  4. Labels Group

  Array Discovery: Must explore filesystem for labels/ directories
  my_image/
  ├── zarr.json          # Image metadata (see above)
  ├── 0/
  └── labels/
      ├── zarr.json      # ← Contains THIS metadata
      ├── cell_seg/
      │   ├── zarr.json  # Label Image Group metadata (see below)
      │   └── 0/
      └── nuclei_seg/
          └── ...

  {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
      "ome": {
        "version": "0.5",
        "labels": [
          "cell_segmentation",
          "nuclei_segmentation"
        ]
      }
    }
  }

  5. Label Image Group

  This is a special case/subclass of the Image Group above, distinguished by
  the presence of the "image-label" key in the metadata.

  Array Discovery: Listed in parent labels group's labels array
  my_image/labels/cell_segmentation/
  ├── zarr.json          # ← Contains THIS metadata
  ├── 0/                 # Resolution level 0 Array
  └── 1/                 # Resolution level 1 Array

  {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
      "ome": {
        "version": "0.5",
        "multiscales": [
          {
            "axes": [
              {"name": "z", "type": "space", "unit": "micrometer"},
              {"name": "y", "type": "space", "unit": "micrometer"},
              {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": [
              {
                "path": "0",
                "coordinateTransformations": [
                  {"type": "scale", "scale": [0.5, 0.5, 0.5]}
                ]
              }
            ]
          }
        ],
        "image-label": {
          "version": "0.5",
          "colors": [
            {
              "label-value": 1,
              "rgba": [255, 0, 0, 128]
            }
          ],
          "properties": [
            {
              "label-value": 1,
              "cell_type": "neuron",
              "area": 1250.5
            }
          ],
          "source": {
            "image": "../../"
          }
        }
      }
    }
  }

  6. Series Collection Group

  Array Discovery: Images listed explicitly in series array
  converted_multiseries/
  ├── zarr.json          # ← Contains THIS metadata
  ├── 0/                 # First image series
  │   ├── zarr.json      # Image metadata
  │   ├── 0/
  │   └── 1/
  ├── 1/                 # Second image series
  │   └── ...
  └── OME/
      └── METADATA.ome.xml

  {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
      "ome": {
        "version": "0.5",
        "series": ["0", "1", "2"]
      }
    }
  }

  7. Bioformats2raw Layout Group

  Array Discovery: Must explore numbered directories (0/, 1/, 2/, etc.)
  converted_file/
  ├── zarr.json          # ← Contains THIS metadata
  ├── 0/                 # First image (by convention)
  │   └── zarr.json      # Image metadata
  ├── 1/                 # Second image (by convention)
  │   └── zarr.json
  ├── 2/                 # Third image (by convention)
  │   └── zarr.json
  └── OME/
      └── METADATA.ome.xml

  {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {
      "ome": {
        "version": "0.5",
        "bioformats2raw.layout": 3
      }
    }
  }

"""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Discriminator, Tag

from yaozarrs._base import ZarrGroupModel, _BaseModel
from yaozarrs.v05._bf2raw import Bf2Raw

from ._bf2raw import Series
from ._image import Image
from ._labels import LabelImage, LabelsGroup
from ._plate import Plate, Well


def _discriminate_ome_v05_metadata(v: Any) -> str | None:
    if isinstance(v, dict):
        if "image-label" in v:
            return "label-image"
        if "multiscales" in v:
            return "image"
        if "plate" in v:
            return "plate"
        if "bioformats2raw.layout" in v or "bioformats2raw_layout" in v:
            return "bf2raw"
        if "well" in v:
            return "well"
        if "labels" in v:
            return "labels-group"
        if "series" in v:
            return "series"
    elif isinstance(v, BaseModel):
        if isinstance(v, LabelImage):
            return "label-image"
        if isinstance(v, Image):
            return "image"
        if isinstance(v, Plate):
            return "plate"
        if isinstance(v, Bf2Raw):
            return "bf2raw"
        if isinstance(v, Well):
            return "well"
        if isinstance(v, LabelsGroup):
            return "labels-group"
        if isinstance(v, Series):  # pragma: no cover
            return "series"
    return None


OMEMetadata: TypeAlias = Annotated[
    (
        Annotated[LabelImage, Tag("label-image")]
        | Annotated[Image, Tag("image")]
        | Annotated[Plate, Tag("plate")]
        | Annotated[Bf2Raw, Tag("bf2raw")]
        | Annotated[Well, Tag("well")]
        | Annotated[LabelsGroup, Tag("labels-group")]
        | Annotated[Series, Tag("series")]
    ),
    Discriminator(_discriminate_ome_v05_metadata),
]
"""Union type for anything that can live in the "ome" key of a v0.5 `zarr.json` file."""


class OMEAttributes(_BaseModel):
    """The attributes field of a `zarr.json` document in an ome-zarr group."""

    ome: OMEMetadata


class OMEZarrGroupJSON(ZarrGroupModel):
    """A `zarr.json` document found in any ome-zarr group."""

    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"
    attributes: OMEAttributes
