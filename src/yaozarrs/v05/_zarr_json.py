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
  ├── zarr.json          # ← Contains THIS metadata
  ├── A/
  │   ├── 1/
  │   │   ├── zarr.json  # Well Group metadata (see below)
  │   │   ├── 0/         # Field 0
  │   │   └── 1/         # Field 1
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

from typing import Literal, TypeAlias

from yaozarrs._base import ZarrGroupModel, _BaseModel
from yaozarrs.v05._bf2raw import Bf2Raw
from yaozarrs.v05._ome import OME

from ._image import Image
from ._label import LabelImage, LabelsGroup
from ._plate import Plate
from ._series import Series
from ._well import Well

# LabelImage must come before Image because it's a subclass
# could also use pydantic.Discriminator, but this is simpler
OMEMetadata: TypeAlias = (
    LabelImage | Image | Plate | Well | LabelsGroup | OME | Series | Bf2Raw
)
"""Anything that can live in the "ome" key of a v0.5 ome-zarr file."""


class OMEAttributes(_BaseModel):
    """The attributes field of a zarr.json document in an ome-zarr group."""

    ome: OMEMetadata


class OMEZarrGroupJSON(ZarrGroupModel):
    """A zarr.json document found in any ome-zarr group."""

    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"
    attributes: OMEAttributes
