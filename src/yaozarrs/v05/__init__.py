"""OME-NGFF 0.5 metadata models.

This module provides Pydantic models for validating and working with
OME-NGFF (Next-Generation File Format) version 0.5 metadata.

## Core Concepts

OME-NGFF is a specification *on top of* the [Zarr
format](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) that
standardizes where metadata for imaging data is stored and how it is structured.
OME-Zarr is "just zarr", with additional conventions for what fields you will find
inside `zarr.json` files, and how the directory hierarchy is structured.

This module provides models for all the different types of metadata you'll encounter.

## Top-Level Models

These models are the primary entry points for most use cases.  They represent
complete `zarr.json` documents found in zarr groups.

- **[`Image`][yaozarrs.v05.Image]**: An Image with up to 5 dimensions, possibly with
  multi-resolution (pyramid) data.
  ***This is the most common starting point for single 5D images.***

    - **[`LabelImage`][yaozarrs.v05.LabelImage]**: This is just an
    [`Image`][yaozarrs.v05.Image] with an additional `image_label` field.
    ***Use this to for 5D images that are accompanied by masks/labels.***

- **[`Plate`][yaozarrs.v05.Plate]**: High-content screening multi-well plate layout
  organizing wells in a grid.
  ***Use this for multi-well experiments..***

- **[`Bf2Raw`][yaozarrs.v05.Bf2Raw]**: This is a transitional spec, but it is currently
  the only way to represent multi-position image collections.  It also has a recommended
  place for OME-XML metadata. ***Use this for images that contain any dimensions
  beyond TCZYX.*** (until a better [Collections](https://github.com/ome/ngff/issues/31)
  spec is available)


## Secondary Group Models

These models represent complete `zarr.json` documents found in zarr groups that are
typically nested inside the directory structure of the top-level models.

- **[`Well`][yaozarrs.v05.Well]**: Collection of fields-of-view within a single
  well. *Found inside plates.*

- **[`LabelsGroup`][yaozarrs.v05.LabelsGroup]**: Annotations/labels for an Image.
  *Usually found in a subdirectory of an Image group.*

- **[`Series`][yaozarrs.v05.Series]**: Represents `OME/zarr.json` files found inside of
  [`Bf2Raw`][yaozarrs.v05.Bf2Raw] collections.

## Quick Start

Reading an image's metadata:
```python
from yaozarrs.v05 import Image

# Parse from zarr.json content
zarr_json_content = Path("path/to/your/zarr/OME/zarr.json").read_text()
image = Image.model_validate_json(zarr_json_content)

# Access coordinate info
print(image.multiscales[0].axes)  # Dimension axes
print(image.multiscales[0].datasets)  # Resolution levels
```

Working with plate data:
```python
from yaozarrs.v05 import Plate

plate_zarr_json = Path("path/to/your/zarr/OME/zarr.json").read_text()
plate = Plate.model_validate_json(plate_zarr_json)
print(f"Plate has {len(plate.plate.wells)} wells")
for well in plate.plate.wells:
    print(f"Well at {well.path}")
```

## Reference

Specification: <https://ngff.openmicroscopy.org/0.5/>

Schema source: <https://github.com/ome/ngff/tree/8cbba216e37407bd2d4bd5c7128ab13bd0a6404e>

--------------------------------------------------------------------------------
"""

from yaozarrs._omero import Omero, OmeroChannel, OmeroRenderingDefs, OmeroWindow

from ._bf2raw import Bf2Raw
from ._image import (
    ChannelAxis,
    CustomAxis,
    Dataset,
    Image,
    Multiscale,
    ScaleTransformation,
    SpaceAxis,
    TimeAxis,
    TranslationTransformation,
)
from ._label import (
    ImageLabel,
    LabelColor,
    LabelImage,
    LabelProperty,
    LabelsGroup,
    LabelSource,
)
from ._plate import Acquisition, Column, Plate, PlateDef, PlateWell, Row
from ._series import Series
from ._well import FieldOfView, Well, WellDef
from ._zarr_json import OMEAttributes, OMEMetadata, OMEZarrGroupJSON

__all__ = [
    "Acquisition",
    "Bf2Raw",
    "ChannelAxis",
    "Column",
    "CustomAxis",
    "Dataset",
    "FieldOfView",
    "Image",
    "ImageLabel",
    "LabelColor",
    "LabelImage",
    "LabelProperty",
    "LabelSource",
    "LabelsGroup",
    "Multiscale",
    "OMEAttributes",
    "OMEMetadata",
    "OMEZarrGroupJSON",
    "Omero",
    "OmeroChannel",
    "OmeroRenderingDefs",
    "OmeroWindow",
    "Plate",
    "PlateDef",
    "PlateWell",
    "Row",
    "ScaleTransformation",
    "Series",
    "Series",
    "SpaceAxis",
    "TimeAxis",
    "TranslationTransformation",
    "Well",
    "WellDef",
]
