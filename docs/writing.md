---
icon: material/content-save
title: Writing OME-Zarr
---

# Writing OME-Zarr

[`yaozarrs.write.v05`][] and [`yaozarrs.write.v06`][] write complete OME-Zarr
stores (metadata *and* array data) to a local directory.  They have the same
functions: use the one that matches the version of your metadata models.  v0.6
adds [`write_scene`][yaozarrs.write.v06.write_scene] and
[`SceneBuilder`][yaozarrs.write.v06.SceneBuilder], for
[scenes](ome_zarr_guide/index.md#working-with-scenes).

Writing needs an array backend, see [Array writer backends](#array-writer-backends).

## The general pattern

1. Build the metadata model that matches your data (see the [Guide to
   OME-Zarr](ome_zarr_guide/index.md)): an `Image`, a `Plate`, a `Bf2Raw`
   collection, or (v0.6) a `Scene`.
2. Pick a [high level `write_*` function](#high-level-functions), or a
   [lower level builder](#builders).
3. Call it with your model and your data.

Each `Dataset` (resolution level) in an image's multiscale becomes one array in
the store, so functions that take arrays (or `(shape, dtype)` pairs) expect one
per dataset, in the same order.

```python
import numpy as np

from yaozarrs import DimSpec, v06
from yaozarrs.write.v06 import prepare_image, write_image

dims = [
    DimSpec(name="y", scale=0.1, unit="micrometer"),
    DimSpec(name="x", scale=0.1, unit="micrometer"),
]
image = v06.Image(multiscales=[v06.Multiscale.from_dims(dims)])
data = np.zeros((256, 256), dtype=np.uint16)

# all at once
write_image("image.ome.zarr", image, data)

# or: create the hierarchy with empty arrays, and fill them yourself
path, arrays = prepare_image("prepared.ome.zarr", image, (data.shape, data.dtype))
arrays["0"][:] = data
```

## High level functions

These take all the data at once, write it, and return the `Path` to the root of
the store.

| Function | Writes | |
| -------- | ------ | - |
| `write_image` | an image (with optional labels) | [v0.5][yaozarrs.write.v05.write_image] · [v0.6][yaozarrs.write.v06.write_image] |
| `write_plate` | a multi-well plate | [v0.5][yaozarrs.write.v05.write_plate] · [v0.6][yaozarrs.write.v06.write_plate] |
| `write_bioformats2raw` | a bioformats2raw collection | [v0.5][yaozarrs.write.v05.write_bioformats2raw] · [v0.6][yaozarrs.write.v06.write_bioformats2raw] |
| `write_scene` | a scene of images | [v0.6][yaozarrs.write.v06.write_scene] |

## Builders

When you don't have all the data up front, or want more control over how it's
written, the lower level functions *prepare* the hierarchy: they write the
metadata, create empty arrays, and give them back to you (keyed by path) to fill
however you like (chunk by chunk, in parallel, ...).

| Function | Prepares | |
| -------- | -------- | - |
| `prepare_image` | an image | [v0.5][yaozarrs.write.v05.prepare_image] · [v0.6][yaozarrs.write.v06.prepare_image] |
| `LabelsBuilder` | the label images of an image | [v0.5][yaozarrs.write.v05.LabelsBuilder] · [v0.6][yaozarrs.write.v06.LabelsBuilder] |
| `PlateBuilder` | a multi-well plate | [v0.5][yaozarrs.write.v05.PlateBuilder] · [v0.6][yaozarrs.write.v06.PlateBuilder] |
| `Bf2RawBuilder` | a bioformats2raw collection | [v0.5][yaozarrs.write.v05.Bf2RawBuilder] · [v0.6][yaozarrs.write.v06.Bf2RawBuilder] |
| `SceneBuilder` | a scene of images | [v0.6][yaozarrs.write.v06.SceneBuilder] |

The builders can be used in two ways:

- **incremental**: write one well, label or image at a time (e.g.
  `PlateBuilder.write_well()`), and the parent metadata is updated after each
  one.
- **prepare-only**: register everything with `add_*()`, then `prepare()` creates
  the whole hierarchy with empty arrays.

For plates and bioformats2raw collections, the builders are usually the easier
choice.

## Array writer backends

Every function takes a `writer` argument, which picks the library that creates
the arrays:

- `"zarr"`: [zarr-python](https://github.com/zarr-developers/zarr-python),
  `pip install "yaozarrs[write-zarr]"`
- `"tensorstore"`: [tensorstore](https://github.com/google/tensorstore),
  `pip install "yaozarrs[write-tensorstore]"`
- `"auto"` (the default): tensorstore if it's installed, zarr-python otherwise
- your own function, see [Custom writers](#custom-writers)

## Custom writers

!!! warning "Advanced"
    Not needed for most users.  Skip this if the built-in zarr-python or
    tensorstore backends work for you.

::: yaozarrs.write._common._arrays.CreateArrayFunc
      options:
        heading_level: 3
        heading: "CreateArrayFunc"
        show_root_heading: true
        show_source: false

## Types

The type hints used in the signatures of both modules.  (They aren't importable:
they're only there for type checkers, and for these docs.)

::: yaozarrs.write._common._arrays.ArrayLike
      options:
        heading_level: 3
        heading: "ArrayLike"
        show_root_heading: true

::: yaozarrs.write._common._arrays.ArrayOrPyramid
      options:
        heading_level: 3
        heading: "ArrayOrPyramid"
        show_root_heading: true

::: yaozarrs.write._common._arrays.ShapeLike
      options:
        heading_level: 3
        heading: "ShapeLike"
        show_root_heading: true

::: yaozarrs.write._common._arrays.ShapeAndDType
      options:
        heading_level: 3
        heading: "ShapeAndDType"
        show_root_heading: true

::: yaozarrs.write._common._arrays.ShapeAndDTypeOrPyramid
      options:
        heading_level: 3
        heading: "ShapeAndDTypeOrPyramid"
        show_root_heading: true

::: yaozarrs.write._common._arrays.ZarrWriter
      options:
        heading_level: 3
        heading: "ZarrWriter"
        show_root_heading: true

::: yaozarrs.write._common._arrays.WriterName
      options:
        heading_level: 3
        heading: "WriterName"
        show_root_heading: true

::: yaozarrs.write._common._arrays.CompressionName
      options:
        heading_level: 3
        heading: "CompressionName"
        show_root_heading: true

::: yaozarrs.write._common._arrays.AnyZarrArray
      options:
        heading_level: 3
        heading: "AnyZarrArray"
        show_root_heading: true

::: yaozarrs.write._common._builders.Image
      options:
        heading_level: 3
        heading: "Image"
        show_root_heading: true

::: yaozarrs.write._common._builders.LabelImage
      options:
        heading_level: 3
        heading: "LabelImage"
        show_root_heading: true

::: yaozarrs.write._common._builders.ImageWithDatasets
      options:
        heading_level: 3
        heading: "ImageWithDatasets"
        show_root_heading: true

::: yaozarrs.write._common._builders.ImageWithShapeSpecs
      options:
        heading_level: 3
        heading: "ImageWithShapeSpecs"
        show_root_heading: true
