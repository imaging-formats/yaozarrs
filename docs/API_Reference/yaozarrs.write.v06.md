---
title: yaozarrs.write.v06
---

# <code class="doc-symbol-module"></code> `yaozarrs.write.v06`

This module provides convenience functions to write OME-Zarr v0.6 stores.

!!! tip "Coming from `yaozarrs.write.v05`?"
    The functions here are the same as in [`yaozarrs.write.v05`][], with one
    addition: [`write_scene`][yaozarrs.write.v06.write_scene] (and
    [`SceneBuilder`][yaozarrs.write.v06.SceneBuilder]), for images that share a
    spatial relationship.  The difference is in the metadata models you pass in,
    which come from [`yaozarrs.v06`](../API_Reference/yaozarrs.v06.md).

## Overview

The general pattern is:

1. Decide what kind of OME **model** best matches your data (see [Guide to
   OME-Zarr](../ome_zarr_guide/index.md) if you're new to OME Zarr), and construct your
   OME-Zarr metadata model using
   [yaozarrs.v06](../API_Reference/yaozarrs.v06.md) models:
    - Single Image (<=5D): use [`Image`][yaozarrs.v06.Image].
    - Multi-Well Plate: use [`Plate`][yaozarrs.v06.Plate].
    - Collection of images (e.g. 5D images at multiple positions, or any other
        6+D data): use bioformats2raw layout with
        [`Bf2Raw`][yaozarrs.v06.Bf2Raw].
    - Images that share a spatial relationship (e.g. tiles of a mosaic, or
        multiple views of the same sample): use [`Scene`][yaozarrs.v06.Scene].
1. Decide whether to use **high level** write functions
   (`write_image`, `write_plate`, etc...) or **lower level**
   prepare/Builder methods (`prepare_image`, `PlateBuilder`, etc...):

    - **High Level** `write_*` functions immediately write data and return a `Path` to the root
      of the written zarr group.

        | Name | Description |
        | ---- | ----------- |
        | [`write_image`][yaozarrs.write.v06.write_image] | Write a single Image OME-Zarr v0.6 store. |
        | [`write_plate`][yaozarrs.write.v06.write_plate] | Write a multi-well Plate OME-Zarr v0.6 store. |
        | [`write_bioformats2raw`][yaozarrs.write.v06.write_bioformats2raw] | Write a bioformats2raw-style OME-Zarr v0.6 store. |
        | [`write_scene`][yaozarrs.write.v06.write_scene] | Write a Scene OME-Zarr v0.6 store (several images that share a spatial relationship). |

    - **Low Level** `prepare_*/Builder` methods *prepare* the zarr hierarchy,
      instantiate empty Arrays, and return a mapping of dataset-path to backend
      Array object, which you can then use to write data.  For complex groups,
        they can write one dataset at a time, or in any order you like.

        | Name | Description |
        | ---- | ----------- |
        | [`prepare_image`][yaozarrs.write.v06.prepare_image] | Prepare a single Image OME-Zarr v0.6 store for writing. |
        | [`LabelsBuilder`][yaozarrs.write.v06.LabelsBuilder] | Prepare a Labels OME-Zarr v0.6 store for writing. |
        | [`PlateBuilder`][yaozarrs.write.v06.PlateBuilder] | Prepare a multi-well Plate OME-Zarr v0.6 store for writing. |
        | [`Bf2RawBuilder`][yaozarrs.write.v06.Bf2RawBuilder] | Prepare a bioformats2raw-style OME-Zarr v0.6 store for writing. |
        | [`SceneBuilder`][yaozarrs.write.v06.SceneBuilder] | Prepare a Scene OME-Zarr v0.6 store for writing. |

    ??? question "Still confused: Which API should I use?"
        - Use high level `write_*` functions for simple one-shot writes, **where
        you can provide the full data arrays up front** (either in-memory with numpy,
        or dask, etc...)
        - Use lower level `prepare_*/Builder` APIs **when you need to
        customize how data is written, perhaps in a streaming, or slice-by-slice
        manner**, or when you want to use the backend array writing API directly.

        *Note: lower level builders are recommended for Plates and Bf2Raw collections*

1. Call the appropriate function with your metadata model and data arrays.

!!! tip
    A key observation here is that **there is generally a one-to-one mapping
    between a [`Dataset`][yaozarrs.v06.Dataset] (nested inside the `Image`
    model), and an array node in the output zarr hierarchy**.  Most functions that
    accept arrays and/or `(shape, dtype)` pairs are expecting one per Dataset in
    the model.

### Things that are specific to v0.6

- **`dimension_names` are still written.**  v0.5 *required* each array's
  `zarr.json` to include `dimension_names` matching the axes.  v0.6 no longer
  requires it, but other Zarr tools find it useful, so yaozarrs still writes them,
  using the axis names of the multiscale's ["intrinsic" coordinate
  system](../ome_zarr_guide/coordinate_systems.md#the-intrinsic-coordinate-system).
- **No transformation is added between an image and its labels.**  Without one, a
  label image is usually understood to share the image's intrinsic coordinate
  system.  If it doesn't (e.g. your labels were written for a crop of the
  image, with the crop's corner as their origin), add a
  transformation to the image's `Multiscale.coordinateTransformations` (see [Lining labels up with the
  image](../ome_zarr_guide/index.md#lining-labels-up-with-the-image-v06)).
- **The scene metadata is written as given.**  Unlike plate metadata, which
  `write_plate` can generate for you, a [`Scene`][yaozarrs.v06.Scene] says how
  *your* images relate to each other, so you write it yourself.  `write_scene`
  checks it against the images before anything is written: every image it refers
  to must be there, with the coordinate systems it names, and the scene's
  transformations must connect them all.
- **Transformation parameters stored in Zarr aren't written for you.**  An
  `affine` or `rotation` may keep its matrix in a Zarr array (referenced by
  `path`) instead of in the JSON, and `displacements` and `coordinates` always keep
  their vector field in a separate multiscale image group.  If you use those,
  write that data yourself (for a scene, the spec recommends a group called
  `coordinateTransformations`, next to the images).

## Choosing an Array Writer Backend

All functions in this module eventually write zarr arrays.  You can control what
backend is used to write arrays using the `writer` argument to each function,
which takes a string literal (name of the backend), or a custom
`CreateArrayFunc` function.  If you want to use builtin writers backends, you
must install with the appropriate extras:

- [zarr-python](https://github.com/zarr-developers/zarr-python): `pip install yaozarrs[write-zarr]`
- [tensorstore](https://github.com/google/tensorstore): `pip install yaozarrs[write-tensorstore]`

You may also implement and pass in your own `writer` function.  See the
[Custom Writers](#custom-writers) guide for details.

-------------

::: yaozarrs.write.v06
      options:
        heading_level: 2
        show_root_heading: true
        summary: false
        show_overloads: false
        inherited_members: true

## Custom Writers

!!! warning "Advanced"
    This is an advanced feature, not needed for most users.  You can skip this
    section if you are happy using the built-in zarr or tensorstore implementations.

::: yaozarrs.write.v06._write.CreateArrayFunc
      options:
        heading_level: 3
        heading: "CreateArrayFunc"
        show_root_heading: true
        show_source: false
