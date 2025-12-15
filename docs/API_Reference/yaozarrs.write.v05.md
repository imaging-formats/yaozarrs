---
title: yaozarrs.write.v05
---

# <code class="doc-symbol-module"></code> `yaozarrs.write.v05`

This module provides convenience functions to write OME-Zarr v0.5 stores.

## Overview

The general pattern is:

1. Decide what kind of OME **model** best matches your data (see [Guide to
   OME-Zarr](../ome_zarr_guide.md) if you're new to OME Zarr), and construct your
   OME-Zarr metadata model using
   [yaozarrs.v05](../API_Reference/yaozarrs.v05.md) models:
    - Single Image (<=5D): use [`Image`][yaozarrs.v05.Image].
    - Multi-Well Plate: use [`Plate`][yaozarrs.v05.Plate].
    - Collection of images (e.g. 5D images at multiple positions, or any other
        6+D data): use bioformats2raw layout with
        [`Bf2Raw`][yaozarrs.v05.Bf2Raw].
1. Decide whether to use **high level** write functions
   (`write_image`, `write_plate`, etc...) or **lower level**
   prepare/Builder methods (`prepare_image`, `PlateBuilder`, etc...):

    - **High Level** `write_*` functions immediately write data and return a `Path` to the root
      of the written zarr group.

        | Name | Description |
        | ---- | ----------- |
        | [`write_image`][yaozarrs.write.v05.write_image] | Write a single Image OME-Zarr v0.5 store. |
        | [`write_plate`][yaozarrs.write.v05.write_plate] | Write a multi-well Plate OME-Zarr v0.5 store. |
        | [`write_bioformats2raw`][yaozarrs.write.v05.write_bioformats2raw] | Write a bioformats2raw-style OME-Zarr v0.5 store. |

    - **Low Level** `prepare_*/Builder` methods *prepare* the zarr hierarchy,
      instantiate empty Arrays, and return a mapping of dataset-path to backend
      Array object, which you can then use to write data.  For complex groups,
        they can write one dataset at a time, or in any order you like.

        | Name | Description |
        | ---- | ----------- |
        | [`prepare_image`][yaozarrs.write.v05.prepare_image] | Prepare a single Image OME-Zarr v0.5 store for writing. |
        | [`LabelsBuilder`][yaozarrs.write.v05.LabelsBuilder] | Prepare a Labels OME-Zarr v0.5 store for writing. |
        | [`PlateBuilder`][yaozarrs.write.v05.PlateBuilder] | Prepare a multi-well Plate OME-Zarr v0.5 store for writing. |
        | [`Bf2RawBuilder`][yaozarrs.write.v05.Bf2RawBuilder] | Prepare a bioformats2raw-style OME-Zarr v0.5 store for writing. |

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
    between a [`Dataset`][yaozarrs.v05.Dataset] (nested inside the `Image`
    model), and an array node in the output zarr hierarchy**.  Most functions that
    accept arrays and/or `(shape, dtype)` pairs are expecting one per Dataset in
    the model.

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

::: yaozarrs.write.v05
      options:
        heading_level: 2
        show_root_heading: true
        summary: false
        show_overloads: false

## Custom Writers

!!! warning "Advanced"
    This is an advanced feature, not needed for most users.  You can skip this
    section if you are happy using the built-in zarr or tensorstore implementations.

::: yaozarrs.write.v05._write.CreateArrayFunc
      options:
        heading_level: 3
        heading: "CreateArrayFunc"
        show_root_heading: true
        show_source: false
