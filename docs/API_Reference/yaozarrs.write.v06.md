---
title: yaozarrs.write.v06
---

# <code class="doc-symbol-module"></code> `yaozarrs.write.v06`

Functions to write OME-Zarr v0.6 stores from
[`yaozarrs.v06`](yaozarrs.v06.md) models.  See [Writing OME-Zarr](../writing.md)
for how to use them.

## Notes for v0.6

- `dimension_names` are still written.  v0.5 *required* each array's
  `zarr.json` to include `dimension_names` matching the axes.  v0.6 no longer
  requires it, but yaozarrs still writes them, using the axis names of the
  multiscale's ["intrinsic" coordinate
  system](../ome_zarr_guide/coordinate_systems.md#the-intrinsic-coordinate-system).
- No transformation is added between an image and its labels.  Without one, a
  label image is usually understood to share the image's intrinsic coordinate
  system.  If it doesn't (e.g. your labels were written for a crop of the
  image, with the crop's corner as their origin), add a
  transformation to the image's `Multiscale.coordinateTransformations` (see [Lining labels up with the
  image](../ome_zarr_guide/index.md#lining-labels-up-with-the-image-v06)).
- `write_scene` writes the [`Scene`][yaozarrs.v06.Scene] as given, after
  checking it against the images.  Transformation parameters stored in Zarr
  arrays (at a `path`) aren't written for you.

::: yaozarrs.write.v06
      options:
        heading_level: 2
        show_root_heading: true
        summary: false
        inherited_members: true

## Types

See also the [types shared by both write modules](../writing.md#types).

::: yaozarrs.write.v06._write.ImageWithDatasets
      options:
        heading_level: 3
        heading: "ImageWithDatasets"
        show_root_heading: true

::: yaozarrs.write.v06._write.LabelsWithDatasets
      options:
        heading_level: 3
        heading: "LabelsWithDatasets"
        show_root_heading: true

::: yaozarrs.write.v06._write.LabelsWithSpecs
      options:
        heading_level: 3
        heading: "LabelsWithSpecs"
        show_root_heading: true

::: yaozarrs.write._common._builders._Default
      options:
        heading_level: 3
        heading: "_Default"
        show_root_heading: true
        members_order: source
