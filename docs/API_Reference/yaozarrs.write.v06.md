---
title: yaozarrs.write.v06
---

# <code class="doc-symbol-module"></code> `yaozarrs.write.v06`

Functions to write OME-Zarr v0.6 stores from
[`yaozarrs.v06`](yaozarrs.v06.md) models.  See [Writing OME-Zarr](../writing.md)
for how to use them.

## Things that are specific to v0.6

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
  checks it against the images (and their `labels`) before anything is written:
  every image it refers to must be there, with the coordinate systems it names,
  and the scene's transformations must connect them all.  (A reference to a label
  image you don't pass in isn't checked, and neither is the graph, in that case.)
- **Transformation parameters stored in Zarr aren't written for you.**  An
  `affine` or `rotation` may keep its matrix in a Zarr array (referenced by
  `path`) instead of in the JSON, and `displacements` and `coordinates` always keep
  their vector field in a separate multiscale image group.  If you use those,
  write that data yourself (for a scene, the spec recommends a group called
  `coordinateTransformations`, next to the images).

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
