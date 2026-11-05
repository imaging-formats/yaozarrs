---
icon: material/vector-polyline
title: Coordinate Systems & Transformations
---

# Coordinate Systems & Transformations

!!! tip "What you'll learn"
    Since v0.6, OME-Zarr describes *where your pixels are* with **coordinate
    systems** and **transformations**.  This page explains what they are, and
    how they appear in the metadata.

    If you only need a single image with a pixel size, you can mostly skip
    this: [`Multiscale.from_dims`][yaozarrs.v06._image.Multiscale.from_dims] handles it
    for you.  Read on when you need to place images relative to each other,
    line up labels, register images, or just want to understand what you're
    looking at.

!!! tip ""
    This page is based on the [OME-Zarr v0.6
    specification](https://ngff.openmicroscopy.org/specifications/0.6/index.html)
    and [RFC-5](https://ngff.openmicroscopy.org/rfc/5/index.html).  In case of any
    discrepancies, the official spec takes precedence!

## Coordinate systems

A **coordinate system** is just a `name`, plus a list of `axes`:

```json
{
  "name": "physical",
  "axes": [
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"}
  ]
}
```

A point in this coordinate system is a list of 3 numbers, `(z, y, x)`, in
micrometers.  The number of axes is the coordinate system's *dimensionality* (3,
here), and the order of the axes is the order of the numbers in each point.

Coordinate systems are defined in two places:

- **In an image** (required): each multiscale lists its coordinate systems under
  `coordinateSystems`.  This replaces the `axes` field of v0.5.
- **In a scene** (optional): a [scene](index.md#working-with-scenes)
  may define its own, e.g. a shared "stage" or "world" coordinate system that
  several images are placed into.

!!! warning "Identical axes aren't the same space"
    Two coordinate systems with identical axes are *not* necessarily "the
    same".  An image of sample A and an image of sample B may both have
    `z, y, x` axes in micrometers, but a point at `(0, 0, 0)` in one has nothing
    to do with `(0, 0, 0)` in the other.

    A coordinate system is identified by its `name` *and where it is defined*
    (which `zarr.json`).  The spec says analyses that combine images, labels,
    ROIs etc. **SHOULD** make sure they are in the same coordinate system (or can
    be transformed into one) first.

---

## Transformations

A **transformation** describes how to take a *point* in one coordinate system
(its `input`) and find where it is in another (its `output`):

```json
{
  "type": "scale",
  "scale": [2.0, 0.5, 0.5],
  "input": {"name": "pixels"},
  "output": {"name": "physical"}
}
```

This one says: take a point `(z, y, x)` in `pixels`, multiply each coordinate by
its scale factor, and you get the same point in `physical`:

```
z_physical = 2.0 * z_pixels
y_physical = 0.5 * y_pixels
x_physical = 0.5 * x_pixels
```

Every transformation has a `type`, which says what it does and which other
fields it needs (here: `scale`).  It may also have a `name`, to identify
it.  The [Transformation types](#transformation-types) section
lists them all.

!!! warning "Transformations point from `input` to `output`"
    Transformations map *points* from the input space to the output space
    (the spec calls this the "forward" direction).  To *draw* image A into the
    coordinate system of image B, a viewer usually needs the *inverse*: for
    each pixel of the output, where should it sample A?  For simple
    transformations (scale, translation, rotation...) the viewer can compute
    that inverse itself.

    See also the [`bijection`](#transformation-types) type, which stores both
    directions explicitly.

### Referring to coordinate systems

The `input` and `output` of a transformation are objects with (up to) two
fields:

- **`name`**: the name of a coordinate system.
- **`path`**: where to find it, if it's defined in a *different* group (e.g.
  `"labels/cells"`, or `"tile_0"`).  If the `path` is left out (or `null`), the
  coordinate system is in the same `zarr.json`.

Together, `name` and `path` always point at exactly one coordinate system.
Which of them are required depends on where the transformation is stored:

| Where it's stored | `input` | `output` |
|-------------------|---------|----------|
| multiscales > datasets | `{"path": "<dataset path>"}` | `{"name": "physical"}` |
| multiscales > coordinateTransformations | `{"name": "physical"}` | `{"name": "other"}` <br> or <br> `{"name": "physical", "path": "labels/cells"}` |
| scene > coordinateTransformations | `{"name": "physical", "path": "imageA"}` | `{"name": "physical", "path": "imageB"}` |

(The first row is special: a dataset's `input` is the dataset's *array*
itself, referenced by its `path`.)

---

## The intrinsic coordinate system

Every multiscale image has one special coordinate system: the one that **all of
its datasets (resolution levels) map into**.  The spec calls it the
**"intrinsic" coordinate system** of the image.  It is "a representation of the
image in its native physical coordinate system", and the one to use "for
viewing and processing unless a use case dictates otherwise".

```json
{
  "multiscales": [{
    "coordinateSystems": [
      {
        "name": "physical",  // ← the intrinsic coordinate system
        "axes": [
          {"name": "y", "type": "space", "unit": "micrometer"},
          {"name": "x", "type": "space", "unit": "micrometer"}
        ]
      }
    ],
    "datasets": [
      {
        "path": "0",
        "coordinateTransformations": [{
          "type": "scale",
          "scale": [0.5, 0.5],
          "input": {"path": "0"},
          "output": {"name": "physical"}
        }]
      },
      {
        "path": "1",
        "coordinateTransformations": [{
          "type": "sequence",
          "input": {"path": "1"},
          "output": {"name": "physical"},
          "transformations": [
            {"type": "scale", "scale": [1.0, 1.0]},
            {"type": "translation", "translation": [0.25, 0.25]}
          ]
        }]
      }
    ]
  }]
}
```

Here, the intrinsic coordinate system is called `physical`, but the name is up
to you.  (The spec's examples mostly use `physical` or `intrinsic`;
[`Multiscale.from_dims`][yaozarrs.v06._image.Multiscale.from_dims] uses `"intrinsic"`
unless you pass `coordinate_system=...`).

The rules for these dataset transformations are listed under [Coordinate
Transformations](index.md#coordinate-transformations) in the guide.

### Where is a pixel?

Level `1` above has a translation of `0.25` because of this convention:

!!! info "The pixel center is the origin"
    Pixel `(0, 0)` of an array is the point `(0.0, 0.0)` in its own (array)
    coordinates, and that point is the *center* of the pixel.  The pixel
    itself covers the area from `-0.5` to `0.5` along each axis (including
    `-0.5`, excluding `0.5`).

So when you downsample by 2 (e.g. by averaging 2×2 blocks), the center of the
first big pixel is *not* at the center of the first small pixel: it's halfway
between the first two small pixels.  To keep the levels aligned, each level needs
a translation of:

```
translation = (pixel size at this level - pixel size at level 0) / 2
```

(plus level 0's own translation, if it has one).  Here: `(1.0 - 0.5) / 2 = 0.25`.  Try it below: hover over a pixel to see where its
center is, and toggle the translation to see what happens without it.

<ome-pixel-grid></ome-pixel-grid>

!!! tip "yaozarrs does this for you"
    [`Multiscale.from_dims`][yaozarrs.v06._image.Multiscale.from_dims] adds this
    translation to every downsampled level, using the `scale_factor` of each
    [`DimSpec`][yaozarrs.DimSpec].  That's right for common downsampling
    methods like binning; if your pyramid was made differently, adjust it.

---

## More coordinate systems in one image

A multiscale may define more than one coordinate system, linked to the
intrinsic one with transformations in the multiscale's own
`coordinateTransformations`.  These apply to all resolution levels at once.

For example, raw light sheet data acquired at an angle (e.g. with an oblique
plane microscope) might be stored as-is, with a second `deskewed` coordinate
system for the corrected geometry:

```json
{
  "multiscales": [{
    "coordinateSystems": [
      {"name": "intrinsic", "axes": [...]},  // z, y, x as acquired
      {"name": "deskewed", "axes": [...]}    // z, y, x after deskewing
    ],
    "datasets": [...],  // all map to "intrinsic"
    "coordinateTransformations": [
      {
        "type": "affine",
        "name": "deskew",
        "input": {"name": "intrinsic"},
        "output": {"name": "deskewed"},
        "affine": [
          [0.5,   0.0, 0.0, 0.0],
          [0.866, 1.0, 0.0, 0.0],
          [0.0,   0.0, 1.0, 0.0]
        ]
      }
    ]
  }]
}
```

!!! important "Rules for multiscale transformations"
    - One side (`input` or `output`) **MUST** be the intrinsic coordinate
      system (by `name`).
    - The other side **MUST** be either another coordinate system of the same
      multiscale (by `name`), or the coordinate system of a label image in the
      image's `labels` group (by `name` and `path`, see [lining labels up with
      the image](index.md#lining-labels-up-with-the-image-v06)).
    - A transformation to a label image **MUST** be an `identity`, `scale`,
      `translation`, or a `sequence` of a scale and a translation.

    So within the image's own metadata, every transformation touches the
    intrinsic coordinate system: there are no chains like
    `intrinsic → a → b`.

---

## Between images: scenes

Transformations between *different* images live in a
[scene](index.md#working-with-scenes): a group whose `zarr.json` has a
`scene` object, with the images as subgroups.  Scene transformations refer to
each image's coordinate systems by `name` *and* `path` (the image's path within
the scene), and to the scene's own coordinate systems by `name` alone:

```json
{
  "scene": {
    "coordinateSystems": [{"name": "stage", "axes": [...]}],
    "coordinateTransformations": [
      {
        "type": "translation",
        "translation": [0.0, 270.0],
        "input": {"path": "tile_1", "name": "physical"},
        "output": {"name": "stage"}
      }
    ]
  }
}
```

See [Working with Scenes](index.md#working-with-scenes) for the
details.

---

## Putting it together: the transformation graph

If you draw every coordinate system as a dot, and every transformation as an
arrow between two dots, you get a **graph**.  The spec has one firm rule about
it:

!!! important "The graph MUST be connected"
    Any two coordinate systems (in the multiscales and scene metadata) **MUST**
    be linked by a path of one or more transformations: either directly, or
    through other coordinate systems.  They don't *all* need to link to each
    other directly.

To get from any coordinate system to any other, follow the arrows (or walk them
backwards, where they can be inverted), composing the
transformations along the way.

Pick an example below, then click on an arrow (or a coordinate system) to see
its metadata, and which `zarr.json` it's stored in:

<ome-transform-graph></ome-transform-graph>

---

## Transformation types

v0.6 defines 12 types of transformations: `identity`, `scale`, `translation`,
`sequence`, `mapAxis`, `affine`, `rotation`, `byDimension`, `projectAxis`,
`bijection`, `displacements` and `coordinates`.  Readers **MUST** parse the
first four (for `sequence`, sequences of those), and **SHOULD** parse
`mapAxis`, `affine`, `rotation` and `byDimension`.  Actually *applying*
transformations to points or images is a **SHOULD**.  See the
[spec](https://ngff.openmicroscopy.org/specifications/0.6/index.html) for
what each one does and its parameters.

!!! warning "Gotchas"
    - Parameters follow the axis order.  The first number of a `scale` or
      `translation` is for the first axis of the coordinate system, and so on.
      With axes `c, z, y, x`, a scale is `[c, z, y, x]`.
    - Discrete axes still count.  If a coordinate system has a channel
      axis, it needs a parameter too (usually `1.0` for a scale, `0.0` for a
      translation), even though channels aren't "transformed".
    - Sequences apply in order: for `[f0, f1, f2]`, `f0` is applied first,
      and `f2` last (i.e. `f2(f1(f0(x)))`).
    - Matrices are lists of rows, applied to points as column vectors.  An
      `affine` from `N` to `M` dimensions has `M` rows of `N+1` numbers: the
      last number of each row is the translation (the `[0, ..., 0, 1]` row of
      the full homogeneous matrix is left out).  A `rotation` is `N x N`, with
      determinant 1 and orthonormal rows and columns.
    - Prefer simple transformations.  The spec says implementations
      **SHOULD** prefer a sequence of less expressive transformations (e.g.
      `sequence[translation, rotation]`) over an equivalent `affine`.

---

## In yaozarrs

Every transformation type has a model in
[`yaozarrs.v06`](../API_Reference/yaozarrs.v06.md), and they validate
what can be checked from the metadata alone (e.g. that a scale has one value per
axis, a rotation matrix is orthonormal, or that every dataset outputs to the
same coordinate system):

```python
from yaozarrs import DimSpec, v06

# an image, with its intrinsic coordinate system ("intrinsic")
ms = v06.Multiscale.from_dims(
    [
        DimSpec(name="z", scale=2.0, unit="micrometer", scale_factor=1),
        DimSpec(name="y", scale=0.5, unit="micrometer"),
        DimSpec(name="x", scale=0.5, unit="micrometer"),
    ],
    n_levels=3,
)
print(ms.intrinsic_coordinate_system.name)  # "intrinsic"

# add a second coordinate system, rotated 90 degrees in the yx plane
data = ms.model_dump(by_alias=True)
data["coordinateSystems"].append({**data["coordinateSystems"][0], "name": "rotated"})
data["coordinateTransformations"] = [
    v06.RotationTransformation(
        rotation=[[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        input={"name": "intrinsic"},
        output={"name": "rotated"},
    ).model_dump(by_alias=True)
]
image = v06.Image(multiscales=[v06.Multiscale.model_validate(data)])
```

Things that need the rest of the hierarchy (that a `path` points to a real
image, that the graph is connected across a scene, that an `affine` stored in a
Zarr array has the right shape...) are checked by
[`yaozarrs.validate_zarr_store`][].

!!! note "Disconnected coordinate systems in a single image"
    The "connected graph" rule is about the multiscales and the scene metadata
    *combined*, so a scene could link two coordinate systems of an image that
    the image itself doesn't.  For that reason, when
    [`validate_zarr_store`][yaozarrs.validate_zarr_store] checks an image on
    its own (not as part of a scene), it only *warns* about coordinate
    systems that aren't linked to anything; inside a scene, it's an error.

---

## Further reading

- The coordinate systems, transformations and scene sections of the [v0.6
  specification](https://ngff.openmicroscopy.org/specifications/0.6/index.html)
- [RFC-5: Coordinate Systems and
  Transformations](https://ngff.openmicroscopy.org/rfc/5/index.html), with its
  [reviews](https://ngff.openmicroscopy.org/rfc/5/reviews/index.html) and
  [responses](https://ngff.openmicroscopy.org/rfc/5/responses/index.html)
- The [ITK Software Guide](https://itk.org/ItkSoftwareGuide.pdf) (chapter 4),
  which the spec points to for a more formal definition of the pixel
  convention
