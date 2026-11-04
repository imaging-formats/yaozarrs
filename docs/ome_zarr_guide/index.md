---
icon: material/book-open-variant
title: Guide to OME-Zarr
---

# Yaozarrs Guide to OME-Zarr

<script type="module" src="/javascripts/ome_explorer.js"></script>

!!! tip "What you'll learn"
    This guide attempts to demystify the **OME-Zarr (OME-Zarr)** specification and shows
    you how to work with it using yaozarrs.  It is designed to answer common questions
    and confusions encountered in the community.

### :material-rocket-launch: Quicklinks

<div class="grid cards" markdown>
- :material-cube-outline:{ .lg .middle } **I have images**

    ---
    Any data with 5 or less dimensions, typically `[T][C][Z]YX`.

    [:octicons-arrow-right-24: Go to Images](#working-with-images)

- :material-grid:{ .lg .middle } **I have plate data**

    ---
    Multi-well plates and high-content screening (HCS) experiments

    [:octicons-arrow-right-24: Go to Plates](#working-with-plates)

- :material-tag-multiple:{ .lg .middle } **I have image annotations**

    ---

    Segmentation masks, annotation labels, and regions of interest (ROIs)

    [:octicons-arrow-right-24: Go to Labels](#labels-segmentation-masks)

- :material-folder-multiple:{ .lg .middle } **I have multiple images**

    ---

    Collections of related images (multi-FOV, stage positions, split files)

    [:octicons-arrow-right-24: Go to Collections](#working-with-collections)

- :material-vector-polyline:{ .lg .middle } **My images are related in space**

    ---

    Tiles of a mosaic, multiple views of a sample, or images to register onto
    each other (*v0.6 and later*)

    [:octicons-arrow-right-24: Go to Scenes](#working-with-scenes)

</div>

!!! info "Which version is this guide about?"
    This guide covers OME-Zarr **v0.4, v0.5 and v0.6**, with tabs to show the
    differences.  The coordinate systems and transformations of v0.6 get their
    own page: [Coordinate Systems & Transformations](coordinate_systems.md).

    ??? question "Which version should I write?"
        It depends on who needs to read your data.  Newer versions can express
        more, but your files are only useful if the tools you (and your
        collaborators) rely on can read them, so check those first.

## What is OME-Zarr?

!!! tip ""
    The official OME-Zarr specification can be found at
    <https://ngff.openmicroscopy.org/>.  In case of
    any discrepancies between this guide and the official spec, the official
    spec takes precedence!

OME-Zarr is a file format specification used by the bioimaging community for
storing multi-dimensional data. It is a "meta-specification", based on the
pre-existing Zarr format, which is designed for the storage of chunked,
compressed, N-dimensional arrays. OME-Zarr **extends** Zarr by adding metadata
conventions specific to bioimaging, making it easier to store and share complex
imaging datasets.

!!! question "But what *is* it?"

    _To resolve a somewhat common confusion..._

    OME-Zarr is "just" [Zarr](https://zarr.dev) (A file format used in many
    domains). The "OME" part is a specification *on top of* the zarr
    format that additionally defines:

    1. **How domain specific metadata should be stored.**      
      The details are version-specific, but this generally defines the exact
      form of the data inside of the `.zattrs` or `zarr.json` files that
      accompany the zarr groups.

    1. **How datasets are organized.**  
      Beyond metadata, the OME-Zarr specification also defines how datasets
      should be organized. For example: it defines how the images collected
      across a multi-well plate experiment should be organized in a single
      Zarr directory, or how the different resolutions of a multi-scale
      (pyramidal) image should be stored.

---

## Working with Images

An **Image** is the fundamental building block of OME-Zarr.

As of v0.6 (and since v0.4), a single image may have **no less than 2 and no
more than 5 dimensions**, and may store multiple resolution levels.

- **Spatial dimensions**: X, Y, optionally Z
- **Time**: T (temporal axis)
- **Channels**: C (fluorescence channels, RGB, etc.)

??? question "What if I have more than 5 dimensions?"
    While it is common to have datasets with more than 5 dimensions (e.g.,
    different stage positions in a shared coordinate space, angles in light
    sheet microscopy, etc.), there is currently no formal specification for more
    than 5 dimensions in OME-Zarr.  You may use the transitional
    `bioformats2raw.layout` to store multiple images in a single zarr group.
    See [Working with Collections](#working-with-collections)

    If the extra dimension is *where* each image was taken (stage positions,
    views from different angles, etc...), v0.6 also lets you describe how the
    images relate to each other in space with a *scene*.  See [Working with
    Scenes](#working-with-scenes).

    See also: an RFC ("request for comments") proposing a relaxation of this
    restriction: [RFC-3](https://ngff.openmicroscopy.org/rfc/3/index.html)

??? question "What if I have both RGB and optical channels?"
    As of v0.6, there is no formal specification for mixing the concepts
    of RGB image components and conventional "channels" (like optical
    configurations).  You will need to either create a custom group layout
    or flatten them all into a single channel dimension.

### Directory Structure

=== "v0.6"

    ```
    image.zarr/
    ├── zarr.json            # {"zarr_format": 3} group, with attributes.ome.multiscales
    ├── 0/                   # Full resolution array
    │   ├── zarr.json        # Array metadata (standard zarr schema)
    │   └── c/0/1/2/3        # Chunk files
    ├── 1/                   # downsampled level 1
    │   └── ...
    └── 2/                   # downsampled level 2
        └── ...
    ```

    *The same layout as v0.5: what changed in v0.6 is the metadata inside
    `zarr.json`.*

=== "v0.5"

    ```
    image.zarr/
    ├── zarr.json            # {"zarr_format": 3} group, with attributes.ome.multiscales
    ├── 0/                   # Full resolution array  
    │   ├── zarr.json        # Array metadata (standard zarr schema)
    │   └── c/0/1/2/3        # Chunk files
    ├── 1/                   # downsampled level 1
    │   └── ...
    └── 2/                   # downsampled level 2 
        └── ...
    ```

=== "v0.4"

    ```
    image.zarr/
    ├── .zgroup              # {"zarr_format": 2} group
    ├── .zattrs              # Contains "multiscales"
    ├── 0/                   # Full resolution array
    │   ├── .zarray          # Array metadata (standard zarr schema)
    │   └── t/c/z/y/x        # Chunk files with "/" separator
    ├── 1/                   # downsampled level 1
    │   └── ...
    └── 2/                   # downsampled level 2 
        └── ...
    ```

!!! tip "Key difference"
    Most of the structural changes between v0.4 and v0.5 relate to the transition
    from [Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html)
    to [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html).

    - **<=v0.4**: "multiscales" metadata directly in root of `.zattrs` files
    - **>=v0.5**: "multiscales" metadata in `zarr.json` under `attributes.ome` namespace

    Between v0.5 and v0.6, the storage is the same (Zarr v3). The changes are all
    in the metadata, mostly in how axes and transformations are described (see
    below).

---

### Axes

Axes define the dimensions of your image data. Starting with v0.4, axes are
**objects** with a `name`, and optional `type` and/or `unit` (v0.6 adds the
optional `discrete` and `longName`, and moves axes into named coordinate systems):

!!! important "Axis Constraints"
    Constraints for image axes in OME-Zarr are the same in v0.4, v0.5 and v0.6
    (in v0.6, they apply to every coordinate system inside a multiscale):

    - **MUST** have 2-5 dimensions total
    - **MUST** have 2-3 spatial axes
    - **MAY** have 0-1 time axis
    - **MAY** have 0-1 channel axis
    - **Ordering enforced**: time → channel/custom → space

    In practice, this limits valid axis combinations to: `[T][C][Z] Y X`  
    *(though no explicit restriction is placed on naming conventions)*

=== "v0.6"

    In v0.6, the axes are no longer listed directly in the multiscale.  Instead,
    they belong to a named **coordinate system**, and each multiscale lists one
    or more of these under `coordinateSystems`.  (The *why* is explained in
    [Coordinate Systems & Transformations](coordinate_systems.md).)

    **Spec JSON:**

    ```json
    {
      // found in a "multiscales" object
      "coordinateSystems": [
        {
          "name": "physical",
          "axes": [
            {"name": "c", "type": "channel", "discrete": true},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
          ]
        }
      ]
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v06

    physical = v06.CoordinateSystem(
        name="physical",
        axes=[
            v06.ChannelAxis(name="c", discrete=True),
            v06.SpaceAxis(name="z", unit="micrometer"),
            v06.SpaceAxis(name="y", unit="micrometer"),
            v06.SpaceAxis(name="x", unit="micrometer"),
        ],
    )
    ```

    !!! info "What's new for axes in v0.6"
        - **`discrete`** (optional): `true` if the axis may only be indexed by
          whole numbers (channels usually are), `false` if values *in between*
          samples are meaningful, so interpolating along it makes sense (space
          and time usually are).
        - **`longName`** (optional): a longer name or description of the axis.
        - **More suggested `type`s**: besides `space`, `time` and `channel`, the
          spec now lists `array`, `coordinate` and `displacement` (the latter two
          are used by [vector fields](coordinate_systems.md#transformation-types)).
          Other strings are still allowed for custom types.
        - Axis names **MUST** be non-empty, and unique *within their coordinate
          system*.

=== "v0.5"

    **Spec JSON:**

    ```json
    {
      // found in a "multiscales" object
      "axes": [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}
      ]
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v05

    axes = [
        v05.ChannelAxis(name="c"),
        v05.SpaceAxis(name="z", unit="micrometer"),
        v05.SpaceAxis(name="y", unit="micrometer"),
        v05.SpaceAxis(name="x", unit="micrometer"),
    ]
    ```

=== "v0.4"

    **Spec JSON:**

    ```json
    {
      // found in a "multiscales" object
      "axes": [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}
      ]
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v04

    axes = [
        v04.ChannelAxis(name="c"),
        v04.SpaceAxis(name="z", unit="micrometer"),
        v04.SpaceAxis(name="y", unit="micrometer"),
        v04.SpaceAxis(name="x", unit="micrometer"),
    ]
    ```

    !!! warning "Breaking change from v0.3"
        In v0.3, axes were simple strings: `["c", "z", "y", "x"]`. In v0.4+, they must be objects with explicit types.

---

### Coordinate Transformations

Starting in v0.4, **every dataset MUST include coordinate transformations** that
map data coordinates to physical coordinates.  Coordinate transforms are where
you would specify physical units (micrometers, seconds), multi-resolution scales,
as well as stage positions and spatial offsets for registration.

=== "v0.6"

    Each dataset still has a transformation from its array (pixel) coordinates
    to physical coordinates, but now it explicitly says *where* it maps from and
    to: its `input` is the dataset's array (by `path`), and its `output` is a
    coordinate system of the multiscale (by `name`).  Every dataset maps into
    the **same** coordinate system, which the spec calls the image's
    ["intrinsic" coordinate
    system](coordinate_systems.md#the-intrinsic-coordinate-system).

    **Scale Transformation:**

    **Spec JSON:**

    ```json
    {
      "coordinateSystems": [
        {"name": "physical", "axes": [...]}  // the 4 axes from above
      ],
      "datasets": [{
        "path": "0",
        "coordinateTransformations": [{
          "type": "scale",
          "scale": [1.0, 0.5, 0.1, 0.1],
          "input": {"path": "0"},
          "output": {"name": "physical"}
        }]
      }]
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v06

    dataset = v06.Dataset(
        path="0",
        coordinateTransformations=[
            v06.ScaleTransformation(
                scale=[1.0, 0.5, 0.1, 0.1],
                input={"path": "0"},
                output={"name": "physical"},
            )
        ],
    )
    ```

    **Scale + Translation:**

    A scale and a translation are no longer listed side by side.  To add a
    translation, wrap both in a single `sequence` transformation (scale first):

    **Spec JSON:**

    ```json
    {
      "path": "0",
      "coordinateTransformations": [{
        "type": "sequence",
        "input": {"path": "0"},
        "output": {"name": "physical"},
        "transformations": [
          {"type": "scale", "scale": [1.0, 0.5, 0.1, 0.1]},
          {"type": "translation", "translation": [0.0, 0.0, 100.0, 200.0]}
        ]
      }]
    }
    ```

    **yaozarrs Code:**

    ```python
    dataset = v06.Dataset(
        path="0",
        coordinateTransformations=[
            v06.SequenceTransformation(
                input={"path": "0"},
                output={"name": "physical"},
                transformations=[
                    v06.ScaleTransformation(scale=[1.0, 0.5, 0.1, 0.1]),
                    v06.TranslationTransformation(translation=[0.0, 0.0, 100.0, 200.0]),
                ],
            )
        ],
    )
    ```

    !!! warning "Transformation Rules (v0.6)"
        - **MUST** have exactly one transformation per dataset: a `scale`, an
          `identity`, or a `sequence` of a scale followed by a translation
        - Its `input` **MUST** have a `path` matching the dataset's `path`, and
          should omit `name` (see below)
        - Its `output` **MUST** have the `name` of a coordinate system in
          `coordinateSystems`, the same one for every dataset, and should omit
          `path` (see below)
        - Parameter lengths **MUST** match the number of axes
        - The scale gives the pixel size in physical units (or the time step).
          If that isn't known (or doesn't apply) for an axis, the value **MUST**
          be the downsampling factor relative to the first level instead (`1.0`
          if that axis isn't downsampled)

        *About omitting `input.name` and `output.path`:* the spec says this
        both ways.  Its "Constraints" section says they **MUST** be omitted,
        while its "multiscales" section says they **SHOULD** be.  To be safe,
        leave them out.  (yaozarrs warns, rather than failing, if they're
        present.)

    !!! tip "Use `Multiscale.from_dims`"
        Writing all of this by hand gets tedious.
        [`v06.Multiscale.from_dims`][yaozarrs.v06._image.Multiscale.from_dims] builds
        the coordinate system and every level's transformation for you (see
        [`DimSpec`][yaozarrs.DimSpec]).

    !!! info "`dimension_names` are no longer required"
        v0.5 required `dimension_names` in each array's `zarr.json`; v0.6 does
        not mention them.  (yaozarrs still writes them, as other Zarr tools find
        them useful.)

    **There's a lot more** that v0.6 can express with coordinate systems and
    transformations (rotations, affines, linking images to their labels or to
    each other...).  See [Coordinate Systems &
    Transformations](coordinate_systems.md).

=== "v0.5"

    Identical to v0.4, just stored under `attributes.ome` namespace.

    **yaozarrs Code (same as v0.4):**

    ```python
    from yaozarrs import v05

    dataset = v05.Dataset(
        path="0",
        coordinateTransformations=[
            v05.ScaleTransformation(scale=[1.0, 0.5, 0.1, 0.1]),
            v05.TranslationTransformation(translation=[0.0, 0.0, 100.0, 200.0]),
        ],
    )
    ```

    !!! info "v0.5 Additional Requirement"
        In v0.5, each array's `zarr.json` **MUST** include `dimension_names` matching the axes:

        ```json
        {
          "dimension_names": ["c", "z", "y", "x"]
        }
        ```

=== "v0.4"

    **Scale Transformation (REQUIRED):**

    Maps array indices to physical coordinates. Scale values represent the physical size per pixel for each dimension.

    **Spec JSON:**

    ```json
    {
      "datasets": [{
        "path": "0",
        "coordinateTransformations": [
          {"type": "scale", "scale": [1.0, 0.5, 0.1, 0.1]}
        ]
      }]
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v04

    dataset = v04.Dataset(
        path="0",
        coordinateTransformations=[v04.ScaleTransformation(scale=[1.0, 0.5, 0.1, 0.1])],
    )
    ```

    **Translation Transformation (OPTIONAL):**

    Adds a spatial offset. Must come after scale.

    **Spec JSON:**

    ```json
    {
      "coordinateTransformations": [
        {"type": "scale", "scale": [1.0, 0.5, 0.1, 0.1]},
        {"type": "translation", "translation": [0.0, 0.0, 100.0, 200.0]}
      ]
    }
    ```

    **yaozarrs Code:**

    ```python
    dataset = v04.Dataset(
        path="0",
        coordinateTransformations=[
            v04.ScaleTransformation(scale=[1.0, 0.5, 0.1, 0.1]),
            v04.TranslationTransformation(translation=[0.0, 0.0, 100.0, 200.0]),
        ],
    )
    ```

    !!! warning "Transformation Rules"
        - **MUST** have exactly one scale transformation per dataset
        - **MAY** have at most one translation transformation
        - If translation exists, it **MUST** come after scale
        - Transformation length **MUST** match number of axes

---

### Interactive Example

Modify the parameters below to see how different image configurations are
represented in OME-Zarr:

<ome-explorer preset=5d plate-control="false"></ome-explorer>

## Labels (Segmentation Masks)

Labels are specialized [images](#working-with-images) with integer dtype
representing segmentation masks (nuclei, cells, regions of interest, etc.).

They are represented as a special group named "labels/" within an image group.

!!! warning "Careful"

    This is one of the only places in the specification where the *name* of the
    group itself is normative: it **MUST** be named `labels/`.
    
    Conforming readers must *search* for the `labels/` group within an image
    group to discover it: its presence is not indicated in the parent image
    metadata.  (In v0.6, an image *may* refer to one of its label images, in an
    optional transformation, but doesn't have to.)

??? example "Label Structure and Code"

    === "v0.6"
        **Directory Structure:**

        Same as v0.5:

        ```
        image.zarr/
        ├── zarr.json            # Image metadata ("attributes.ome.multiscales")
        ├── 0/                   # Full resolution image
        ├── 1/                   # Downsampled level 1
        └── labels/
            ├── zarr.json        # Labels group metadata ("attributes.ome.labels")
            ├── nuclei/          # Label image (integer dtype)
            │   ├── zarr.json    # Label metadata ("attributes.ome.multiscales", "attributes.ome.image-label")
            │   ├── 0/           # Full resolution labels
            │   └── 1/           # Downsampled labels
            └── cells/
                └── ...
        ```

        **Labels Group Metadata (`labels/zarr.json`):**
        ```json
        {
          "zarr_format": 3,
          "node_type": "group",
          "attributes": {
            "ome": {
              "version": "0.6",
              "labels": ["nuclei", "cells"]
            }
          }
        }
        ```

        **Label Image Metadata (`labels/nuclei/zarr.json`):**
        ```json
        {
          "zarr_format": 3,
          "node_type": "group",
          "attributes": {
            "ome": {
              "version": "0.6",
              "multiscales": [...],
              "image-label": {
                "colors": [
                  {"label-value": 1, "rgba": [255, 0, 0, 255]},
                  {"label-value": 2, "rgba": [0, 255, 0, 255]}
                ],
                "source": {
                  "image": "../../"
                }
              }
            }
          }
        }
        ```

        *The v0.5 spec text also asked for a `version` key inside
        `image-label`.  v0.6 doesn't: the version is given once, at
        `ome.version`.*

        **yaozarrs Code:**
        ```python
        from yaozarrs import DimSpec, v06

        # Label metadata stored at labels/nuclei/zarr.json
        label_image = v06.LabelImage(
            # same structure as a regular image
            multiscales=[v06.Multiscale.from_dims([DimSpec(name="y"), DimSpec(name="x")])],
            image_label=v06.ImageLabel(
                colors=[
                    v06.LabelColor(label_value=1, rgba=[255, 0, 0, 255]),
                    v06.LabelColor(label_value=2, rgba=[0, 255, 0, 255]),
                ],
                source=v06.LabelSource(image="../../"),
            ),
        )
        ```

    === "v0.5"
        **Directory Structure:**

        ```
        image.zarr/
        ├── zarr.json            # Image metadata ("attributes.ome.multiscales")
        ├── 0/                   # Full resolution image
        ├── 1/                   # Downsampled level 1
        ├── ...                  # Downsampled level 2
        └── labels/
            ├── zarr.json        # Labels group metadata ("attributes.ome.labels")
            ├── nuclei/          # Label image (integer dtype)
            │   ├── zarr.json    # Label metadata ("attributes.ome.multiscales", "attributes.ome.image-label")
            │   ├── 0/           # Full resolution labels
            │   └── 1/           # Downsampled labels
            └── cells/
                └── ...
        ```

        **Labels Group Metadata (`labels/zarr.json`):**
        ```json
        {
          "zarr_format": 3,
          "node_type": "group",
          "attributes": {
            "ome": {
              "version": "0.5",
              "labels": ["nuclei", "cells"]
            }
          }
        }
        ```

        **Label Image Metadata (`labels/nuclei/zarr.json`):**
        ```json
        {
          "zarr_format": 3,
          "node_type": "group",
          "attributes": {
            "ome": {
              "version": "0.5",
              "multiscales": [...],
              "image-label": {
                "colors": [
                  {"label-value": 1, "rgba": [255, 0, 0, 255]},
                  {"label-value": 2, "rgba": [0, 255, 0, 255]}
                ],
                "source": {
                  "image": "../../"
                }
              }
            }
          }
        }
        ```

        **yaozarrs Code:**
        ```python
        from yaozarrs import DimSpec, v05

        # Label metadata stored at labels/nuclei/zarr.json
        label_image = v05.LabelImage(
            # same structure as a regular image
            multiscales=[v05.Multiscale.from_dims([DimSpec(name="y"), DimSpec(name="x")])],
            image_label=v05.ImageLabel(
                colors=[
                    v05.LabelColor(label_value=1, rgba=[255, 0, 0, 255]),
                    v05.LabelColor(label_value=2, rgba=[0, 255, 0, 255]),
                ],
                source=v05.LabelSource(image="../../"),
            ),
        )
        ```

    === "v0.4"

        **Directory Structure:**
        ```
        image.zarr/
        ├── .zgroup
        ├── .zattrs              # Image metadata ("multiscales")
        ├── 0/                   # Full resolution image
        ├── 1/                   # Downsampled level 1
        ├── ...                  # Downsampled level 2
        └── labels/
            ├── .zgroup
            ├── .zattrs          # Labels group metadata ("labels")
            ├── nuclei/          # Label image (integer dtype)
            │   ├── .zgroup
            │   ├── .zattrs      # Label metadata ("multiscales", "image-label")
            │   ├── 0/           # Full resolution labels
            │   └── 1/           # Downsampled labels
            └── cells/
                └── ...
        ```

        **Labels Group Metadata (`labels/.zattrs`):**
        ```json
        {
          "labels": ["nuclei", "cells"]
        }
        ```

        **Label Image Metadata (`labels/nuclei/.zattrs`):**
        ```json
        {
          "multiscales": [...],
          "image-label": {
            "version": "0.4",
            "colors": [
              {"label-value": 1, "rgba": [255, 0, 0, 255]},
              {"label-value": 2, "rgba": [0, 255, 0, 255]}
            ],
            "source": {
              "image": "../../"
            }
          }
        }
        ```

        **yaozarrs Code:**
        ```python
        from yaozarrs import DimSpec, v04

        # Label metadata stored at labels/nuclei/.zattrs
        label_image = v04.LabelImage(
            # same structure as a regular image
            multiscales=[v04.Multiscale.from_dims([DimSpec(name="y"), DimSpec(name="x")])],
            image_label=v04.ImageLabel(
                colors=[
                    v04.LabelColor(label_value=1, rgba=[255, 0, 0, 255]),
                    v04.LabelColor(label_value=2, rgba=[0, 255, 0, 255]),
                ],
                source=v04.LabelSource(image="../../"),
            ),
        )
        ```

    !!! warning "Labels must use integer dtype"
        Validation will fail if label arrays use float dtypes. Use `uint8`, `uint16`, `uint32`, or `int32`.

### Lining labels up with the image (v0.6)

By default, a label image's [intrinsic coordinate
system](coordinate_systems.md#the-intrinsic-coordinate-system) is "usually
understood to be the same" as its image's.  In v0.6, you can also link the two
when they *aren't* the same.  Say a
segmentation tool wrote labels for a crop of the image, with the corner of the
crop as its origin.  Rather than rewriting the label image's metadata, you can add
a transformation to the **parent image's** multiscale `coordinateTransformations`,
linking its intrinsic coordinate system to the label image's (referenced by `name`
*and* `path`, relative to the image group):

```json
{
  // found in the multiscales object of the *image* (not the label image)
  "coordinateTransformations": [
    {
      "type": "translation",
      "translation": [0.0, 100.0, 50.0],
      "input": {"name": "physical", "path": "labels/cells"},
      "output": {"name": "physical"}
    }
  ]
}
```

Here, a point in the `cells` label image's `physical` coordinate system, shifted
by `(0, 100, 50)`, gives the matching point in the image's own `physical`
coordinate system.  (Either direction is allowed: the image's intrinsic
coordinate system may be the `input` or the `output`.)

!!! warning "Only simple transformations"
    A transformation to a label image **MUST** be an `identity`, `scale` or
    `translation`.  The "labels" section of the spec also allows a `sequence` of
    a scale and a translation (its "Constraints" and "multiscales" sections
    don't), and yaozarrs accepts that too.

---

## Working with Plates

A **Plate** represents multi-well plate data from high-content screening (HCS) experiments. The hierarchy is:

**Plate** → **Rows/Columns** → **Wells** → **Fields of View (Images)**

Each well can contain multiple fields of view (FOVs) across multiple acquisitions (timepoints).

### Directory Structure

=== "v0.6"

    Same as v0.5:

    ```
    plate.zarr/
    ├── zarr.json              # contains Plate metadata ("attributes.ome.plate")
    ├── A/                     # Row A
    │   ├── 1/                 # Well A1
    │   │   ├── zarr.json      # contains Well metadata ("attributes.ome.well")
    │   │   ├── 0/             # Field 0 (Standard multiscales image)
    │   │   │   ├── zarr.json  # contains Image metadata ("attributes.ome.multiscales")
    │   │   │   ├── 0/         # Full resolution
    │   │   │   ├── 1/         # Downsampled
    │   │   │   └── labels/    # Optional labels group (see above)
    │   │   └── 1/             # Field 1
    │   └── 2/                 # Well A2
    └── B/                     # Row B
        └── 1/
    ```

=== "v0.5"

    ```
    plate.zarr/
    ├── zarr.json              # contains Plate metadata ("attributes.ome.plate")
    ├── A/                     # Row A
    │   ├── 1/                 # Well A1
    │   │   ├── zarr.json      # contains Well metadata ("attributes.ome.well")
    │   │   ├── 0/             # Field 0 (Standard multiscales image)
    │   │   │   ├── zarr.json  # contains Image metadata ("attributes.ome.multiscales")
    │   │   │   ├── 0/         # Full resolution
    │   │   │   ├── 1/         # Downsampled
    │   │   │   └── labels/    # Optional labels group (see above)
    │   │   └── 1/             # Field 1
    │   └── 2/                 # Well A2
    └── B/                     # Row B
        └── 1/
    ```

=== "v0.4"

    ```
    plate.zarr/
    ├── .zgroup
    ├── .zattrs              # contains Plate metadata ("plate")
    ├── A/                   # Row A
    │   ├── 1/               # Well A1
    │   │   ├── .zgroup
    │   │   ├── .zattrs      # contains Well metadata ("well")
    │   │   ├── 0/           # Field 0 (Standard multiscales image)
    │   │   │   ├── .zgroup
    │   │   │   ├── .zattrs  # contains Image metadata ("multiscales")
    │   │   │   ├── 0/       # Full resolution
    │   │   │   ├── 1/       # Downsampled
    │   │   │   └── labels/  # Optional labels group (see above)
    │   │   └── 1/           # Field 1
    │   └── 2/               # Well A2
    └── B/                   # Row B
        └── 1/
    ```

!!! note "Three-level hierarchy"
    Three groups **MUST** exist above images: **plate** → **row** → **well**

---

### Plate Metadata

=== "v0.6"

    The same structure as v0.5, with `"version": "0.6"` at `ome.version`.
    (The v0.5 spec *text* still asked for a separate `version` key inside
    `plate` and `well`, although its schemas didn't.  v0.6 removed it from the
    text too, since the version is already given at `ome.version`.)

    In yaozarrs, just swap `v05` for `v06`:

    ```python
    from yaozarrs import v06

    plate_def = v06.PlateDef(
        name="HCS Experiment",
        columns=[v06.Column(name="1"), v06.Column(name="2"), v06.Column(name="3")],
        rows=[v06.Row(name="A"), v06.Row(name="B")],
        wells=[
            v06.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            v06.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
            v06.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        ],
        acquisitions=[
            v06.Acquisition(id=0, name="Initial", maximumfieldcount=4),
            v06.Acquisition(id=1, name="24h", maximumfieldcount=4),
        ],
        field_count=4,
    )

    plate = v06.Plate(plate=plate_def)
    ```

=== "v0.5"

    Same structure as v0.4, stored under `attributes.ome` in `zarr.json`:

    **Spec JSON (`zarr.json` at plate root):**

    ```json
    {
      "zarr_format": 3,
      "node_type": "group",
      "attributes": {
        "ome": {
          "version": "0.5",
          "plate": {
            "columns": [
              { "name": "1" },
              { "name": "2" },
              { "name": "3" }
            ],
            "rows": [
              { "name": "A" },
              { "name": "B" }
            ],
            "wells": [
              { "path": "A/1", "rowIndex": 0, "columnIndex": 0 },
              { "path": "A/2", "rowIndex": 0, "columnIndex": 1 },
              { "path": "B/1", "rowIndex": 1, "columnIndex": 0 }
            ],
            "acquisitions": [
              { "id": 0, "maximumfieldcount": 4, "name": "Initial" },
              { "id": 1, "maximumfieldcount": 4, "name": "24h" }
            ],
            "field_count": 4,
            "name": "HCS Experiment"
          }
        }
      }
    }
    ```
    **yaozarrs Code:**

    ```python
    from yaozarrs import v05

    plate_def = v05.PlateDef(
        name="HCS Experiment",
        columns=[  # must have at least 1 column
            v05.Column(name="1"),
            v05.Column(name="2"),
            v05.Column(name="3"),
        ],
        rows=[  # must have at least 1 row
            v05.Row(name="A"),
            v05.Row(name="B"),
        ],
        wells=[  # must have at least 1 well, paths match tree structure
            v05.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            v05.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
            v05.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        ],
        acquisitions=[  # optional
            v05.Acquisition(id=0, name="Initial", maximumfieldcount=4),
            v05.Acquisition(id=1, name="24h", maximumfieldcount=4),
        ],
        field_count=4,  # max FOV per well
    )

    plate = v05.Plate(plate=plate_def)

    # Create full zarr.json
    zarr_json = v05.OMEZarrGroupJSON(attributes={"ome": plate})
    json_str = zarr_json.model_dump_json(indent=2)
    ```

=== "v0.4"

    **Spec JSON (`.zattrs` at plate root):**

    ```json
    {
      "plate": {
        "version": "0.4",
        "name": "HCS Experiment",
        "columns": [
          {"name": "1"},
          {"name": "2"},
          {"name": "3"}
        ],
        "rows": [
          {"name": "A"},
          {"name": "B"}
        ],
        "wells": [
          {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
          {"path": "A/2", "rowIndex": 0, "columnIndex": 1},
          {"path": "B/1", "rowIndex": 1, "columnIndex": 0}
        ],
        "acquisitions": [
          {"id": 0, "name": "Initial", "maximumfieldcount": 4},
          {"id": 1, "name": "24h", "maximumfieldcount": 4}
        ]
      }
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v04

    plate_def = v04.PlateDef(
        name="HCS Experiment",
        columns=[v04.Column(name="1"), v04.Column(name="2"), v04.Column(name="3")],
        rows=[v04.Row(name="A"), v04.Row(name="B")],
        wells=[
            v04.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            v04.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
            v04.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        ],
        acquisitions=[
            v04.Acquisition(id=0, name="Initial", maximumfieldcount=4),
            v04.Acquisition(id=1, name="24h", maximumfieldcount=4),
        ],
    )

    plate = v04.Plate(plate=plate_def)
    ```

    !!! warning "Breaking change from v0.3"
        v0.4 added `rowIndex` and `columnIndex`, **required** for all wells. This enables efficient sparse plate handling without path parsing.

---

### Well Metadata

Wells list the fields of view (images) they contain:

=== "v0.6"

    **Spec JSON (`zarr.json` in well directory):**

    ```json
    {
      "zarr_format": 3,
      "node_type": "group",
      "attributes": {
        "ome": {
          "version": "0.6",
          "well": {
            "images": [
              {"path": "0", "acquisition": 0},
              {"path": "1", "acquisition": 0},
              {"path": "2", "acquisition": 1}
            ]
          }
        }
      }
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v06

    well_def = v06.WellDef(
        images=[
            v06.FieldOfView(path="0", acquisition=0),
            v06.FieldOfView(path="1", acquisition=0),
            v06.FieldOfView(path="2", acquisition=1),
        ]
    )

    well = v06.Well(well=well_def)
    ```

    !!! info "Field of view paths in v0.6"
        Up to v0.5, a field of view's `path` had to be alphanumeric.  v0.6
        follows Zarr's rules for node names instead: letters, digits, `-`, `_`
        and `.` are allowed, but a path **MUST NOT** be empty, consist only of
        periods (like `.` or `..`), or start with `__`.

=== "v0.5"

    **Spec JSON (`zarr.json` in well directory):**

    ```json
    {
      "zarr_format": 3,
      "node_type": "group",
      "attributes": {
        "ome": {
          "version": "0.5",
          "well": {
            "images": [
              {"path": "0", "acquisition": 0},
              {"path": "1", "acquisition": 0},
              {"path": "2", "acquisition": 1}
            ]
          }
        }
      }
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v05

    well_def = v05.WellDef(
        images=[
            v05.FieldOfView(path="0", acquisition=0),
            v05.FieldOfView(path="1", acquisition=0),
            v05.FieldOfView(path="2", acquisition=1),
        ]
    )

    well = v05.Well(well=well_def)
    ```

=== "v0.4"

    **Spec JSON (`.zattrs` in well directory):**

    ```json
    {
      "well": {
        "version": "0.4",
        "images": [
          {"path": "0", "acquisition": 0},
          {"path": "1", "acquisition": 0},
          {"path": "2", "acquisition": 1}
        ]
      }
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v04

    well_def = v04.WellDef(
        images=[
            v04.FieldOfView(path="0", acquisition=0),
            v04.FieldOfView(path="1", acquisition=0),
            v04.FieldOfView(path="2", acquisition=1),
        ]
    )

    well = v04.Well(well=well_def)
    ```

| Field | Requirement | Description |
|-------|-------------|-------------|
| `images` | **MUST** | List of field of view objects |
| `images[].path` | **MUST** | Path to image group |
| `images[].acquisition` | **MUST (if multiple acquisitions exist)** | Links to plate acquisition ID |

### Interactive Example

Modify the parameters below to see how different image configurations are
represented in OME-Zarr:

<ome-explorer preset=3d plate-expanded></ome-explorer>

---

## Working with Collections

OME-Zarr **does not** currently have an official specification for collections of images.

By **Collections** of images, we mean groups of related images, usually sharing a coordinate space,
that do not fit into the plate model.  Examples include:

- Multiple stage positions on single coverslip
- Multiple angles in light sheet microscopy
- Tomographic tilt series
- Jagged or otherwise irregular sets of related images that don't fit the multiscales model

!!! information "Status"

    There is a [long-standing github issue](https://github.com/ome/ngff/issues/31)
    that discusses potential future standards for collections, and
    [RFC-8](https://ngff.openmicroscopy.org/rfc/8/index.html), which covers this
    topic (still under review).  But as of v0.6, there is no official spec for
    collections.

    v0.6 *does* add [scenes](#working-with-scenes), which cover one common
    case: images that share a spatial relationship.

    The [`"bioformats2raw"`
    layout](https://ngff.openmicroscopy.org/0.5/index.html#bf2raw) is a
    _transitional_ solution, internally employed by the
    [bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) tool when
    dumping multiple series (commonly found in image formats supported by
    [bioformats](https://github.com/ome/bioformats)) into a single zarr hierarchy.

This bioformats2raw layout described in the NGFF spec, is described below:

### Directory Structure

=== "v0.6"

    ```
    series.ome.zarr               # One converted fileset from bioformats2raw
        ├── zarr.json             # Contains "bioformats2raw.layout" metadata
        ├── OME                   # Special group for containing OME metadata
        │   ├── zarr.json         # Contains "series" metadata
        │   └── METADATA.ome.xml  # OME-XML file stored within the Zarr fileset
        ├── 0                     # First image in the collection
        ├── 1                     # Second image in the collection
        └── ...
    ```

=== "v0.5"

    ```
    series.ome.zarr               # One converted fileset from bioformats2raw
        ├── zarr.json             # Contains "bioformats2raw.layout" metadata
        ├── OME                   # Special group for containing OME metadata
        │   ├── zarr.json         # Contains "series" metadata
        │   └── METADATA.ome.xml  # OME-XML file stored within the Zarr fileset
        ├── 0                     # First image in the collection
        ├── 1                     # Second image in the collection
        └── ...
    ```

=== "v0.4"

    ```
    series.ome.zarr               # One converted fileset from bioformats2raw
        ├── .zgroup
        ├── .zattrs               # Contains "bioformats2raw.layout" metadata
        ├── OME                   # Special group for containing OME metadata
        │   ├── .zgroup
        │   ├── .zattrs           # Contains "series" metadata
        │   └── METADATA.ome.xml  # OME-XML file stored within the Zarr fileset
        ├── 0                     # First image in the collection
        ├── 1                     # Second image in the collection
        └── ...
    ```

### Metadata

=== "v0.6"

    Unchanged from v0.5 (it's still *transitional*).

    **yaozarrs Code:**

    ```python
    from yaozarrs import v06

    # Root zarr.json
    root_zarr_json = v06.OMEZarrGroupJSON(attributes={"ome": v06.Bf2Raw()})

    # OME/zarr.json
    ome_zarr_json = v06.OMEZarrGroupJSON(
        attributes={"ome": v06.Series(series=["0", "1", "2", "3"])}
    )
    ```

=== "v0.5"

    **yaozarrs Code:**

    ```python
    from yaozarrs import v05

    # Root zarr.json
    root_zarr_json = v05.OMEZarrGroupJSON(attributes={"ome": v05.Bf2Raw()})

    # OME/zarr.json
    ome_zarr_json = v05.OMEZarrGroupJSON(
        attributes={"ome": v05.Series(series=["0", "1", "2", "3"])}
    )
    ```

=== "v0.4"

    **Spec JSON (`.zattrs` at root):**

    ```json
    {
      "bioformats2raw.layout": 3
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v04

    # Root .zattrs
    bf2raw = v04.Bf2Raw()  # layout defaults to 3

    # OME/.zattrs
    series = v04.Series(series=["0", "1", "2", "3"])
    ```

!!! info "Image Location Rules"
    1. If `plate` metadata exists → use plate structure
    2. If `series` attribute exists in the `OME` group → paths must match OME-XML Image element order
    3. Otherwise → consecutively numbered groups: `0/`, `1/`, `2/`...

---

## Working with Scenes

A **Scene** (v0.6 and later) is a group of images that **share a spatial relationship** with each
other.  For example:

- Tiles of a mosaic, placed side by side in a common "stage" coordinate system
- The same sample imaged from several angles (e.g. multiview light sheet)
- Images from different instruments (or modalities), registered onto each other

A scene group has no image data of its own.  Its `zarr.json` holds
`scene` metadata, describing **transformations between the coordinate systems of
its images**, and each image is a normal multiscale image group inside it.

!!! tip "Not sure what a coordinate system is?"
    Scenes build on the coordinate systems and transformations of v0.6.  If those are new to you, start with [Coordinate Systems &
    Transformations](coordinate_systems.md), then come back here.

### Directory Structure

```
mosaic.zarr/
├── zarr.json                  # contains Scene metadata ("attributes.ome.scene")
├── tile_0/                    # a normal multiscale image
│   ├── zarr.json              # contains Image metadata ("attributes.ome.multiscales")
│   ├── 0/                     # Full resolution
│   └── labels/                # Optional labels group
├── tile_1/                    # another multiscale image
│   └── ...
└── coordinateTransformations/ # (optional) transformation parameters stored in Zarr
```

### Scene Metadata

The `scene` object **MUST** contain `coordinateTransformations`, and **MAY**
contain `coordinateSystems` of its own.  Here, two tiles are placed in a shared
`stage` coordinate system, defined by the scene:

**Spec JSON (`zarr.json` at the scene root):**

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "ome": {
      "version": "0.6",
      "scene": {
        "coordinateSystems": [
          {
            "name": "stage",
            "axes": [
              {"name": "y", "type": "space", "unit": "micrometer"},
              {"name": "x", "type": "space", "unit": "micrometer"}
            ]
          }
        ],
        "coordinateTransformations": [
          {
            "type": "translation",
            "translation": [0.0, 0.0],
            "input": {"path": "tile_0", "name": "physical"},
            "output": {"name": "stage"}
          },
          {
            "type": "translation",
            "translation": [0.0, 270.0],
            "input": {"path": "tile_1", "name": "physical"},
            "output": {"name": "stage"}
          }
        ]
      }
    }
  }
}
```

Each `input` and `output` refers to a coordinate system by `name`:

- With a `path`, it's a coordinate system of the image at that `path`
  (relative to the scene group).  Here, `physical` is the [intrinsic coordinate
  system](coordinate_systems.md#the-intrinsic-coordinate-system) of each tile.
- Without a `path`, it's one of the scene's own `coordinateSystems` (here,
  `stage`).

So these two transformations say "a point in `tile_1`'s physical coordinates,
shifted by 270 µm along x, is that point's position on the stage".

**yaozarrs Code:**

```python
from yaozarrs import v06

stage = v06.CoordinateSystem(
    name="stage",
    axes=[
        v06.SpaceAxis(name="y", unit="micrometer"),
        v06.SpaceAxis(name="x", unit="micrometer"),
    ],
)
scene = v06.Scene(
    scene=v06.SceneDef(
        coordinateSystems=[stage],
        coordinateTransformations=[
            v06.TranslationTransformation(
                translation=[0.0, 270.0 * i],
                input={"path": f"tile_{i}", "name": "physical"},
                output={"name": "stage"},
            )
            for i in range(2)
        ],
    )
)
```

??? example "Writing it (with data)"
    [`write_scene`][yaozarrs.write.v06.write_scene] writes the scene group and
    all of its images at once, after checking that the scene and the images fit
    together:

    ```python
    import numpy as np

    from yaozarrs import DimSpec, v06
    from yaozarrs.write.v06 import write_scene

    # each tile is 600x600 pixels, at 0.5 µm per pixel (300 µm wide)
    tile = v06.Image(
        multiscales=[
            v06.Multiscale.from_dims(
                [
                    DimSpec(name="y", scale=0.5, unit="micrometer"),
                    DimSpec(name="x", scale=0.5, unit="micrometer"),
                ],
                coordinate_system="physical",
            )
        ]
    )
    data = np.zeros((600, 600), dtype=np.uint16)
    write_scene("mosaic.zarr", scene, {"tile_0": (tile, data), "tile_1": (tile, data)})
    ```

    For bigger scenes (or to write data piece by piece), see
    [`SceneBuilder`][yaozarrs.write.v06.SceneBuilder].

!!! warning "Scene rules"
    - In a scene, both `input` and `output` **MUST** have a `name`.  A `path`
      is **required** when the coordinate system belongs to an image, and may be
      left out when it's one of the scene's own.
    - All of the coordinate systems (the scene's and its images') **MUST** be
      connected by transformations: there must be a path of one or more
      transformations between *any* two of them (they don't all need to link
      to each other directly).
    - A transformation's parameters **MUST** fit the coordinate systems it
      links (e.g. a `translation` between two 2D coordinate systems has 2
      values).

!!! tip "Which coordinate system is the 'main' one?"
    If a viewer picks a default coordinate system to display a scene in, the
    spec **RECOMMENDS** the **first** of the scene's own `coordinateSystems`.  Likewise, when
    writing a scene, if one coordinate system acts as the common reference for
    the others (like `stage` above), put it first.

??? question "Do I need scene coordinate systems at all?"
    No, they're optional.  A transformation may link one image directly to
    another (both with a `path`), which is natural for registration, where you
    map a "moving" image onto a "fixed" one:

    ```json
    {
      "type": "affine",
      "affine": [[...], [...], [...]],  // a 3x4 matrix, for 3D images
      "input": {"path": "moving_image", "name": "physical"},
      "output": {"path": "fixed_image", "name": "physical"}
    }
    ```

??? question "Where do large transformation parameters go?"
    Some transformations store their parameters in Zarr: an `affine` or
    `rotation` matrix *can* (instead of inline JSON), and the vector field of a
    `displacements` or `coordinates` transformation (e.g. for non-rigid
    registration) *must*, as a multiscale image group.  In a scene, those
    **SHOULD** be stored in a group called `coordinateTransformations`, next to
    the images.

---

## Reference

### Version Comparison Matrix

| Feature | v0.2 | v0.3 | v0.4 | v0.5 | v0.6 |
|---------|:----:|:----:|:----:|:----:|:----:|
| Zarr version | v2 | v2 | v2 | v3 | v3 |
| Axes format | Implicit TCZYX | Strings | Objects | Objects | Objects, in named coordinate systems |
| Axis type field | N/A | N/A | SHOULD | SHOULD | SHOULD |
| Axis unit field | N/A | N/A | SHOULD | SHOULD | SHOULD |
| Coordinate transforms | N/A | N/A | **MUST** | **MUST** | **MUST** |
| Transformation types | N/A | N/A | identity, scale, translation | identity, scale, translation | 12 types |
| Metadata location | `.zattrs` | `.zattrs` | `.zattrs` | `zarr.json` | `zarr.json` |
| OME namespace | N/A | N/A | N/A | `attributes.ome` | `attributes.ome` |
| `dimension_names` | N/A | N/A | N/A | **MUST** | Not required |
| Plate indices | N/A | N/A | **MUST** | **MUST** | **MUST** |
| Scenes | N/A | N/A | N/A | N/A | MAY |

### What changed in each version

| Version | Main changes |
|---------|--------------|
| **v0.3** | Explicit `axes` (as strings) |
| **v0.4** | Axes become objects with a `type` and `unit`; each dataset needs `coordinateTransformations`; plate wells get `rowIndex`/`columnIndex` (v0.4.1 added `bioformats2raw.layout`) |
| **v0.5** | Zarr v3: metadata moves to `zarr.json`, under `attributes.ome` |
| **v0.6** | Axes move into named coordinate systems, transformations get an explicit `input`/`output`, many more transformation types, and scenes ([RFC-5](https://ngff.openmicroscopy.org/rfc/5/index.html)) |

To convert metadata models between v0.4, v0.5 and v0.6, see
[`yaozarrs.convert_version`][].

---

## Additional Resources

- [OME-Zarr Specification](https://ngff.openmicroscopy.org/) - Official specification
- [yaozarrs API Documentation](../API_Reference/yaozarrs.md) - Complete API reference
- [Zarr Format Specification](https://zarr-specs.readthedocs.io/) - Zarr v2 and v3 specs
- [OME Data Model](https://docs.openmicroscopy.org/ome-model/) - Full OME-XML specification
- [GitHub Repository](https://github.com/imaging-formats/yaozarrs) - Source code and issues

---

!!! success "You're ready!"
    You now understand the OME-Zarr specification and how to work with it using yaozarrs. Happy imaging!
