---
icon: material/book-open-variant
title: Guide to OME-Zarr
---

# Yaozzars Guide to OME-Zarr

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

</div>

## What is OME-Zarr?

!!! tip ""
    The official OME-Zarr specification can be found at
    <https://ngff.openmicroscopy.org/>.  In case of
    any discrepancies between this guide and the official spec, the official
    spec takes precedence!

OME-Zarr is a file format specification used by the bioimaging community for
storing multi-dimensional data. It is a "meta-specication", based on the
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

As of v0.5, a single image may have **no less than 2 and no more than 5
dimensions**, and may store multiple resolution levels.

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

    See also: an RFC ("request for comments") proposing a relaxation of this
    restriction: [RFC-3](https://ngff.openmicroscopy.org/rfc/3/index.html)

??? question "What if I have both RGB and optical channels?"
    As of v0.5, there is no formal specification for mixing the concepts
    of RGB image components and conventional "channels" (like optical
    configurations).  You will need to either create a custom group layout
    or flatten them all into a single channel dimension.

### Directory Structure

=== "OME-Zarr v0.5 (Zarr v3)"

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

=== "OME-Zarr v0.4 (Zarr v2)"

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

---

### Axes

Axes define the dimensions of your image data. As of v0.4, axes are **objects**
with `name`, and optional `type` and/or `unit`:

!!! important "Axis Constraints"
    Constraints for image axes in OME-Zarr are the same in v0.4 and v0.5:

    - **MUST** have 2-5 dimensions total
    - **MUST** have 2-3 spatial axes
    - **MAY** have 0-1 time axis
    - **MAY** have 0-1 channel axis
    - **Ordering enforced**: time → channel/custom → space

    In practice, this limits valid axis combinations to: `[T][C][Z] Y X`  
    *(though no explicit restriction is placed on naming conventions)*

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

---

### Coordinate Transformations

Starting in v0.4, **every dataset MUST include coordinate transformations** that
map data coordinates to physical coordinates.  Coordinate transforms are where
you would specify physical units (micrometers, seconds), multi-resolution scales,
as well as stage positions and spatial offsets for registration.

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
        coordinateTransformations=[
            v04.ScaleTransformation(scale=[1.0, 0.5, 0.1, 0.1])
        ]
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
            v04.TranslationTransformation(translation=[0.0, 0.0, 100.0, 200.0])
        ]
    )
    ```

    !!! warning "Transformation Rules"
        - **MUST** have exactly one scale transformation per dataset
        - **MAY** have at most one translation transformation
        - If translation exists, it **MUST** come after scale
        - Transformation length **MUST** match number of axes

=== "v0.5"

    Identical to v0.4, just stored under `attributes.ome` namespace.

    **yaozarrs Code (same as v0.4):**

    ```python
    from yaozarrs import v05

    dataset = v05.Dataset(
        path="0",
        coordinateTransformations=[
            v05.ScaleTransformation(scale=[1.0, 0.5, 0.1, 0.1]),
            v05.TranslationTransformation(translation=[0.0, 0.0, 100.0, 200.0])
        ]
    )
    ```

    !!! info "v0.5 Additional Requirement"
        In v0.5, each array's `zarr.json` **MUST** include `dimension_names` matching the axes:

        ```json
        {
          "dimension_names": ["c", "z", "y", "x"]
        }
        ```

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
    metadata.

??? example "Label Structure and Code"

    === "OME-Zarr v0.5 (Zarr v3)"
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
            │   ├── zarr.json    # Label metadata ("attributes.ome.multiscales", "attributes.ome.image_label")
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
              "multiscales": [...],
              "image_label": {
                "version": "0.5",
                "colors": [
                  {"label_value": 1, "rgba": [255, 0, 0, 255]},
                  {"label_value": 2, "rgba": [0, 255, 0, 255]}
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
        from yaozarrs import v05

        # Label metadata stored at labels/nuclei/zarr.json
        label_image = v05.LabelImage(
            multiscales=[...],  # Same structure as regular image
            image_label=v05.ImageLabel(
                colors=[
                    v05.LabelColor(label_value=1, rgba=[255, 0, 0, 255]),
                    v05.LabelColor(label_value=2, rgba=[0, 255, 0, 255])
                ],
                source=v05.LabelSource(image="../../")
            )
        )
        ```

    === "OME-Zarr v0.4 (Zarr v2)"

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
        from yaozarrs import v04

        # Label metadata stored at labels/nuclei/.zattrs
        label_image = v04.LabelImage(
            multiscales=[...],  # Same structure as regular image
            image_label=v04.ImageLabel(
                colors=[
                    v04.LabelColor(label_value=1, rgba=[255, 0, 0, 255]),
                    v04.LabelColor(label_value=2, rgba=[0, 255, 0, 255])
                ],
                source=v04.LabelSource(image="../../")
            )
        )
        ```

    !!! warning "Labels must use integer dtype"
        Validation will fail if label arrays use float dtypes. Use `uint8`, `uint16`, `uint32`, or `int32`.

---

## Working with Plates

A **Plate** represents multi-well plate data from high-content screening (HCS) experiments. The hierarchy is:

**Plate** → **Rows/Columns** → **Wells** → **Fields of View (Images)**

Each well can contain multiple fields of view (FOVs) across multiple acquisitions (timepoints).

### Directory Structure

=== "OME-Zarr v0.5 (Zarr v3)"

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

=== "OME-Zarr v0.4 (Zarr v2)"

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
        ],
      }
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v04

    plate_def = v04.PlateDef(
        name="HCS Experiment",
        columns=[
            v04.Column(name="1"),
            v04.Column(name="2"),
            v04.Column(name="3")
        ],
        rows=[
            v04.Row(name="A"),
            v04.Row(name="B")
        ],
        wells=[
            v04.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            v04.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
            v04.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        ],
        acquisitions=[
            v04.Acquisition(id=0, name="Initial", maximumfieldcount=4),
            v04.Acquisition(id=1, name="24h", maximumfieldcount=4),
        ]
    )

    plate = v04.Plate(plate=plate_def)
    ```

    !!! warning "Breaking change from v0.3"
        In v0.4, `rowIndex` and `columnIndex` became **required** for all wells. This enables efficient sparse plate handling without path parsing.

=== "v0.5"

    Same structure as v0.4, stored under `attributes.ome` in `zarr.json`:

    **Spec JSON (`zarr.json` at plate root):**

    ```json
    {
      "attributes": {
        "ome": {
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
            v05.Column(name="3")
        ],
        rows=[  # must have at least 1 row
            v05.Row(name="A"),
            v05.Row(name="B")
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
        field_count=4  # max FOV per well
    )

    plate = v05.Plate(plate=plate_def)

    # Create full zarr.json
    zarr_json = v05.OMEZarrGroupJSON(attributes={"ome": plate})
    json_str = zarr_json.model_dump_json(indent=2, exclude_unset=True)
    ```

---

### Well Metadata

Wells list the fields of view (images) they contain:

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
    that discusses potential future standards for collections, and a (currently
    pending) [pull request for RFC-8](https://github.com/ome/ngff/pull/343), which covers this topic.
    But as of v0.5 and January 2026, there is no official spec.

    The [`"bioformats2raw"`
    layout](https://ngff.openmicroscopy.org/0.5/index.html#bf2raw) is a
    _transitional_ solution, internally employed by the
    [bioformats2raw](https://github.com/glencoesoftware/bioformats2raw) tool when
    dumping multiple series (commonly found in image formats supported by
    [bioformats](https://github.com/ome/bioformats)) into a single zarr hierarchy.

This bioformats2raw layout described in the NGFF spec, is described below:

### Directory Structure

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

### Metadata

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

=== "v0.5"

    **yaozarrs Code:**

    ```python
    from yaozarrs import v05

    # Root zarr.json
    root_zarr_json = v05.OMEZarrGroupJSON(
      attributes={"ome": v05.Bf2Raw()}
    )

    # OME/zarr.json
    ome_zarr_json = v05.OMEZarrGroupJSON(
        attributes={"ome": v05.Series(series=["0", "1", "2", "3"])}
    )
    ```

!!! info "Image Location Rules"
    1. If `plate` metadata exists → use plate structure
    2. If `series` attribute exists in `OME/.zattrs` → paths must match OME-XML Image element order
    3. Otherwise → consecutively numbered groups: `0/`, `1/`, `2/`...

---

### When to Use Collections vs. Plates

| Scenario | Use Collection | Use Plate |
|----------|:--------------:|:---------:|
| Multiple FOVs on coverslip | :white_check_mark: | :x: |
| Irregular stage positions | :white_check_mark: | :x: |
| Time-lapse split across files | :white_check_mark: | :x: |
| Multi-well HCS experiment | :x: | :white_check_mark: |
| Regular grid with well labels | :x: | :white_check_mark: |

!!! tip "Rule of thumb"
    If your data has **rows and columns** (like A1, B2, etc.), use a **Plate**.
    If it's just **multiple related images**, use a **Collection**.

---

## Reference

### Version Comparison Matrix

| Feature | v0.2 | v0.3 | v0.4 | v0.5 |
|---------|:----:|:----:|:----:|:----:|
| Zarr version | v2 | v2 | v2 | v3 |
| Axes format | Implicit TCZYX | Strings | Objects | Objects |
| Axis type field | N/A | N/A | SHOULD | SHOULD |
| Axis unit field | N/A | N/A | SHOULD | SHOULD |
| Coordinate transforms | N/A | N/A | **MUST** | **MUST** |
| Metadata location | `.zattrs` | `.zattrs` | `.zattrs` | `zarr.json` |
| OME namespace | N/A | N/A | N/A | `attributes.ome` |
| `dimension_names` | N/A | N/A | N/A | **MUST** |
| Plate indices | Optional | Optional | **MUST** | **MUST** |

### Breaking Changes Quick Reference

| Migration | Key Breaking Change | Impact |
|-----------|---------------------|--------|
| **v0.2 → v0.3** | Axes must be explicit strings | :warning: Moderate - add `axes` field |
| **v0.3 → v0.4** | Axes become objects + coordinate transforms required | :warning::warning: Major - restructure metadata |
| **v0.4 → v0.5** | Zarr v3 file structure + OME namespace | :warning::warning::warning: Critical - completely different storage |

---

## Additional Resources

- [OME-Zarr Specification](https://ngff.openmicroscopy.org/) - Official specification
- [yaozarrs API Documentation](API_Reference/yaozarrs.md) - Complete API reference
- [Zarr Format Specification](https://zarr-specs.readthedocs.io/) - Zarr v2 and v3 specs
- [OME Data Model](https://docs.openmicroscopy.org/ome-model/) - Full OME-XML specification
- [GitHub Repository](https://github.com/tlambert03/yaozarrs) - Source code and issues

---

!!! success "You're ready!"
    You now understand the OME-Zarr specification and how to work with it using yaozarrs. Happy imaging!
