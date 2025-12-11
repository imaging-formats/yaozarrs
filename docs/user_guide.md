---
icon: lucide/book-open
title: Yaozzars Guide to OME-NGFF
---

!!! tip "What you'll learn"
    This guide demystifies the **OME-NGFF (OME-Zarr)** specification and shows
    you how to work with it using yaozarrs. Whether you're new to cloud-native
    bioimaging or an experienced developer, you'll find practical guidance for
    creating, validating, and understanding OME-Zarr data.

---

## Choose Your Path

<script type="module" src="/javascripts/ome_explorer.js"></script>
<div class="grid cards" markdown>

- :lucide-box:{ .lg .middle } **I have images**

    ---
    Single images, z-stacks, timelapses, or multi-channel confocal data

    [:octicons-arrow-right-24: Go to Images](#working-with-images)

- :lucide-grid-3x2:{ .lg .middle } **I have plate data**

    ---
    Multi-well plates from high-content screening (HCS) experiments

    [:octicons-arrow-right-24: Go to Plates](#working-with-plates)

- :lucide-tags:{ .lg .middle } **I have image annotations**

    ---

    Segmentation masks, annotation labels, and regions of interest (ROIs)

    [:octicons-arrow-right-24: Go to Labels](#labels-segmentation-masks)

- :lucide-folders:{ .lg .middle } **I have multiple images**

    ---

    Collections of related images (multi-FOV, stage positions, split files)

    [:octicons-arrow-right-24: Go to Collections](#working-with-collections)

</div>

---

## Quick Start: Version Selection

!!! question "Which version should I use?"

    === "New Projects (Recommended: v0.4)"

        **Use v0.4 if:**

        - You're starting a new project
        - You need maximum compatibility with existing tools
        - You're working with Zarr v2

        ```python
        from yaozarrs import v04
        ```

    === "Cloud-Scale Projects (v0.5)"

        **Use v0.5 if:**

        - You need sharding for massive datasets
        - You're adopting Zarr v3
        - You want future-proof architecture

        ```python
        from yaozarrs import v05
        ```

### Breaking Changes Quick Reference

| Migration | Key Breaking Change | Impact |
|-----------|---------------------|--------|
| **v0.2 → v0.3** | Axes must be explicit strings | :warning: Moderate - add `axes` field |
| **v0.3 → v0.4** | Axes become objects + coordinate transforms required | :warning::warning: Major - restructure metadata |
| **v0.4 → v0.5** | Zarr v3 file structure + OME namespace | :warning::warning::warning: Critical - completely different storage |

---

## Working with Images

An **Image** is the fundamental building block of OME-NGFF.

As of v0.5, a single image may have **no less than 2 and no more than 5
dimensions**, and may store multiple resolution levels.

- **Spatial dimensions**: X, Y, optionally Z
- **Time**: T (temporal axis)
- **Channels**: C (fluorescence channels, RGB, etc.)

??? question "What if I have more than 5 dimensions?"
    While it is common to have datasets with more than 5 dimensions (e.g.,
    different stage positions in a shared coordinate space, angles in light
    sheet microscopy, etc.), there is currently no formal specification for more
    than 5 dimensions in OME-NGFF.  You may use the transitional
    `bioformats2raw.layout` to store multiple images in a single zarr group.
    See [Working with Collections](#working-with-collections)

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

    - **<=v0.4**: Metadata in root of `.zattrs` files
    - **>=v0.5**: Metadata in `zarr.json` under `attributes.ome` namespace

---

### Axes

Axes define the dimensions of your image data. The specification evolved significantly across versions:

=== "v0.4"

    Axes are **objects** with `name`, `type`, and optional `unit`:

    **Spec JSON:**

    ```json
    {
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

    **Axis Constraints:**

    - **MUST** have 2-5 dimensions total
    - **MUST** have 2-3 space axes (X, Y, optionally Z)
    - **MAY** have 0-1 time axis
    - **MAY** have 0-1 channel axis
    - **Ordering enforced**: time → channel → space

=== "v0.5"

    Same structure as v0.4, but stored under `attributes.ome` namespace:

    **Spec JSON:**

    ```json
    {
      "attributes": {
        "ome": {
          "axes": [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
          ]
        }
      }
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

    !!! info "Available Axis Types"

        - `SpaceAxis` - X, Y, Z spatial dimensions
        - `TimeAxis` - T temporal dimension
        - `ChannelAxis` - C channel dimension
        - `CustomAxis` - Any other dimension type

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
represented in OME-NGFF:

<ome-explorer preset=5d></ome-explorer>

### Labels (Segmentation Masks)

Labels are specialized images with integer dtype representing segmentation masks (nuclei, cells, regions of interest, etc.).

??? example "Label Structure and Code"

    **Directory Structure:**
    ```
    image.zarr/
    ├── 0/, 1/, 2/           # Image pyramid
    └── labels/
        ├── nuclei/          # Label image (integer dtype)
        │   ├── .zattrs      # Label metadata
        │   ├── 0/           # Full resolution labels
        │   └── 1/           # Downsampled labels
        └── cells/
            └── ...
    ```

    **Labels Group Metadata:**
    ```json
    {
      "labels": ["nuclei", "cells"]
    }
    ```

    **Label Image Metadata (`.zattrs` in labels/nuclei/):**
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
            version="0.4",
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

```
plate.zarr/
├── .zattrs              # Plate metadata
├── A/                   # Row A
│   ├── 1/               # Well A1
│   │   ├── .zattrs      # Well metadata
│   │   ├── 0/           # Field 0 (Image with multiscales)
│   │   │   ├── .zattrs  # Image metadata
│   │   │   ├── 0/       # Full resolution
│   │   │   └── 1/       # Downsampled
│   │   ├── 1/           # Field 1
│   │   └── labels/      # Optional segmentation
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
        "field_count": 4
      }
    }
    ```

    **yaozarrs Code:**

    ```python
    from yaozarrs import v04

    plate_def = v04.PlateDef(
        version="0.4",
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
        ],
        field_count=4
    )

    plate = v04.Plate(plate=plate_def)
    ```

    !!! warning "Breaking change from v0.3"
        In v0.4, `rowIndex` and `columnIndex` became **required** for all wells. This enables efficient sparse plate handling without path parsing.

=== "v0.5"

    Same structure as v0.4, stored under `attributes.ome` in `zarr.json`:

    **yaozarrs Code:**

    ```python
    from yaozarrs import v05

    plate_def = v05.PlateDef(
        version="0.4",  # Note: still uses "0.4" internally
        name="HCS Experiment",
        columns=[
            v05.Column(name="1"),
            v05.Column(name="2"),
            v05.Column(name="3")
        ],
        rows=[
            v05.Row(name="A"),
            v05.Row(name="B")
        ],
        wells=[
            v05.PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            v05.PlateWell(path="A/2", rowIndex=0, columnIndex=1),
            v05.PlateWell(path="B/1", rowIndex=1, columnIndex=0),
        ],
        acquisitions=[
            v05.Acquisition(id=0, name="Initial", maximumfieldcount=4),
            v05.Acquisition(id=1, name="24h", maximumfieldcount=4),
        ],
        field_count=4
    )

    plate = v05.Plate(plate=plate_def)

    # Create full zarr.json
    zarr_json = v05.OMEZarrGroupJSON(attributes={"ome": plate})
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
    version="0.4",
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
| `images[].acquisition` | **SHOULD** | Links to plate acquisition ID |

---

## Working with Collections

A **Collection** (bioformats2raw layout) groups multiple related images that don't fit the plate model. Use cases:

- Multiple stage positions on single coverslip
- Split file series (z-stacks across files)
- Multi-timepoint acquisitions stored separately
- Any multi-image dataset without regular grid structure

### Directory Structure

```
collection.zarr/
├── .zattrs                  # {"bioformats2raw.layout": 3}
├── OME/
│   ├── .zattrs              # {"series": ["0", "1", "2"]}
│   └── METADATA.ome.xml     # Complete OME-XML metadata
├── 0/                       # First image
│   ├── .zattrs              # multiscales
│   └── 0/, 1/               # Pyramid levels
├── 1/                       # Second image
└── 2/
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

    # Marker at root
    bf2raw = v04.Bf2Raw()  # layout defaults to 3

    # Optionally specify series order in OME/.zattrs
    series = v04.Series(series=["0", "1", "2", "3"])
    ```

    !!! info "Image Location Rules"
        1. If `plate` metadata exists → use plate structure
        2. If `series` attribute exists in `OME/.zattrs` → paths must match OME-XML Image element order
        3. Otherwise → consecutively numbered groups: `0/`, `1/`, `2/`...

=== "v0.5"

    **yaozarrs Code:**

    ```python
    from yaozarrs import v05

    # Root zarr.json
    bf2raw = v05.Bf2Raw()

    # OME/zarr.json
    series = v05.Series(series=["0", "1", "2", "3"])

    # Create documents
    root_zarr_json = v05.OMEZarrGroupJSON(attributes={"ome": bf2raw})
    ome_zarr_json = v05.OMEZarrGroupJSON(attributes={"ome": series})
    ```

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

## Validation

yaozarrs provides comprehensive validation at multiple levels:

### Validate JSON Documents

```python
import yaozarrs

# Validate any OME JSON string
json_str = '{"multiscales": [...]}'
obj = yaozarrs.validate_ome_json(json_str)

# Auto-detects version and type
print(type(obj))  # v05.Image or v04.Image

# Validate against specific version
obj = yaozarrs.validate_ome_json(json_str, cls=yaozarrs.v05.Image)
```

### Validate Python Objects

```python
# Validate arbitrary dict
obj = yaozarrs.validate_ome_object({
    "version": "0.5",
    "multiscales": [...]
})

print(obj)  # Returns typed model
```

### Validate Zarr Stores

Validates both metadata **and** store structure:

```python
# Validate local or remote store
yaozarrs.validate_zarr_store("path/to/image.zarr")
yaozarrs.validate_zarr_store("https://example.com/data.zarr")
yaozarrs.validate_zarr_store("s3://bucket/image.zarr")
```

**Validation checks:**

- :white_check_mark: Metadata structure and types
- :white_check_mark: Dataset paths exist
- :white_check_mark: Arrays have correct dimensions
- :white_check_mark: `dimension_names` match axes (v0.5)
- :white_check_mark: Label arrays use integer dtype
- :white_check_mark: Well/plate indices are valid
- :white_check_mark: Resolution levels properly ordered

### CLI Validation

```bash
# Validate any URI
yaozarrs validate path/to/image.zarr

# Or use uvx without install
uvx "yaozarrs[io]" validate https://example.com/data.zarr
```

**Example output:**

```
✓ Valid OME-Zarr store
  Version: 0.5
  Type: Image
```

---

## Loading and Reading Data

### Load from URI

```python
import yaozarrs

# Load OME metadata from any source
obj = yaozarrs.from_uri("path/to/image.zarr")
obj = yaozarrs.from_uri("https://example.com/data.zarr")
obj = yaozarrs.from_uri("s3://bucket/image.zarr")

# Access typed metadata
print(obj.attributes.ome.multiscales[0].axes)
```

### Open Zarr Groups

```python
from yaozarrs import open_group

# Open with yaozarrs' minimal zarr implementation
group = open_group("https://example.com/data.zarr")

# Access arrays
array = group['0']
print(array.shape)  # (3, 50, 512, 512)

# Inspect OME metadata
metadata = group.ome_metadata()
print(metadata.multiscales[0].name)

# Convert to zarr-python or tensorstore
zarr_array = array.to_zarr_python()  # requires zarr
ts_array = array.to_tensorstore()    # requires tensorstore
```

---

## Practical Examples

### Example 1: Basic 3D Confocal Image

??? example "Scenario: CZYX confocal stack, 3 channels, 50 z-slices"

    === "v0.4"

        ```python
        from yaozarrs import DimSpec, v04

        # Define dimensions
        dims = [
            DimSpec(name="c", size=3),
            DimSpec(name="z", size=50, scale=0.3, unit="micrometer", scale_factor=1.0),
            DimSpec(name="y", size=2048, scale=0.1, unit="micrometer"),
            DimSpec(name="x", size=2048, scale=0.1, unit="micrometer"),
        ]

        # Create multiscale with 3 levels
        multiscale = v04.Multiscale.from_dims(dims, name="confocal", n_levels=3)
        image = v04.Image(multiscales=[multiscale])

        # Add OMERO rendering
        omero = v04.Omero(
            channels=[
                v04.OmeroChannel(label="DAPI", color="0000FF"),
                v04.OmeroChannel(label="GFP", color="00FF00"),
                v04.OmeroChannel(label="mCherry", color="FF0000"),
            ]
        )
        image.omero = omero

        # Export
        print(image.model_dump_json(exclude_unset=True, indent=2))
        ```

    === "v0.5"

        ```python
        from yaozarrs import DimSpec, v05

        # Define dimensions (same as v0.4)
        dims = [
            DimSpec(name="c", size=3),
            DimSpec(name="z", size=50, scale=0.3, unit="micrometer", scale_factor=1.0),
            DimSpec(name="y", size=2048, scale=0.1, unit="micrometer"),
            DimSpec(name="x", size=2048, scale=0.1, unit="micrometer"),
        ]

        # Create multiscale
        multiscale = v05.Multiscale.from_dims(dims, name="confocal", n_levels=3)
        image = v05.Image(multiscales=[multiscale])

        # Wrap in zarr.json
        zarr_json = v05.OMEZarrGroupJSON(attributes={"ome": image})
        print(zarr_json.model_dump_json(exclude_unset=True, indent=2))
        ```

### Example 2: Multi-FOV Coverslip

??? example "Scenario: 12 fields of view across single coverslip with stage positions"

    ```python
    from yaozarrs import DimSpec, v04

    # Create bioformats2raw collection marker
    bf2raw = v04.Bf2Raw()
    series = v04.Series(series=[str(i) for i in range(12)])

    # Base dimensions for each FOV
    base_dims = [
        DimSpec(name="c", size=2),
        DimSpec(name="y", size=1024, scale=0.1, unit="micrometer"),
        DimSpec(name="x", size=1024, scale=0.1, unit="micrometer"),
    ]

    # FOV 0 at origin
    fov0_dims = [
        DimSpec(name="c", size=2),
        DimSpec(name="y", size=1024, scale=0.1, unit="micrometer", translation=0.0),
        DimSpec(name="x", size=1024, scale=0.1, unit="micrometer", translation=0.0),
    ]
    multiscale0 = v04.Multiscale.from_dims(fov0_dims, name="FOV_0", n_levels=2)
    image0 = v04.Image(multiscales=[multiscale0])

    # FOV 1 at X=500, Y=200 micrometers
    fov1_dims = [
        DimSpec(name="c", size=2),
        DimSpec(name="y", size=1024, scale=0.1, unit="micrometer", translation=200.0),
        DimSpec(name="x", size=1024, scale=0.1, unit="micrometer", translation=500.0),
    ]
    multiscale1 = v04.Multiscale.from_dims(fov1_dims, name="FOV_1", n_levels=2)
    image1 = v04.Image(multiscales=[multiscale1])

    # Export each FOV to 0/.zattrs, 1/.zattrs, etc.
    # ... create remaining FOVs with appropriate translations
    ```

### Example 3: 96-Well Plate

??? example "Scenario: 96-well plate, 4 fields per well, 2 acquisitions"

    ```python
    from yaozarrs import v04

    # Create plate structure
    columns = [v04.Column(name=str(i)) for i in range(1, 13)]  # 1-12
    rows = [v04.Row(name=chr(65 + i)) for i in range(8)]  # A-H

    # Generate wells (sparse example - only populate some wells)
    wells = []
    for row_idx, row_name in enumerate(['A', 'B', 'C', 'D']):
        for col_idx in range(6):  # First 6 columns
            wells.append(
                v04.PlateWell(
                    path=f"{row_name}/{col_idx + 1}",
                    rowIndex=row_idx,
                    columnIndex=col_idx
                )
            )

    plate_def = v04.PlateDef(
        name="Drug Screen Plate 1",
        columns=columns,
        rows=rows,
        wells=wells,
        acquisitions=[
            v04.Acquisition(id=0, name="Initial", maximumfieldcount=4),
            v04.Acquisition(id=1, name="24h treatment", maximumfieldcount=4),
        ],
        field_count=4
    )

    plate = v04.Plate(plate=plate_def)

    # Create well metadata (for well A/1)
    well_def = v04.WellDef(
        images=[
            # 4 fields from initial acquisition
            v04.FieldOfView(path="0", acquisition=0),
            v04.FieldOfView(path="1", acquisition=0),
            v04.FieldOfView(path="2", acquisition=0),
            v04.FieldOfView(path="3", acquisition=0),
            # 4 fields from 24h acquisition
            v04.FieldOfView(path="4", acquisition=1),
            v04.FieldOfView(path="5", acquisition=1),
            v04.FieldOfView(path="6", acquisition=1),
            v04.FieldOfView(path="7", acquisition=1),
        ]
    )

    well = v04.Well(well=well_def)
    ```

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

### yaozarrs API Quick Reference

```python
import yaozarrs
from yaozarrs import v04, v05, DimSpec

# Validation
yaozarrs.validate_ome_json(json_str)           # Auto-detect version
yaozarrs.validate_ome_object(dict_obj)         # Validate dict
yaozarrs.validate_zarr_store(uri)              # Validate complete store

# Loading
yaozarrs.from_uri(uri)                         # Load metadata from URI
yaozarrs.open_group(uri)                       # Open zarr group

# Export
obj.model_dump_json(exclude_unset=True, indent=2)
```

## Additional Resources

- [OME-NGFF Specification](https://ngff.openmicroscopy.org/) - Official specification
- [yaozarrs API Documentation](api/index.md) - Complete API reference
- [Zarr Format Specification](https://zarr-specs.readthedocs.io/) - Zarr v2 and v3 specs
- [OME Data Model](https://docs.openmicroscopy.org/ome-model/) - Full OME-XML specification
- [GitHub Repository](https://github.com/tlambert03/yaozarrs) - Source code and issues

---

!!! success "You're ready!"
    You now understand the OME-NGFF specification and how to work with it using yaozarrs. Happy imaging!
