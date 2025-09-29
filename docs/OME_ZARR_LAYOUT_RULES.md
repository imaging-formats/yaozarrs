# OME-ZARR Storage Layout Rules

This document outlines the directory structure and validation requirements for
OME-ZARR hierarchies based on the official specifications.

## Version Differences

### OME-ZARR v0.4

- Uses Zarr v2 specification
- Metadata in `.zattrs` and `.zgroup` files
- Groups defined by `.zgroup` files
- Arrays have individual `.zattrs` files

### OME-ZARR v0.5

- Uses Zarr v3 specification
- Metadata in `zarr.json` files
- Single `zarr.json` per group/array
- Node type explicitly specified

## Group Types and Directory Layouts

### 1. Image Groups

#### v0.5 Structure

```
my_image/
├── zarr.json          # Group metadata with "multiscales"
├── 0/                 # Resolution level 0 (highest resolution)
│   └── zarr.json      # Array metadata
├── 1/                 # Resolution level 1
│   └── zarr.json
└── labels/            # Optional labels group
    └── ...
```

#### v0.4 Structure

```
my_image/
├── .zgroup            # Marks as Zarr group
├── .zattrs            # Contains "multiscales" metadata
├── 0/                 # Resolution level 0
│   ├── .zarray        # Array structure
│   └── .zattrs        # Array metadata
├── 1/                 # Resolution level 1
│   ├── .zarray
│   └── .zattrs
└── labels/            # Optional labels group
    └── ...
```

**Requirements:**

- MUST have `multiscales` metadata
- Resolution levels numbered sequentially (0, 1, 2...)
- Ordered from highest to lowest resolution
- Each level is a complete Zarr array
- MAY have `omero` metadata for rendering
- MAY have `labels` subdirectory

### 2. Plate Groups

#### v0.5 Structure

```
my_plate/
├── zarr.json          # Group metadata with "plate"
├── A/                 # Row A
│   ├── 1/             # Column 1 well
│   │   ├── zarr.json  # Well metadata
│   │   ├── 0/         # Field 0 (Image group)
│   │   └── 1/         # Field 1 (Image group)
│   └── 2/             # Column 2 well
└── B/                 # Row B
    └── 1/             # Column 1 well
```

#### v0.4 Structure

```
my_plate/
├── .zgroup
├── .zattrs            # Contains "plate" metadata
├── A/
│   ├── 1/
│   │   ├── .zgroup
│   │   ├── .zattrs    # Contains "well" metadata
│   │   ├── 0/         # Field 0 (Image group)
│   │   └── 1/         # Field 1 (Image group)
│   └── 2/
└── B/
    └── 1/
```

**Requirements:**

- MUST have `plate` metadata with:
  - `columns` list (names MUST be alphanumeric)
  - `rows` list (names MUST be alphanumeric)
  - `wells` list with path, rowIndex, columnIndex
- MAY have `acquisitions` list
- Directory structure MUST match `wells` paths
- Row/column names MUST be case-sensitive and unique

### 3. Well Groups

**Requirements:**

- MUST have `well` metadata with `images` list
- Each image entry has `path` to field directory
- MAY have `acquisition` reference
- Field directories are complete Image groups

### 4. Labels Groups

#### Structure

```
my_image/labels/
├── zarr.json/.zattrs  # Contains "labels" list
├── cell_seg/          # Label image (name from labels list)
│   ├── zarr.json      # Image + "image-label" metadata
│   ├── 0/             # Resolution levels
│   └── 1/
└── nuclei_seg/        # Another label image
    └── ...
```

**Requirements:**

- MUST be located within an Image group as `labels/` subdirectory
- MUST have `labels` metadata listing label image names
- Each label image is a complete Image group
- Each label image MUST have `image-label` metadata
- Label arrays MUST use integer data types
- MAY have `colors` and `properties` in `image-label`
- MAY have `source` reference to parent image

### 5. Series Collection Groups

#### v0.5 Structure

```
converted_multiseries/
├── zarr.json          # Contains "series" list
├── 0/                 # First image series
│   ├── zarr.json      # Image metadata
│   ├── 0/
│   └── 1/
├── 1/                 # Second image series
│   └── ...
└── OME/               # Optional OME metadata
    └── METADATA.ome.xml
```

**Requirements:**

- MUST have `series` metadata listing image paths
- Each series path is a complete Image group
- MAY have OME metadata directory

### 6. Bioformats2raw Layout Groups

#### Structure

```
converted_file/
├── zarr.json/.zattrs  # Contains "bioformats2raw.layout"
├── 0/                 # First image (by convention)
│   └── zarr.json      # Image metadata
├── 1/                 # Second image
│   └── zarr.json
├── 2/                 # Third image
│   └── zarr.json
└── OME/               # OME metadata
    └── METADATA.ome.xml
```

**Requirements:**

- MUST have `bioformats2raw.layout` metadata
- Images in numbered directories (filesystem discovery)
- SHOULD have OME metadata directory

## Validation Rules

### Path Validation

- Column and row names MUST contain only alphanumeric characters
- Names MUST be case-sensitive
- Names MUST NOT be duplicates within same list
- Paths should avoid case-insensitive filesystem collisions

### Array Requirements

- 2-5 dimensional arrays
- Dimension order: time (optional) → channel (optional) → spatial axes
- Spatial axes ordered "zyx" for anisotropic data
- Dimensions MUST match "axes" metadata length
- Label images MUST use integer data types: `uint8`, `int8`, `uint16`, `int16`,
  `uint32`, `int32`, `uint64`, `int64`

### Metadata Consistency

- OME-ZARR version MUST be consistent within hierarchy
- Multiscale datasets MUST have same number of resolution levels
- Metadata references MUST match actual filesystem structure
- Axes names MUST be unique
- Axis types SHOULD be "space", "time", or "channel"

### File Existence

- v0.5: `zarr.json` MUST exist for all groups and arrays
- v0.4: `.zgroup` MUST exist for groups, `.zarray` for arrays
- v0.4: `.zattrs` files contain metadata
- Multiscale images MUST have all referenced resolution levels
- Referenced paths in metadata MUST exist on filesystem

### Coordinate Transformations

- MUST include at least one `scale` transformation
- Optional `translation` MUST follow `scale`
- Transformation arrays MUST match axes length
- Only `translation` and `scale` types allowed

### Cross-References

- Well paths MUST match plate row/column definitions
- Image paths MUST be valid within hierarchy
- Label source references MUST point to valid images
- Acquisition references MUST match defined acquisitions

## Array Discovery Methods

### Explicit Discovery (Metadata-driven)

- **Images**: Listed in `multiscales.datasets[].path`
- **Plates**: Wells listed in `plate.wells[].path`
- **Wells**: Images listed in `well.images[].path`
- **Labels**: Listed in `labels` array
- **Series**: Listed in `series` array

### Filesystem Discovery

- **Bioformats2raw**: Numbered directories (0/, 1/, 2/, etc.)
- **Labels directory**: Explore for subdirectories

## Error Conditions

### Critical Errors (MUST fail validation)

- Missing required metadata files
- Referenced paths don't exist on filesystem
- Invalid data types for label arrays
- Inconsistent OME-ZARR versions
- Malformed JSON in metadata files
- Missing required metadata keys

### Warnings (SHOULD report but may not fail)

- Non-standard axis ordering
- Missing optional metadata
- Case-insensitive path conflicts
- Non-alphanumeric row/column names
- Missing coordinate transformations
