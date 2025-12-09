# OME-NGFF Specification: Complete Guide Across Versions 0.2–0.5

The **OME-NGFF (OME-Zarr)** specification defines cloud-native storage for bioimaging data using Zarr arrays. Version **0.5** represents a major architectural shift to Zarr v3, while versions 0.2–0.4 use Zarr v2. For new projects, **0.4 is the most widely supported** in current tooling, though 0.5 adoption is increasing. This report organizes all concepts by data type—image, plate, and collection—to support documentation that helps newcomers understand core patterns while enabling experts to quickly find version-specific details.

---

## Three core data types define OME-NGFF

OME-NGFF organizes bioimaging data into three primary structures: **images** (including multiscale pyramids), **plates** (multi-well HCS data), and **collections** (grouped filesets via bioformats2raw). Each type has distinct metadata requirements that evolved across versions. Understanding these categories prevents confusion when choosing how to store different experimental scenarios.

| Data Type | Purpose | Version Introduced |
|-----------|---------|-------------------|
| **Image (multiscales)** | Single/multi-resolution images with axes | 0.1 (refined 0.3+) |
| **Plate/Well** | High-content screening multi-well plates | 0.1 (refined 0.4) |
| **Collection (bioformats2raw)** | Multi-image filesets with OME-XML | Formalized 0.4 |
| **Labels** | Segmentation masks as integer arrays | 0.2+ |

---

## Image storage: The multiscales specification

### Directory structure fundamentals

Every OME-NGFF image is a Zarr group containing one or more resolution levels as nested arrays. The structure changed significantly between **Zarr v2** (0.2–0.4) and **Zarr v3** (0.5):

**Zarr v2 (versions 0.2–0.4):**

```
image.zarr/
├── .zgroup                    # {"zarr_format": 2}
├── .zattrs                    # Contains "multiscales", "omero"
├── 0/                         # Full resolution array
│   ├── .zarray                # Array metadata (shape, chunks, dtype)
│   └── t/c/z/y/x              # Chunk files with "/" separator
├── 1/                         # 2x downsampled
└── n/                         # Additional pyramid levels
```

**Zarr v3 (version 0.5):**

```
image.zarr/
├── zarr.json                  # Group + OME metadata under "attributes.ome"
├── 0/
│   ├── zarr.json              # Array metadata + dimension_names
│   └── c/0/1/2/3              # Chunk files
└── n/
```

### Multiscales metadata evolution

The `multiscales` attribute defines the image pyramid structure. This is where the most significant cross-version differences appear:

**Version 0.2** — Implicit axes, no transforms:

```json
{
  "multiscales": [{
    "version": "0.2",
    "name": "example",
    "datasets": [{"path": "0"}, {"path": "1"}],
    "type": "gaussian"
  }]
}
```

Axes were **implicitly assumed TCZYX** with no explicit declaration.

**Version 0.3** — Mandatory axes as strings:

```json
{
  "multiscales": [{
    "version": "0.3",
    "axes": ["t", "c", "z", "y", "x"],
    "datasets": [{"path": "0"}, {"path": "1"}]
  }]
}
```

The `axes` field became **MUST** requirement, supporting 2D–5D with explicit dimension names.

**Version 0.4** — Axes as objects with units + coordinateTransformations:

```json
{
  "multiscales": [{
    "version": "0.4",
    "axes": [
      {"name": "c", "type": "channel"},
      {"name": "z", "type": "space", "unit": "micrometer"},
      {"name": "y", "type": "space", "unit": "micrometer"},
      {"name": "x", "type": "space", "unit": "micrometer"}
    ],
    "datasets": [{
      "path": "0",
      "coordinateTransformations": [
        {"type": "scale", "scale": [1.0, 0.5, 0.5, 0.5]}
      ]
    }]
  }]
}
```

**Breaking change**: Axes became objects with `name`, `type`, `unit`. Each dataset **MUST** include `coordinateTransformations` with at least a `scale` transform.

**Version 0.5** — Namespaced under "ome" key:

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "ome": {
      "version": "0.5",
      "multiscales": [{
        "axes": [
          {"name": "c", "type": "channel"},
          {"name": "z", "type": "space", "unit": "micrometer"},
          {"name": "y", "type": "space", "unit": "micrometer"},
          {"name": "x", "type": "space", "unit": "micrometer"}
        ],
        "datasets": [{
          "path": "0",
          "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.5, 0.5, 0.5]}]
        }]
      }]
    }
  }
}
```

All OME metadata moves under the `"ome"` namespace. Array metadata **MUST** include `dimension_names` matching axes.

### Axis and coordinate transform requirements by version

| Feature | 0.2 | 0.3 | 0.4 | 0.5 |
|---------|-----|-----|-----|-----|
| Axes field | Implicit TCZYX | MUST (strings) | MUST (objects) | MUST (objects) |
| Axis type | N/A | N/A | SHOULD (space/time/channel) | SHOULD |
| Axis unit | N/A | N/A | SHOULD for space/time | SHOULD |
| coordinateTransformations | N/A | N/A | MUST per dataset | MUST per dataset |
| Scale transform | N/A | N/A | MUST (first in list) | MUST |
| Translation transform | N/A | N/A | MAY (after scale) | MAY |
| dimension_names in array | N/A | N/A | N/A | MUST match axes |

---

## Multi-well plate storage: HCS specification

### Plate hierarchy across all versions

The plate structure has remained consistent since 0.2, with refinements in 0.4:

```
plate.zarr/
├── .zattrs                    # "plate" specification
├── A/                         # Row A
│   ├── 1/                     # Well A1
│   │   ├── .zattrs            # "well" specification
│   │   ├── 0/                 # Field of view 0
│   │   │   ├── .zattrs        # "multiscales", "omero"
│   │   │   ├── 0/, 1/, n/     # Resolution levels
│   │   │   └── labels/        # Optional segmentation
│   │   └── m/                 # Additional fields
│   └── 12/                    # Well A12
└── H/                         # Row H
```

Three groups **MUST** exist above images: **plate** → **row** → **well**.

### Plate metadata differences

**Versions 0.2–0.3** — Basic well paths:

```json
{
  "plate": {
    "version": "0.3",
    "columns": [{"name": "1"}, {"name": "2"}],
    "rows": [{"name": "A"}, {"name": "B"}],
    "wells": [
      {"path": "A/1"},
      {"path": "B/2"}
    ],
    "acquisitions": [{"id": 0, "name": "Run 1"}]
  }
}
```

**Versions 0.4–0.5** — Added rowIndex/columnIndex for sparse plates:

```json
{
  "plate": {
    "version": "0.4",
    "columns": [{"name": "1"}, {"name": "2"}],
    "rows": [{"name": "A"}, {"name": "B"}],
    "wells": [
      {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
      {"path": "B/2", "rowIndex": 1, "columnIndex": 1}
    ],
    "acquisitions": [{"id": 0, "name": "Run 1", "maximumfieldcount": 4}],
    "field_count": 4
  }
}
```

**Breaking change in 0.4**: `rowIndex` and `columnIndex` became MUST requirements for well entries, enabling efficient sparse plate handling without path parsing.

### Well metadata

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

| Field | Requirement | Description |
|-------|-------------|-------------|
| images | MUST | List of field of view objects |
| images[].path | MUST | Path to image group |
| images[].acquisition | SHOULD | Links to plate acquisition ID |

---

## Collection storage: bioformats2raw layout

### When to use collections

The bioformats2raw layout handles **multi-image filesets** that don't fit the plate model—such as multiple stage positions on a single coverslip, time-lapse series stored as separate files, or z-stacks split across files. This is **transitional** metadata that will be replaced with explicit specifications in future versions.

### Collection structure

```
series.ome.zarr/
├── .zattrs                    # {"bioformats2raw.layout": 3}
├── OME/
│   ├── .zattrs                # {"series": ["0", "1", "2"]}
│   └── METADATA.ome.xml       # Complete OME-XML metadata
├── 0/                         # First image (multiscales)
├── 1/                         # Second image
└── n/                         # Additional images
```

**Version availability**: Formalized in **0.4** but pattern existed earlier. In 0.5, metadata moves to `zarr.json` but structure remains similar.

### bioformats2raw metadata requirements

```json
{
  "bioformats2raw.layout": 3
}
```

| Requirement | Description |
|-------------|-------------|
| MUST have value `3` | Layout version indicator |
| SHOULD have `OME/METADATA.ome.xml` | Full OME-XML preservation |
| MAY have `series` attribute | Explicit image path ordering |

**Image location logic**:

1. If `plate` metadata present → images at plate-defined locations
2. If `series` attribute exists → paths must match OME-XML Image element order
3. Otherwise → consecutively numbered groups: `0/`, `1/`, `2/`...

---

## Metadata storage locations by version

### Version 0.2–0.4 (Zarr v2)

| File | Contents |
|------|----------|
| `.zgroup` | `{"zarr_format": 2}` |
| `.zarray` | Array shape, chunks, dtype, compressor |
| `.zattrs` | All OME-NGFF metadata (multiscales, omero, plate, well, labels) |
| `OME/METADATA.ome.xml` | Full OME-XML (bioformats2raw only) |

### Version 0.5 (Zarr v3)

| File | Contents |
|------|----------|
| `zarr.json` (group) | `zarr_format`, `node_type`, `attributes.ome.*` |
| `zarr.json` (array) | Shape, chunks, codecs, `dimension_names` |
| `OME/METADATA.ome.xml` | Full OME-XML (bioformats2raw only) |

**Key 0.5 change**: All OME metadata moves under `attributes.ome` namespace, enabling cleaner separation from other Zarr attributes.

### Storing experimental metadata (timestamps, instrument settings)

OME-NGFF provides **two mechanisms** for experimental metadata:

1. **OME-XML file** (`OME/METADATA.ome.xml`): Complete instrument, acquisition, and experimental metadata using the established OME data model. Available in bioformats2raw layouts.

2. **"omero" transitional metadata**: Rendering hints including channel colors, contrast limits, and display names:

```json
{
  "omero": {
    "channels": [{
      "color": "00FF00",
      "label": "GFP",
      "window": {"start": 0, "end": 4095, "min": 0, "max": 65535},
      "active": true
    }],
    "rdefs": {"defaultZ": 50, "defaultT": 0, "model": "color"}
  }
}
```

---

## Practical storage scenarios

### Scenario 1: Basic 3D confocal image

**Use case**: Single-position CZYX confocal stack with 3 channels.

**Recommended structure** (version 0.4):

```
confocal_image.zarr/
├── .zattrs                    # multiscales with axes [c,z,y,x]
├── 0/                         # Full resolution (e.g., 3×100×2048×2048)
│   └── c/z/y/x chunks
├── 1/                         # 2x downsampled in XY
├── 2/                         # 4x downsampled
└── labels/                    # Optional segmentation
    └── nuclei/
```

**Metadata**:

```json
{
  "multiscales": [{
    "version": "0.4",
    "axes": [
      {"name": "c", "type": "channel"},
      {"name": "z", "type": "space", "unit": "micrometer"},
      {"name": "y", "type": "space", "unit": "micrometer"},
      {"name": "x", "type": "space", "unit": "micrometer"}
    ],
    "datasets": [
      {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.3, 0.1, 0.1]}]},
      {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.3, 0.2, 0.2]}]},
      {"path": "2", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.3, 0.4, 0.4]}]}
    ]
  }]
}
```

### Scenario 2: Multiple stage positions (FOVs) on single coverslip

**Use case**: 12 fields of view across a coverslip, not a multi-well plate.

**Recommended approach**: Use **bioformats2raw layout** (collection):

```
coverslip_experiment.zarr/
├── .zattrs                    # {"bioformats2raw.layout": 3}
├── OME/
│   ├── .zattrs                # {"series": ["0","1","2"..."11"]}
│   └── METADATA.ome.xml       # Contains stage positions, timestamps
├── 0/                         # FOV 0 (position X=100, Y=200)
│   ├── .zattrs                # multiscales
│   └── 0/, 1/, 2/             # Resolution levels
├── 1/                         # FOV 1 (position X=500, Y=200)
└── 11/                        # FOV 11
```

**Why this approach**:

- Preserves stage position coordinates in OME-XML
- Maintains logical grouping without artificial plate structure
- Each FOV gets independent multiscales with `coordinateTransformations` that can include `translation` for absolute positioning

**Per-FOV coordinate transforms** (0.4+):

```json
{
  "multiscales": [{
    "coordinateTransformations": [
      {"type": "translation", "translation": [0.0, 0.0, 100.0, 200.0]}
    ],
    "datasets": [{
      "path": "0",
      "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.5, 0.1, 0.1]}]
    }]
  }]
}
```

### Scenario 3: Multi-well plate experiment

**Use case**: 96-well plate, 4 fields per well, 2 acquisition rounds.

**Structure**:

```
plate_experiment.zarr/
├── .zattrs                    # plate specification
├── A/
│   ├── 1/
│   │   ├── .zattrs            # well with 8 images (4 fields × 2 acquisitions)
│   │   ├── 0/...3/            # Fields from acquisition 0
│   │   └── 4/...7/            # Fields from acquisition 1
│   └── 12/
└── H/
```

**Plate metadata**:

```json
{
  "plate": {
    "version": "0.4",
    "acquisitions": [
      {"id": 0, "name": "Initial", "maximumfieldcount": 4},
      {"id": 1, "name": "24h treatment", "maximumfieldcount": 4}
    ],
    "columns": [{"name": "1"}, {"name": "2"}, ... {"name": "12"}],
    "rows": [{"name": "A"}, ... {"name": "H"}],
    "wells": [
      {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
      {"path": "A/2", "rowIndex": 0, "columnIndex": 1}
    ],
    "field_count": 4
  }
}
```

---

## Version evolution: Breaking changes and migrations

### 0.2 → 0.3: Explicit axes

| Change | Impact |
|--------|--------|
| `axes` field mandatory | **Breaking**: 0.2 readers won't understand 0.3 files |
| 2D–5D support | Expansion: No longer assumes 5D |
| String axis names | `["t", "c", "z", "y", "x"]` format |

### 0.3 → 0.4: Rich axis metadata + transforms

| Change | Impact |
|--------|--------|
| Axes become objects | **Breaking**: `{"name": "x", "type": "space", "unit": "micrometer"}` |
| coordinateTransformations required | **Breaking**: Must include scale per dataset |
| Plate rowIndex/columnIndex | **Breaking**: Required for wells |
| bioformats2raw.layout formalized | Addition: Transitional spec |

### 0.4 → 0.5: Zarr v3 transition

| Change | Impact |
|--------|--------|
| Zarr v3 only | **Breaking**: Completely different file structure |
| Metadata namespacing | All under `attributes.ome` |
| dimension_names in arrays | **MUST** match multiscales axes |
| Sharding support | New capability for large datasets |
| Codec chains | More flexible compression |

### Migration considerations

- **0.4 is most compatible**: Widest tooling support currently
- **0.5 requires Zarr v3-capable tools**: zarr-python 3.x, zarrita
- **Metadata-only migration**: Zarr v2→v3 can preserve array data; only JSON files change
- **Reading backward**: Implementations SHOULD read older versions; writing is optional

---

## Documentation design recommendations

Based on this analysis, your documentation should:

**1. Lead with data type categories**, not versions:

- "Storing Images" → then tabs for 0.2/0.3/0.4/0.5
- "Multi-well Plates" → version tabs
- "Image Collections" → version tabs (note: 0.4+ only)

**2. Provide clear decision guidance**:

- New projects: Start with 0.4 (stable, well-supported)
- Cloud-scale with sharding needs: Consider 0.5
- Reading existing data: Support all versions

**3. Show complete examples for each scenario**:

- Basic confocal image (most common)
- Multi-FOV experiments (clarify collection vs. plate choice)
- HCS plates (show acquisition linking)

**4. Highlight breaking changes prominently**:

- 0.2→0.3: Axes string format
- 0.3→0.4: Axes objects + coordinateTransformations
- 0.4→0.5: Zarr v3 file structure

**5. Provide version comparison tables** for quick expert reference:

- Axes format by version
- Required vs optional fields
- Metadata file locations

This structure lets newcomers understand the core patterns (multiscales, plates, collections) while experts can jump directly to version-specific syntax differences via tabs.
