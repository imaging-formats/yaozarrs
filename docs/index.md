---
icon: material/rocket-launch
title: Get Started
---

# Get Started with yaozarrs

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
