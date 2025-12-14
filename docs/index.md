---
icon: material/rocket-launch
title: Get Started
---

<span class="yaozarrs-animated">yaozarrs!!</span>

# Get Started with `yaozarrs`

Yaozarrs is a **bottom-up** library for working with OME-Zarr metadata and
stores in Python. It is **lightweight**, with only a **single required
dependency** (on Pydantic), but feature-rich, with structural validation and
writing functions added via optional extras.

## Philosophy

!!! important ""
    **The core philosophy is that NGFF metadata and Zarr array I/O are separate
    concerns.**

    **Yaozarrs focuses on OME-Zarr metadata creation, manipulation, and
    validation, and all other functionality (writing and structural validation)
    is optional, with flexible backend.**

Zarr itself is a _specification_ with multiple implementations:  There are many
ways to read and write Zarr stores (e.g. `zarr-python`, `tensorstore`,
`acquire-zarr`, `zarrs`, etc..) and **yaozarrs makes no assumptions about which
implementation you may want to use**.  Similarly, OME NGFF is a _metadata
sepcification_, defining what JSON documents and hierarchy structure must look
like.

1. At its core, **yaozarrs provides pydantic models for OME-Zarr metadata
  specifications**. You should be able to create/manipulate/validate OME-Zarr
  metadata without any specific zarr array library, or anything beyond
  `yaozarrs`, `pydantic`, and the standard library.

    ??? question "Why pydantic?"
        It's true that one can define dataclasses that mirror the OME-Zarr
        schema; but reinventing **validation** and **ser/deserialization** is
        beyond the scope of this project, and pydantic is a battle-tested
        library for exactly these tasks. It's lightweight in terms of transitive
        dependencies, ~7MB in size, and is broadly compatible. We test against a
        broad range of pydantic versions (v2+) on a broad range of python
        versions and OS, to ensure that yaozarrs is an easy/robust dependency to
        add.

2. Because reading/writing zarr groups is far simpler than arrays, **you
  shouldn't need to depend on a specific complete zarr library just to validate
  that a given _hierarchy_ is structurally correct**.  
  <small>For example: a library implementing a new low-level zarr array backend
  should be able to use yaozarrs to validate that its group structure and
  metadata are correct, without needing to depend on zarr-python or
  tensorstore.</small>

    ??? note "`pip install 'yaozarrs[io]'`"
        If you want to perform structural validation of possibly _remote_ zarr
        stores, then you will need to install the `io` extra, which adds
        dependencies on `fsspec`.

3. Even in the case of writing complete OME-Zarr stores, the "array" part is
   relatively stereotyped, and the metadata is the more user-customized part.  
   With yaozarrs, you can create the metadata using the pydantic models, and
   then use convenience functions to write the zarr stores using _any_ zarr
   array creation method you want, with built-in (optional) implementations for
   `zarr-python` and `tensorstore`.

    ??? note "`pip install 'yaozarrs[write-zarr]'` or `[write-tensorstore]`"
        The builtin backends in the `yaozarrs.write` module require an
        array-writing backend, currently either `zarr` (zarr-python) or
        `tensorstore`.  Install the appropriate extra to enable these features.

## API Quick Reference

### Metadata Creation

(e.g. for Zarr group creation)

```python
import yaozarrs
from yaozarrs import v04, v05, DimSpec

image = v05.Image(
    multiscales=[
        v05.Multiscale.from_dims(
            dims=[
                DimSpec(name="t", unit="second"),  # (1)!
                DimSpec(name="c"),
                DimSpec(name="z", scale=0.3, unit="micrometer"),
                DimSpec(name="y", scale=0.1, unit="micrometer"),
                DimSpec(name="x", scale=0.1, unit="micrometer"),
            ]
        )
    ]
)

# Export
obj.model_dump_json(exclude_unset=True, indent=2)  # (2)!
```

1. :eyes: [`yaozarrs.DimSpec`][] is a convenience class for use with
    [`Multiscale.from_dims`][yaozarrs.v05._image.Multiscale.from_dims].  It's
    not part of the OME-Zarr spec.
2. [`model_dump_json`][pydantic.BaseModel.model_dump_json] is part of the
   standard pydantic API for exporting models to JSON.  Just an example of
   exporting metadata back to JSON format.

### Validation of existing objects

If you have an existing JSON file, string, or python object, you can
validate it and cast it to the appropriate typed `yaozarrs` model:

```python
import yaozarrs

# validate a JSON string/bytes literal
yaozarrs.validate_ome_json(json_str)  # (1)!

# validate any python object (e.g. dict)
yaozarrs.validate_ome_object(dict_obj) # (2)!

# validate entire Zarr hierarchy (both metadata and structure) at any URI
yaozarrs.validate_zarr_store(uri) # (3)!
```

1. [`yaozarrs.validate_ome_json`][]
2. [`yaozarrs.validate_ome_object`][]
3. [`yaozarrs.validate_zarr_store`][]. Requires the `yaozarrs[io]`
   extra to support remote URIs.

### Loading and Navigating with validation

[`yaozarrs.open_group`][] teturns a small wrapper around a zarr group with
minimal functionality: [`yaozarrs.ZarrGroup`][].  Requires the [`yaozarrs[io]`
extra](./installation.md#structural-validation) to support remote URIs.

```python
import yaozarrs

# Open a Zarr group at any URI
yaozarrs.open_group(uri) # (1)!
```

### Writing OME-Zarr Stores

See [`yaozarrs.write.v05`][] for all the writing convenience functions.

```python
import numpy as np

from yaozarrs import write # (7)!

# write a 5D image with single pyramid level

data = np.zeros((10, 3, 9, 64, 64), dtype=np.uint16)
root_path = write.v05.write_image("example.ome.zarr", image, data)

# other high level write functions

write.v05.write_plate(...)
write.v05.write_bioformats2raw(...)

# low-level prepare/builder functions

write.v05.prepare_image(...)
write.v05.LabelsBuilder(...)
write.v05.PlateBuilder(...)
write.v05.Bf2RawBuilder(...)

```

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
