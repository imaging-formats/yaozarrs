
# Structural Validation Requirements

1. Image Group Validation (multiscales key present)

- Dataset Path Existence: Verify each dataset.path points to a valid zarr array
- Array Dimensionality: Confirm each array has dimensions matching len(axes)
- Array Dimension Names: Check zarr array dimension_names attribute matches axes names and order
- Resolution Ordering: Validate datasets are ordered from highest to lowest resolution
- Data Type Constraints: No specific constraints (can be any numeric type)
- Labels Discovery: Check for optional labels/ group at same level as datasets

2. Labels Group Validation (labels key present)

- Label Path Existence: Verify each path in labels array points to a valid LabelImage group
- Label Image Validation: Each referenced label image must be a valid LabelImage with image-label
metadata

3. Label Image Validation (multiscales + image-label keys present)

- Inherits Image Validation: All image validation rules apply
- Integer Data Types: All arrays must contain integer data types (uint8, uint16, uint32, etc.)
- Source Reference Validation: If source.image is present, verify it points to a valid location

4. Plate Group Validation (plate key present)

- Well Path Existence: Verify each well.path points to a valid Well group
- Well Path Format: Paths must follow row/column format
- Row/Column Consistency: Well paths must match existing rows and columns names
- Index Validation: Well rowIndex and columnIndex must be valid for the plate dimensions

5. Well Group Validation (well key present)

- Field Path Existence: Verify each images[].path points to a valid Image group
- Acquisition References: If acquisition IDs are present, they should match plate-level acquisitions

6. Series Collection Validation (series key present)

- Series Path Existence: Verify each path in series array points to a valid Image group

7. Bioformats2raw Layout Validation (bioformats2raw.layout key present)

- Numbered Directory Discovery: Find all numbered directories (0/, 1/, 2/, etc.)
- Image Group Validation: Each numbered directory should contain valid Image metadata

8. Cross-Reference Validation

- Coordinate Transformations Consistency: Transformations should be reasonable across resolution levels
- Axis Consistency: Spatial axes should have consistent units and ordering throughout hierarchy
- Label-Image Relationship: Labels should have compatible dimensions with their source images
