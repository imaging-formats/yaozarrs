---
icon: material/rocket-launch
title: Get Started
---

<span class="yaozarrs-animated">yaozarrs!!</span>

# Get Started with `yaozarrs`

Yaozarrs is a **bottom-up** library for working with OME-Zarr metadata and
stores in Python. It is **lightweight**, with only a **single required
dependency** (on Pydantic), but feature-rich, with structural validation and
writing functions added via [optional extras](./installation.md).

## Quick Links

- :material-download: See [installation guide](./installation.md) for instructions on how to install
  `yaozarrs`.
- :material-api: For a quick overview of the API, see the [API Quick Reference](#api-quick-reference)
  below, or the full [API Reference](./API_Reference/yaozarrs.md).
- :material-book-open-variant: If you're new to OME-Zarr or don't know how to express your data in OME-Zarr
  format, see the [yaozarrs guide to OME-Zarr](./ome_zarr_guide.md).
- :material-application-brackets: If you learn best with examples, check out the interactive
  [OME-Zarr explorer](./ome_zarr_explorer.md) app.
- :material-brain: To understand *why* yaozarrs exists, see the
  [Principles](#principles) section below.

## API Quick Reference

### Metadata Creation

You can create OME-Zarr metadata objects using the pydantic models in
`yaozarrs.v04` and `yaozarrs.v05` modules.  Pick a top-level object (e.g.
[`Image`](./ome_zarr_guide.md#working-with-images), [`Plate`](./ome_zarr_guide.md#working-with-plates), etc..) and create it using standard pydantic model syntax:

=== "From `yaozarrs` models"

    For full IDE autocompletion and type checking, you can create OME-Zarr metadata
    using the yaozarrs models:

    ```python
    from yaozarrs import v04, v05  # import from appropriate version module

    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name=None,
                axes=[
                    v05.TimeAxis(name='t', unit='second'),
                    v05.ChannelAxis(name='c'),
                    v05.SpaceAxis(name='z', unit='micrometer'),
                    v05.SpaceAxis(name='y', unit='micrometer'),
                    v05.SpaceAxis(name='x', unit='micrometer')
                ],
                datasets=[
                    v05.Dataset(
                        path='0',
                        coordinateTransformations=[
                            v05.ScaleTransformation(
                                scale=[1.0, 1.0, 0.3, 0.1, 0.1]
                            )
                        ]
                    )
                ],
            )
        ],
    )

    # Export
    image.model_dump_json(exclude_unset=True, indent=2)  # (1)!
    ```

    1. [`model_dump_json`][pydantic.BaseModel.model_dump_json] is part of the
    standard pydantic API for exporting models to JSON.  Just an example of
    exporting metadata back to JSON format.

=== "From dictionaries"

    If you prefer avoiding explicit model classes, you can create OME-Zarr
    metadata using standard python dictionaries (standard pydantic model parsing):

    ```python
    import yaozarrs
    from yaozarrs import v05

    image = v05.Image.model_validate(  # (1)!
        {
            'multiscales': [
                {
                    'axes': [
                        {'name': 't', 'type': 'time', 'unit': 'second'},
                        {'name': 'c', 'type': 'channel'},
                        {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
                        {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                        {'name': 'x', 'type': 'space', 'unit': 'micrometer'}
                    ],
                    'datasets': [
                        {
                            'path': '0',
                            'coordinateTransformations': [
                                {'type': 'scale', 'scale': [1.0, 1.0, 0.3, 0.1, 0.1]}
                            ]
                        }
                    ]
                }
            ]
        }
    )

    # Export
    image.model_dump_json(exclude_unset=True, indent=2) # (2)!
    ```

    1. [`model_validate`][pydantic.BaseModel.model_validate] is part of the
       standard pydantic API for creating models from any object, like dictionaries.
    2. [`model_dump_json`][pydantic.BaseModel.model_dump_json] is part of the
    standard pydantic API for exporting models to JSON.  Just an example of
    exporting metadata back to JSON format.

=== "`DimSpec` Convenience"

    For the common case of creating multiscale images, you can use the
    [`yaozarrs.DimSpec`][] convenience class to simplify axis and transformation
    creation:

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
    image.model_dump_json(exclude_unset=True, indent=2)  # (2)!
    ```
    
    1. :eyes: [`yaozarrs.DimSpec`][] is a convenience class for use with
        [`Multiscale.from_dims`][yaozarrs.v05._image.Multiscale.from_dims].  It's
        not part of the OME-Zarr spec.
    2. [`model_dump_json`][pydantic.BaseModel.model_dump_json] is part of the
    standard pydantic API for exporting models to JSON.  Just an example of
    exporting metadata back to JSON format.

------------

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

???tip "Storage Validation Errors"

    Validation errors that relate to the structure of the OME-Zarr itself
    (as opposed to metadata) are collected and presented similarly to pydantic
    validation errors for the metadata:

    ```txt
    location
    description [context]
    ```

    An example validation error (for a file that has *many* problems):

    ```bash
    uvx "yaozarrs[io]" validate https://raw.githubusercontent.com/tlambert03/yaozarrs/refs/heads/main/tests/data/broken/broken_v05.ome.zarr/
    ```

    ```txt
    yaozarrs._storage.StorageValidationError: 14 validation error(s) for StorageValidationError
    ome.plate.wells.0.well.images.0.multiscales.0.datasets.0.path
    Dataset path '0' not found in zarr group [type=dataset_path_not_found, fs_path='broken_v05.ome.zarr/A/1/0/0', expected='zarr array']

    ome.plate.wells.0.well.images.0.labels.labels.0
    Label path 'annotations' not found in labels group [type=label_path_not_found, fs_path='broken_v05.ome.zarr/A/1/0/labels/annotations', expected='zarr group']

    ome.plate.wells.0.well.images.1.labels.labels.0
    Label path 'annotations' is not a zarr group [type=label_path_not_group, fs_path='broken_v05.ome.zarr/A/1/1/labels/annotations', expected='group', found='array']

    ome.plate.wells.1.path
    Well path 'A/2' is not a zarr group [type=well_path_not_group, fs_path='broken_v05.ome.zarr/A/2', expected='group', found='array']

    ome.plate.wells.2.well.images.0.labels.labels.0
    Label path 'annotations' does not contain valid Image ('multiscales') metadata [type=label_image_invalid, path='annotations']
    1 validation error for tagged-union[LabelImage,Image,Plate,Bf2Raw,Well,LabelsGroup,Series]
        Unable to extract tag using discriminator _discriminate_ome_v05_metadata() [type=union_tag_not_found, input_value={}, input_type=dict]
        For further information visit https://errors.pydantic.dev/2.12/v/union_tag_not_found

    ome.plate.wells.3.well.images.0.multiscales.0.datasets.0.path
    Dataset '0' has 5 dimensions but axes specify 3 [type=dataset_dimension_mismatch, fs_path='broken_v05.ome.zarr/B/1/0/0', actual_ndim=5, expected_ndim=3, axes=['c', 'y', 'x']]

    ome.plate.wells.3.well.images.0.labels.labels.0.multiscales.0.datasets.0.path
    Label array '0' has non-integer dtype 'float32'. Labels must use integer types. [type=label_non_integer_dtype, path='0', dtype='float32']

    ome.plate.wells.4.well.images.0.multiscales.0.datasets.0.path
    Dataset path '0' exists but is not a zarr array [type=dataset_not_array, fs_path='broken_v05.ome.zarr/B/2/0/0', expected='array', found='group']

    ome.plate.wells.4.well.images.1.multiscales.0.datasets.0.path.dimension_names
    Array dimension_names ['wrong', 'names', 'here'] don't match axes names ['c', 'y', 'x'] [type=dimension_names_mismatch, expected=['c', 'y', 'x'], actual=['wrong', 'names', 'here']]

    ome.plate.wells.5.well.images.0.labels
    Found 'labels' path but it is a <class 'yaozarrs._zarr.ZarrArray'>, not a zarr group [type=labels_not_group, expected='group', found='ZarrArray']

    ome.plate.wells.5.well.images.1.path
    Field path '1' is not a zarr group [type=field_path_not_group, fs_path='broken_v05.ome.zarr/B/3/1', expected='group', found='array']

    ome.plate.wells.6.well.images.1.path
    Field path '1' not found in well group [type=field_path_not_found, fs_path='broken_v05.ome.zarr/C/1/1', expected='zarr group']

    ome.plate.wells.7.well.images.0
    Field path '0' does not contain valid Image metadata [type=field_image_invalid, fs_path='broken_v05.ome.zarr/C/2/0']
    1 validation error for tagged-union[LabelImage,Image,Plate,Bf2Raw,Well,LabelsGroup,Series]
    image.multiscales
        Value should have at least 1 item after validation, not 0 [type=too_short, input_value=[], input_type=list]
        For further information visit https://errors.pydantic.dev/2.12/v/too_short

    ome.plate.wells.8
    Well path 'C/3' does not contain valid Well metadata [type=well_invalid, path='C/3']
    1 validation error for tagged-union[LabelImage,Image,Plate,Bf2Raw,Well,LabelsGroup,Series]
        Unable to extract tag using discriminator _discriminate_ome_v05_metadata() [type=union_tag_not_found, input_value={}, input_type=dict]
        For further information visit https://errors.pydantic.dev/2.12/v/union_tag_not_found
    ```

### Writing OME-Zarr Stores

See [`yaozarrs.write.v05`][] for convenience functions to write OME-Zarr v0.5
stores using your metadata objects, with either `zarr-python`, `tensorstore`,
or custom backends.

```python
import numpy as np

from yaozarrs import write

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

### CLI validation

The CLI command provides a quick way to validate any zarr store as an OME-Zarr
store.  Here, "store" here refers to any URI (local path, http(s) url, or s3 url.

!!!important
    Requires `fsspec`. install with `pip install yaozarrs[io]`

=== "with uvx"

    ```sh
    uvx "yaozarrs[io]" validate https://raw.githubusercontent.com/tlambert03/yaozarrs/refs/heads/main/tests/data/broken/broken_v05.ome.zarr/
    ```

=== "from installed package"

    ```sh
    yaozarrs validate https://raw.githubusercontent.com/tlambert03/yaozarrs/refs/heads/main/tests/data/broken/broken_v05.ome.zarr/
    ```

```sh
✓ Valid OME-Zarr store
  Version: 0.5
  Type: Image
```

### Loading OME-Zarr Stores

[`yaozarrs.open_group`][] teturns a small wrapper around a zarr group with
minimal functionality: [`yaozarrs.ZarrGroup`][].  Requires the [`yaozarrs[io]`
extra](./installation.md#structural-validation) to support remote URIs.  This
class is used behind the scenes for structural validation, but can also be used
directly.

```python
import yaozarrs

# Open a Zarr group at any URI
group = yaozarrs.open_group(uri)
print(group.ome_metadata()) # (1)!
child = group["0"]  # (2)!
```

1. [`ZarrGroup.ome_metadata`][yaozarrs._zarr.ZarrGroup.ome_metadata] attempts to
   extract and validate OME-Zarr metadata from the opened group, returning the
   appropriate typed yaozarrs model (e.g. `yaozarrs.v05.Image`, etc..).
2. Access arrays and sub-groups using standard dictionary-like syntax. Returns
   [`ZarrArray`][yaozarrs._zarr.ZarrArray] or [`ZarrGroup`][yaozarrs._zarr.ZarrGroup]

#### Access array data

This package does *not* depend on `zarr` or `tensorstore`, even for validating
OME-Zarr stores. (It uses a minimal representation of a zarr group internally,
backed by `fsspec`.)  If you *would* like to actually open arrays, you can use
either [`ZarrArray.to_tensorstore`][yaozarrs._zarr.ZarrArray.to_tensorstore] or
[`ZarrArray.to_zarr_python`][yaozarrs._zarr.ZarrArray.to_zarr_python] to cast an
array node to a full zarr array using your preferred backend.

!!!important
    These methods require that you have the appropriate backend installed
    (`tensorstore` or `zarr`).

```python
from yaozarrs import open_group

group = open_group("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr")
array = group['0']
# <ZarrArray https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr/0>

# read bytes using tensorstore or zarr-python:
ts_array = array.to_tensorstore() # isinstance(ts_array, tensorstore.TensorStore)
zarr_array = array.to_zarr_python() # isinstance(zarr_array, zarr.Array)
```

## Principles

!!! important ""
    **The core philosophy is that NGFF metadata and Zarr array I/O are separate
    concerns.**

    **Yaozarrs focuses on OME-Zarr metadata creation, manipulation, and
    validation, and all other functionality (writing and structural validation)
    is optional, with flexible backend.**

Zarr itself is a *specification* with multiple implementations:  There are many
ways to read and write Zarr stores (e.g. `zarr-python`, `tensorstore`,
`acquire-zarr`, `zarrs`, etc..) and **yaozarrs makes no assumptions about which
implementation you may want to use**.  Similarly, OME NGFF is a *metadata
sepcification*, defining what JSON documents and hierarchy structure must look
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
  that a given *hierarchy* is structurally correct**.  
  <small>For example: a library implementing a new low-level zarr array backend
  should be able to use yaozarrs to validate that its group structure and
  metadata are correct, without needing to depend on zarr-python or
  tensorstore.</small>

    ??? note "`pip install 'yaozarrs[io]'`"
        If you want to perform structural validation of possibly *remote* zarr
        stores, then you will need to install the `io` extra, which adds
        dependencies on `fsspec`.

3. Even in the case of writing complete OME-Zarr stores, the "array" part is
   relatively stereotyped, and the metadata is the more user-customized part.  
   With yaozarrs, you can create the metadata using the pydantic models, and
   then use convenience functions to write the zarr stores using *any* zarr
   array creation method you want, with built-in (optional) implementations for
   `zarr-python` and `tensorstore`.

    ??? note "`pip install 'yaozarrs[write-zarr]'` or `[write-tensorstore]`"
        The builtin backends in the `yaozarrs.write` module require an
        array-writing backend, currently either `zarr` (zarr-python) or
        `tensorstore`.  Install the appropriate extra to enable these features.

## See Also

[ome-zarr-models-py](https://github.com/ome-zarr-models/ome-zarr-models-py) is
another set of Pydantic models for OME-Zarr. It has garnered broad community
support, and aligns well with many use cases.  It should also be
considered for OME-Zarr metadata validation and manipulation in Python!

The primary difference is that it depends on `zarr-python`, and is
more directly tied to the zarr-python implementation.
