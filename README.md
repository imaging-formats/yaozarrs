# yaozarrs ‼️

[![License](https://img.shields.io/pypi/l/yaozarrs.svg?color=green)](https://github.com/tlambert03/yaozarrs/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/yaozarrs.svg?color=green)](https://pypi.org/project/yaozarrs)
[![Python Version](https://img.shields.io/pypi/pyversions/yaozarrs.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/yaozarrs/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/yaozarrs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/yaozarrs/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/yaozarrs)

***Yet Another Ome-ZARR Schema!***

`yaozarrs` is a Python library with minimal dependencies (only pydantic) that:

- provides pydantic models for the [ome-zarr NGFF specification](https://ngff.openmicroscopy.org/specifications/index.html)
- can create ome-zarr model objects (with IDE completion and type safety) and dump to JSON.
- can validate ome-zarr metadata, collected either from `zarr.json` documents, or `zarr` stores.
- can **validate ome-zarr stores** (both metadata and structure) from a URI or zarr store; local or remote.  
  (This functionality additionally requires `fsspec`, but does *not* depend on or require zarr-python or other zarr implementations)


## Oh no, not another one 🤦

First, let me apologize. The last thing the world needs is yet another ome-zarr
model. However, I was unable to find a *minimal* ome-zarr model that simply
represents the spec, without introducing additional I/O features or
dependencies.  Please read the [Existing Projects](#existing-projects) section
for more context.

## Installation

```bash
pip install yaozarrs

# or, to load/validate local/remote zarr stores:
pip install yaozarrs[io]
```

## Usage

Here are some things you can do with `yaozarrs`.

1. [Construct valid ome-zarr JSON documents for creating ome-zarr groups](#construct-valid-ome-zarr-json-documents-for-creating-ome-zarr-groups)
2. [Validate & load existing JSON documents](#validate--load-existing-json-documents)
3. [Validate arbitrary python objects as an OME-NGFF object](#validate-arbitrary-python-objects-as-an-ome-ngff-object)
4. [Validate any zarr store using the CLI](#validate-any-zarr-store-using-the-cli)
5. [Validate any zarr store programmatically](#validate-any-zarr-store-programmatically)
6. [Open zarr arrays using zarr-python or tensorstore](#open-zarr-arrays-using-zarr-python-or-tensorstore)

### Construct valid ome-zarr JSON documents for creating ome-zarr groups

This is useful if you are creating OME-Zarr files directly.  Since this
package has no dependencies beyond `pydantic`, it allows downstream projects to
use a common model, without enforcing a specific mechanism for data I/O (e.g.
using `zarr`, `tensorstore`, `acquire-zarr`, etc),

```python
from yaozarrs import v05
from pathlib import Path

scale = v05.Multiscale(
    name="scale0",
    axes=[v05.SpaceAxis(name="x", type="space"), v05.SpaceAxis(name="y", type="space")],
    datasets=[
        v05.Dataset(
            path="0",
            coordinateTransformations=[v05.ScaleTransformation(scale=[1, 1])],
        ),
        v05.Dataset(
            path="1",
            coordinateTransformations=[v05.ScaleTransformation(scale=[1, 1])],
        ),
    ],
)

img = v05.Image(multiscales=[scale])
zarr_json = v05.OMEZarrGroupJSON(attributes={"ome": img})
json_data = zarr_json.model_dump_json(exclude_unset=True)
Path("zarr.json").write_text(json_data)
```

### Validate & load existing JSON documents

If you have an existing JSON document, you can validate and load it, and
benefit from IDE autocompletion and type hints.

```python
from pathlib import Path
import yaozarrs

json_string = Path("zarr.json").read_text()
obj = yaozarrs.validate_ome_json(json_string)

# OMEZarrGroupJSON(
#     zarr_format=3,
#     node_type='group',
#     attributes=OMEAttributes(
#         ome=Image(
#             version='0.5',
#             multiscales=[
#                 Multiscale(
#                     name='scale0',
#                     axes=[SpaceAxis(name='x', type='space', unit=None), SpaceAxis(name='y', type='space', unit=None)],
#                     datasets=[
#                         Dataset(path='0', coordinateTransformations=[ScaleTransformation(type='scale', scale=[0.0, 1.0])]),
#                         Dataset(path='1', coordinateTransformations=[ScaleTransformation(type='scale', scale=[0.0, 1.0])])
#                     ],
#                     coordinateTransformations=None,
#                     type=None,
#                     metadata=None
#                 )
#             ],
#             omero=None
#         )
#     )
# )
```

### Validate arbitrary python objects as an OME-NGFF object

`validate_ome_object` and `validate_ome_json` accept a broad range of inputs,
and will cast to an appropriate model if possible.

```python
import yaozarrs

obj = yaozarrs.validate_ome_object(
  {'version': '0.5', 'series': ["0", "1"]}
)
print(obj)
# Series(version='0.5', series=['0', '1'])
```

### Validate any zarr store using the CLI

> [!IMPORTANT]  
> Requires `fsspec`. install with `pip install yaozarrs[io]`

The CLI command provides a quick way to validate any zarr store as an OME-Zarr
store.  Here, "store" here refers to any URI (local path, http(s) url, s3 url,
etc) or a zarr-python `zarr.Group`.

```bash
$ yaozarrs validate https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr
✓ Valid OME-Zarr store
  Version: 0.5
  Type: Image
```

> [!TIP]  
> Use `uvx` for quick validation of any URI, without pip installing the package.
>
> ```bash
> uvx "yaozarrs[io]" validate https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr
> ```

#### Storage Validation Errors

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

### Validate any zarr store programmatically

> [!IMPORTANT]  
> Requires `fsspec`. install with `pip install yaozarrs[io]`

```python
import yaozarrs

yaozarrs.validate_zarr_store("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr")
```

### Open zarr arrays using zarr-python or tensorstore

> [!IMPORTANT]  
>
> - `to_tensorstore()` requires `tensorstore`
> - `to_zarr_python()` requires `zarr`

This package does not depend on `zarr` or `tensorstore`, even for validating
OME-Zarr stores. (It uses a minimal representation of a zarr group internally,
backed by `fsspec`.)  If you would like to actually open arrays, you can use
either `zarr` or `tensorstore` directly.

```python
from yaozarrs import open_group

group = open_group("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr")
array = group['0']
# <ZarrArray https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr/0>

# read bytes using tensorstore or zarr-python:
ts_array = array.to_tensorstore() # isinstance(ts_array, tensorstore.TensorStore)
zarr_array = array.to_zarr_python() # isinstance(zarr_array, zarr.Array)

# inspect the OME metadata associated with the group:
print(group.ome_metadata())
# Image(
#     version='0.5',
#     multiscales=[
#         Multiscale(
#             name=None,
#             axes=[
#                 ChannelAxis(name='c', type='channel', unit=None),
#                 SpaceAxis(
#                     name='z',
#                     type='space',
#                     unit='micrometer'
#                 ),
#                 SpaceAxis(
#                     name='y',
#                     type='space',
#                     unit='micrometer'
#                 ),
#                 SpaceAxis(
#                     name='x',
#                     type='space',
#                     unit='micrometer'
#                 )
#             ],
#             datasets=[
#                 Dataset(
#                     path='0',
#                     coordinateTransformations=[
#                         ScaleTransformation(
#                             type='scale',
#                             scale=[
#                                 1.0,
#                                 0.5002025531914894,
#                                 0.3603981534640209,
#                                 0.3603981534640209
#                             ]
#                         )
#                     ]
#                 ),
#                 Dataset(
#                     path='1',
#                     coordinateTransformations=[
#                         ScaleTransformation(
#                             type='scale',
#                             scale=[
#                                 1.0,
#                                 0.5002025531914894,
#                                 0.7207963069280418,
#                                 0.7207963069280418
#                             ]
#                         )
#                     ]
#                 ),
#                 Dataset(
#                     path='2',
#                     coordinateTransformations=[
#                         ScaleTransformation(
#                             type='scale',
#                             scale=[
#                                 1.0,
#                                 0.5002025531914894,
#                                 1.4415926138560835,
#                                 1.4415926138560835
#                             ]
#                         )
#                     ]
#                 )
#             ],
#             coordinateTransformations=None,
#             type=None,
#             metadata=None
#         )
#     ],
#     omero=Omero(
#         channels=[
#             OmeroChannel(
#                 window=OmeroWindow(
#                     start=0.0,
#                     min=0.0,
#                     end=1500.0,
#                     max=65535.0
#                 ),
#                 label='LaminB1',
#                 family='linear',
#                 color='0000FF',
#                 active=True,
#                 inverted=False,
#                 coefficient=1.0
#             ),
#             OmeroChannel(
#                 window=OmeroWindow(
#                     start=0.0,
#                     min=0.0,
#                     end=1500.0,
#                     max=65535.0
#                 ),
#                 label='Dapi',
#                 family='linear',
#                 color='FFFF00',
#                 active=True,
#                 inverted=False,
#                 coefficient=1.0
#             )
#         ],
#         id=1
#     )
# )
```

## Existing Projects

You should first check these existing packages to see if they meet your needs:

- [ome-zarr-models-py](https://github.com/ome-zarr-models/ome-zarr-models-py).  
  This project has garnered strong community attention and aligns well with many use cases.  
  For my particular goals, I found a few things diverged from what I need.

  1. It offers convenient I/O helpers (based on and requiring `zarr-python`)
     that are great in many contexts, but I wanted to explore a version with no
     I/O assumptions – just classes mirroring the schema – without the zarr dep.

     There are issues & PRs to this effect:

      - <https://github.com/ome-zarr-models/ome-zarr-models-py/issues/161>
      - <https://github.com/ome-zarr-models/ome-zarr-models-py/pull/280>

      but since `ome-zarr-models-py` also depends on
      [`pydantic-zarr`](https://github.com/zarr-developers/pydantic-zarr), that
      library will *also* need to be modified to remove the `zarr` dependency.

      - <https://github.com/zarr-developers/pydantic-zarr/pull/112>

  1. It currently pins to Python 3.11+ (presumably following NEP-29/SPEC-0),
     whereas I prefer to match the [official python EOL
     schedule](https://devguide.python.org/versions/) (supporting 3.10 until mid
     2026).

  1. Its inheritance and generics provide powerful abstractions, though for my
     experiments I wanted something simpler that just mirrors the spec.

  Ideally, this kind of minimal approach could help inform future directions for
  `ome-zarr-models-py`, and I’d be glad to see ideas converge over time.

- [pydantic-ome-ngff](https://github.com/janeliascicomp/pydantic-ome-ngff).
  *Deprecated.*
- [ngff-zarr](https://github.com/fideus-labs/ngff-zarr).  This also contains
  models, but brings along [far more
  dependencies](https://github.com/fideus-labs/ngff-zarr/blob/baafd774993d4a1dcfe312cfcd626c06496bb69d/py/pyproject.toml#L31-L43)
  and assumptions (and functionality) than `ome-zarr-models-py`.
- [ome-zarr](https://github.com/ome/ome-zarr-py).  This is a general toolkit,
  that provides functions for reading and writing OME-ZARR, among other things,
  but brings in many dependencies (zarr, scikit-image, dask,...) and doesn't export
  metadata models.

*In the meantime:*

This is an experimental package, where I can develop minimal models for my
applications.  The hope would be some future unification, provided the community
can agree on a common denominator of features.

Ultimately, I want a schema-first, I/O-second library.
