# yaozarrs ‼️

[![License](https://img.shields.io/pypi/l/yaozarrs.svg?color=green)](https://github.com/tlambert03/yaozarrs/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/yaozarrs.svg?color=green)](https://pypi.org/project/yaozarrs)
[![Python Version](https://img.shields.io/pypi/pyversions/yaozarrs.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/yaozarrs/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/yaozarrs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/yaozarrs/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/yaozarrs)

***Yet Another Ome-ZARR Schema!***

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

*In the meantime:*

This is an experimental package, where I can develop minimal models for my
applications.  The hope would be some future unification, provided the community
can agree on a common denominator of features.

Ultimately, I want a schema-first, I/O-second library.
