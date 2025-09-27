# yaozarrs❗

[![CI](https://github.com/tlambert03/yaozarrs/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/yaozarrs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/yaozarrs/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/yaozarrs)

***Yet Another Ome-ZARR Schema!***

> [!IMPORTANT]
> **Don't use this "in production".**
>
> Feel free to copy, vendor, whatever. Read below for details.
>
> You should first check out [ome-zarr-models-py](https://github.com/ome-zarr-models/ome-zarr-models-py)

## Oh no, not another one 🤦

First, let me apologize. The last thing the world needs is yet another ome-zarr
model. However, I was unable to find a *minimal* ome-zarr model that simply
represents the spec, without introducing additional I/O features or
dependencies.

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

## Installation

```bash
pip install git+https://github.com/tlambert03/yaozarrs
```

## Usage

Validate any object against the OME-NGFF schema,
where "object" here refers to any dict or JSON object that could
live at the "ome" key of an ome-zarr file.

```python
import yaozarrs

# validate from a dict or other python object:
obj = yaozarrs.validate_ome_object(...)

# or to validate from a JSON string | bytes:
obj = yaozarrs.validate_ome_json(json_string)
```

You can also construct objects directly from python, with IDE autocompletion:

```python
from yaozarrs import v05

scale = v05.Multiscale(
    name="scale0",
    axes=[v05.SpaceAxis(name="x", type="space"), v05.SpaceAxis(name="y", type="space")],
    datasets=[
        v05.Dataset(
            path="0",
            coordinateTransformations=[v05.ScaleTransformation(scale=[0, 1])],
        ),
        v05.Dataset(
            path="1",
            coordinateTransformations=[v05.ScaleTransformation(scale=[0, 1])],
        ),
    ],
)

img = v05.Image(multiscales=[scale])
```

and of course, from dicts:

```python
from yaozarrs import validate_ome_object, v05

obj = {
    'version': '0.5',
    'multiscales': [
        {
            'name': 'scale0',
            'axes': [{'name': 'x', 'type': 'space'}, {'name': 'y', 'type': 'space'}],
            'datasets': [
                {
                  'path': '0', 
                  'coordinateTransformations': [{'type': 'scale', 'scale': [0.0, 1.0]}],
                },
                {
                  'path': '1', 
                  'coordinateTransformations': [{'type': 'scale', 'scale': [0.0, 1.0]}],
                },
            ],
        }
    ],
}

node = validate_ome_object(obj)

assert isinstance(node, v05.Image)
```
