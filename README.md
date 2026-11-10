# yaozarrs ‼️

[![License](https://img.shields.io/pypi/l/yaozarrs.svg?color=green)](https://github.com/imaging-formats/yaozarrs/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/yaozarrs.svg?color=green)](https://pypi.org/project/yaozarrs)
[![Python Version](https://img.shields.io/pypi/pyversions/yaozarrs.svg?color=green)](https://python.org)
[![CI](https://github.com/imaging-formats/yaozarrs/actions/workflows/ci.yml/badge.svg)](https://github.com/imaging-formats/yaozarrs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imaging-formats/yaozarrs/branch/main/graph/badge.svg)](https://codecov.io/gh/imaging-formats/yaozarrs)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://app.codspeed.io/imaging-formats/yaozarrs?utm_source=badge)

***Yet Another Ome-ZARR Schema!***

`yaozarrs` is a Python library for OME-Zarr metadata and stores, built in three
layers. Take only the ones you need:

1. **Metadata.** Pydantic models for the [OME-NGFF
   spec](https://ngff.openmicroscopy.org/specifications/index.html) (v0.4, v0.5,
   v0.6), to create, validate, and convert JSON documents. No I/O at all.\
   *Needs:* only `pydantic`.
2. **Hierarchy.** Open and validate entire OME-Zarr stores (metadata *and*
   structure), with the `yaozarrs validate` and `yaozarrs tree` CLI.\
   *Needs:* nothing more if you fetch the documents yourself, `fsspec` for
   local paths, or `yaozarrs[io]` for http(s) and s3 URLs.
3. **Array data.** Write complete stores, or open arrays for reading, with the
   zarr library of your choice.\
   *Needs:* `zarr-python`, `tensorstore`, or your own writer.

## 📖 Documentation

Full documentation at <https://imaging-formats.github.io/yaozarrs/>

## Why yaozarrs?

Yes, it's yet another OME-Zarr library. It exists because, when I started it,
I couldn't find one that let me work with OME-Zarr *metadata* without also
requiring a particular zarr implementation for I/O.

NGFF is a metadata specification, and zarr itself has several implementations
(`zarr-python`, `tensorstore`, `acquire-zarr`, `zarrs`, ...). yaozarrs keeps
these concerns separate, so that each layer above adds only what it needs:

1. Whether you are ingesting metadata or producing it, you shouldn't need any
   I/O library to model and validate a JSON document.
2. Zarr *groups* are just JSON files in a key-value store, so checking that a
   hierarchy is structurally valid (paths exist, arrays have the right
   dimensions and dtypes, etc.) shouldn't require a full zarr array library.
   A library that implements its own zarr writer can use yaozarrs to check its
   output without depending on zarr-python.
3. When writing a store, the metadata is the part that varies, and the arrays
   are fairly stereotyped. yaozarrs writes the group metadata itself, and hands
   array creation to `zarr-python`, `tensorstore`, or a function you provide.
   It never picks your array library for you.

See [Principles](https://imaging-formats.github.io/yaozarrs/#principles) in the
docs for more, and [Related projects](#related-projects) below for how
yaozarrs compares to other libraries.

## Installation

```bash
pip install yaozarrs
# or
conda install -c conda-forge yaozarrs
```

Optional extras (pip only; the conda package includes none of them, so install
the dependencies yourself):

| extra | adds | enables |
| --- | --- | --- |
| `yaozarrs[io]` | `fsspec[http,s3]` | opening and validating local/remote stores, CLI |
| `yaozarrs[write-zarr]` | `zarr` | `yaozarrs.write` with `writer="zarr"`, `to_zarr_python()` |
| `yaozarrs[write-tensorstore]` | `tensorstore` | `yaozarrs.write` with `writer="tensorstore"`, `to_tensorstore()` |

> [!TIP]
> You may not need the `io` extra. It adds `aiohttp` and `s3fs` (with
> `botocore`) for remote URLs, but local paths need only `fsspec` (which has no
> dependencies), and if you do the I/O yourself you need nothing at all. See
> [Layer 2](#layer-2-hierarchy).

## Usage

(See also the [API Quick
Reference](https://imaging-formats.github.io/yaozarrs/#api-quick-reference) in
the docs.)

- **[Layer 1: metadata](#layer-1-metadata)**
  - [Construct OME-Zarr metadata](#construct-ome-zarr-metadata)
  - [Validate & load existing JSON documents](#validate--load-existing-json-documents)
  - [Validate arbitrary python objects](#validate-arbitrary-python-objects)
  - [Convert metadata between versions](#convert-metadata-between-versions)
- **[Layer 2: hierarchy](#layer-2-hierarchy)**
  - [Validate any zarr store using the CLI](#validate-any-zarr-store-using-the-cli)
  - [Validate any zarr store programmatically](#validate-any-zarr-store-programmatically)
  - [Do the I/O yourself](#do-the-io-yourself)
- **[Layer 3: array data](#layer-3-array-data)**
  - [Write OME-Zarr stores](#write-ome-zarr-stores)
  - [Open zarr arrays using zarr-python or tensorstore](#open-zarr-arrays-using-zarr-python-or-tensorstore)

### Layer 1: metadata

Everything in this section needs only `pydantic`, so downstream projects can
share a common metadata model without committing to any I/O library.

#### Construct OME-Zarr metadata

```python
from pathlib import Path

from yaozarrs import v05

scale = v05.Multiscale(
    name="scale0",
    axes=[
        v05.SpaceAxis(name="y", unit="micrometer"),
        v05.SpaceAxis(name="x", unit="micrometer"),
    ],
    datasets=[
        v05.Dataset(
            path="0",
            coordinateTransformations=[v05.ScaleTransformation(scale=[0.1, 0.1])],
        ),
        v05.Dataset(
            path="1",
            coordinateTransformations=[v05.ScaleTransformation(scale=[0.2, 0.2])],
        ),
    ],
)

img = v05.Image(multiscales=[scale])
zarr_json = v05.OMEZarrGroupJSON(attributes={"ome": img})
Path("zarr.json").write_text(zarr_json.model_dump_json(indent=2))
```

> [!TIP]
> For the common case, `v05.Multiscale.from_dims()` builds the axes and
> transformations from a list of `yaozarrs.DimSpec` objects.

#### Validate & load existing JSON documents

If you have an existing JSON document, you can validate and load it, and
benefit from IDE autocompletion and type hints.

```python
from pathlib import Path

import yaozarrs

obj = yaozarrs.validate_ome_json(Path("zarr.json").read_text())
print(type(obj))
# <class 'yaozarrs.v05._zarr_json.OMEZarrGroupJSON'>

print(repr(obj.attributes.ome.multiscales[0].axes[0]))
# SpaceAxis(name='y', type='space', unit='micrometer')
```

#### Validate arbitrary python objects

`validate_ome_object` and `validate_ome_json` accept a broad range of inputs,
and will cast to an appropriate model if possible.

```python
import yaozarrs

obj = yaozarrs.validate_ome_object({"version": "0.5", "series": ["0", "1"]})
print(repr(obj))
# Series(version='0.5', series=['0', '1'])
```

#### Convert metadata between versions

`convert_version` converts any metadata model to another version of the spec, keeping
its kind (an `Image` stays an `Image`).

```python
from yaozarrs import DimSpec, convert_version, v04

dims = [DimSpec(name="y", size=64), DimSpec(name="x", size=64)]
image = v04.Image(multiscales=[v04.Multiscale.from_dims(dims)])

print(type(convert_version(image, "0.6")))
# <class 'yaozarrs.v06._image.Image'>
```

Metadata that the target version can't express is dropped with a
`ConversionWarning`, which you can turn into an error using
`warnings.filterwarnings("error", category=yaozarrs.ConversionWarning)`.
Keys the spec doesn't define are carried over as they are.

> [!NOTE]
> This converts metadata only: no store is read or written. v0.5 and v0.6 stores
> differ only in their group metadata, but v0.4 stores use Zarr v2 while later
> versions use Zarr v3, so their arrays differ too.

### Layer 2: hierarchy

This layer reads zarr *groups* and array *metadata* (never chunk data), using a
small internal representation of zarr v2 and v3 hierarchies. It needs no zarr
implementation, and how much else it needs depends on how you hand it a store:

| store | install |
| --- | --- |
| any `Mapping[str, bytes]` of store keys (e.g. `"0/zarr.json"`) to metadata documents | nothing more: [you do the I/O](#do-the-io-yourself) |
| a local path, or a zarr-python `Group` | `fsspec` (which has no dependencies) |
| an http(s) or s3 URL | `pip install "yaozarrs[io]"` (adds `aiohttp` and `s3fs`) |

The CLI always takes a path or URL, so it needs at least `fsspec`.

#### Validate any zarr store using the CLI

The `yaozarrs validate` command checks that a store is valid OME-Zarr, both its
metadata and its structure. The store can be any URI: a local path, an http(s)
URL, an s3 URL, etc.

```bash
$ yaozarrs validate https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr
✓ Valid OME-Zarr store
  Version: 0.5
  Type: Image
```

> [!TIP]
> Use `uvx` to validate any URI without installing the package.
>
> ```bash
> uvx "yaozarrs[io]" validate https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr
> ```

`yaozarrs tree` shows the hierarchy of a store, along with the metadata model
found in each group:

```txt
$ yaozarrs tree https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr
🖼️ 6001240_labels.zarr
├── zarr.json  <- v05.Image
├── 📦 0 (uint16, (2, 236, 275, 271))
├── 📦 1 (uint16, (2, 236, 137, 135))
├── 📦 2 (uint16, (2, 236, 68, 67))
└── 🏷️ labels
    ├── zarr.json  <- v05.LabelsGroup
    └── 🖼️ 0
        ├── zarr.json  <- v05.LabelImage
        ├── 📦 0 (int8, (1, 236, 275, 271))
        ├── 📦 1 (int8, (1, 236, 137, 135))
        └── 📦 2 (int8, (1, 236, 68, 67))
```

##### Storage validation errors

Errors in the structure of the store (as opposed to its metadata) are collected
and presented like pydantic validation errors:

```txt
location
  description [context]
```

For example, for a deliberately broken store in this repo:

```bash
uvx "yaozarrs[io]" validate https://raw.githubusercontent.com/imaging-formats/yaozarrs/refs/heads/main/tests/data/broken/broken_v05.ome.zarr/
```

```txt
✗ Validation failed for: https://raw.githubusercontent.com/imaging-formats/yaozarrs/refs/heads/main/tests/data/broken/broken_v05.ome.zarr/

13 validation error(s) for StorageValidationError
ome.plate.wells.0.well.images.0.multiscales.0.datasets.0.path
  Dataset path '0' not found in zarr group [type=dataset_path_not_found, fs_path='broken_v05.ome.zarr/A/1/0/0', expected='zarr array']

ome.plate.wells.1.path
  Well path 'A/2' is not a zarr group [type=well_path_not_group, fs_path='broken_v05.ome.zarr/A/2', expected='group', found='array']

ome.plate.wells.3.well.images.0.multiscales.0.datasets.0.path
  Dataset '0' has 5 dimensions but axes specify 3 [type=dataset_dimension_mismatch, fs_path='broken_v05.ome.zarr/B/1/0/0', actual_ndim=5, expected_ndim=3, axes=['c', 'y', 'x']]

ome.plate.wells.3.well.images.0.labels.labels.0.multiscales.0.datasets.0.path
  Label array '0' has non-integer dtype 'float32'. Labels must use integer types. [type=label_non_integer_dtype, path='0', dtype='float32']

ome.plate.wells.7.well.images.0
  Field path '0' does not contain valid Image metadata [type=field_image_invalid, fs_path='broken_v05.ome.zarr/C/2/0']
  1 validation error for tagged-union[LabelImage,Image,Plate,Bf2Raw,Well,LabelsGroup,Series]
  image.multiscales
    Value should have at least 1 item after validation, not 0 [type=too_short, input_value=[], input_type=list]

...
```

#### Validate any zarr store programmatically

```python
import yaozarrs

url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr"

# raises a `StorageValidationError` describing everything that's wrong
group = yaozarrs.validate_zarr_store(url)

# or, open without validating, and inspect the hierarchy and its metadata
group = yaozarrs.open_group(url)
image = group.ome_metadata()  # a yaozarrs.v05.Image
print([axis.name for axis in image.multiscales[0].axes])
# ['c', 'z', 'y', 'x']
```

#### Do the I/O yourself

`validate_zarr_store` and `open_group` accept any `Mapping[str, bytes]`, so you
decide how metadata documents are fetched: a `dict` you filled up front, or a
mapping that fetches lazily. For example, this validates the same remote store
as above using only the standard library:

```python
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping

import yaozarrs


class HTTPStore(Mapping[str, bytes]):
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def __getitem__(self, key: str) -> bytes:
        try:
            with urllib.request.urlopen(f"{self.url}/{key}") as response:
                return response.read()
        except urllib.error.HTTPError as e:
            e.close()
            raise KeyError(key) from e  # (missing keys must raise KeyError)

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError  # (validation never lists the store)

    def __len__(self) -> int:
        raise NotImplementedError


url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr"
yaozarrs.validate_zarr_store(HTTPStore(url))
```

A mapping holds only what's inside the store, so a label's `image-source` or a
`Series`' parent group that lies outside it is skipped with a warning rather
than checked. See [Doing the IO
yourself](https://imaging-formats.github.io/yaozarrs/#doing-the-io-yourself) in
the docs.

### Layer 3: array data

yaozarrs doesn't choose an array library for you. It writes the group metadata
(`zarr.json`) itself, and delegates array creation and data I/O to
whichever backend you choose:

- `zarr-python`: `pip install "yaozarrs[write-zarr]"`
- `tensorstore`: `pip install "yaozarrs[write-tensorstore]"`
- your own: any function that creates an array and returns an object supporting
  numpy-style assignment (see
  [custom writers](https://imaging-formats.github.io/yaozarrs/API_Reference/yaozarrs.write.v05/#custom-writers))

#### Write OME-Zarr stores

`yaozarrs.write.v05` and `yaozarrs.write.v06` write complete stores (images,
labels, plates, bioformats2raw layouts, and v0.6 scenes) to a local directory,
from your metadata and array data.

```python
import numpy as np

from yaozarrs import DimSpec, v05, write

dims = [
    DimSpec(name="c"),
    DimSpec(name="z", scale=0.5, unit="micrometer"),
    DimSpec(name="y", scale=0.1, unit="micrometer"),
    DimSpec(name="x", scale=0.1, unit="micrometer"),
]
image = v05.Image(multiscales=[v05.Multiscale.from_dims(dims)])
data = np.zeros((2, 16, 256, 256), dtype=np.uint16)

write.v05.write_image("example.ome.zarr", image, data)
```

The default, `writer="auto"`, uses `tensorstore` if it's installed, and
`zarr-python` otherwise. Pass `writer="zarr"`, `writer="tensorstore"`, or your
own array-creation function to choose.

The `write_*` functions take all the data at once. When you don't have it all
up front, or want more control over how it's written, use the builders
(`PlateBuilder`, `LabelsBuilder`, `Bf2RawBuilder`, and `SceneBuilder` in v0.6):

- **incremental**: write one well, label, or series at a time (e.g.
  `PlateBuilder.write_well()`), and the parent metadata is updated after each
  one.
- **prepare-only**: register everything with `add_*()`, then `prepare()` creates
  the hierarchy with empty arrays, which you fill however you like (e.g. chunk
  by chunk, or in parallel). `prepare_image()` does the same for a single
  image.
- **create now, write later**: `Bf2RawBuilder.prepare_series()` adds one series
  at a time with empty arrays, for you to fill whenever you like. Each series
  is listed only once it's complete, so the collection is valid after every
  call, and you can keep adding series (e.g. as a model's output tiles arrive).
  A builder opened on an existing collection adds to it.

See the [writing docs](https://imaging-formats.github.io/yaozarrs/API_Reference/yaozarrs.write.v05/)
for details.

#### Open zarr arrays using zarr-python or tensorstore

yaozarrs itself never reads chunk data. Once you've opened (and validated) a
group from a path or URL, hand its arrays off to `zarr-python` or `tensorstore`
to read them. (These need a URI, so they don't work on groups opened from a
mapping.)

```python
from yaozarrs import open_group

group = open_group(
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr"
)
array = group["0"]
# <ZarrArray https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr/0>

ts_array = array.to_tensorstore()  # a tensorstore.TensorStore (needs tensorstore)
zarr_array = array.to_zarr_python()  # a zarr.Array (needs zarr)
```

## Related projects

These libraries also work with OME-Zarr metadata in Python, and one of them
may suit you better:

- [ome-zarr-models-py](https://github.com/ome-zarr-py/ome-zarr-models-py)
  provides pydantic models for OME-Zarr v0.4 to v0.6, and is what
  `ome-zarr-py` uses for its metadata. Its models represent zarr groups (built
  on [`pydantic-zarr`](https://github.com/zarr-developers/pydantic-zarr)), with
  the OME metadata as their attributes, and it reads and validates stores
  through `zarr-python`, which it requires. yaozarrs models instead mirror the
  spec's JSON documents one-to-one, so `v05.Image` is the `ome` object itself,
  not a zarr group that contains it.
- [ome-zarr-py](https://github.com/ome/ome-zarr-py) is an OME toolkit for
  reading and writing OME-Zarr images, with dependencies on `zarr`, `dask`,
  `scikit-image` and others.
- [ngff-zarr](https://github.com/fideus-labs/ngff-zarr) is a toolkit for
  creating and converting multiscale OME-Zarr images, with its own
  dataclass-based metadata models, and dependencies (currently) on `dask`,
  ITK-Wasm, zarrita, and others.
- [pydantic-ome-ngff](https://github.com/janeliascicomp/pydantic-ome-ngff) is
  archived.

Before starting yaozarrs, the idea of a zarr-optional model was raised with
ome-zarr-models-py
([#161](https://github.com/ome-zarr-py/ome-zarr-models-py/issues/161)), and
PRs attempted to make zarr-python optional there
([#280](https://github.com/ome-zarr-py/ome-zarr-models-py/pull/280)) and in
pydantic-zarr, which it builds on
([#112](https://github.com/zarr-developers/pydantic-zarr/pull/112), merged).
yaozarrs started as a way to see what a metadata-first library would look like,
and has since grown into its own library, with models that mirror the spec and
layered I/O that leaves the choice of array library to you.
