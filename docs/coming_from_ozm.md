---
icon: material/swap-horizontal
title: Coming from ome-zarr-models
---

# Coming from ome-zarr-models

This page is for porting code from
[ome-zarr-models-py](https://github.com/ome-zarr-py/ome-zarr-models-py) to
yaozarrs.

Most of the differences come from one design choice. An ome-zarr-models class
(`Image`, `HCS`, ...) is a whole zarr *group*: its attributes plus its members.  A
yaozarrs model is only the OME *metadata document* (the value of the `"ome"` key
in `zarr.json`, or the `.zattrs` of a v0.4 group).  The store itself is read
through [`open_group`][yaozarrs.open_group] and
[`validate_zarr_store`][yaozarrs.validate_zarr_store], which accept a path, a URL,
or a zarr-python `Group` or `Store`.

## Reading a store

```python
import numpy as np

from yaozarrs import DimSpec, open_group, v05, validate_zarr_store
from yaozarrs.write.v05 import write_image

dims = [DimSpec(name="y", scale=0.5, unit="micrometer"), DimSpec(name="x")]
image = v05.Image(multiscales=[v05.Multiscale.from_dims(dims)])
write_image("example.ome.zarr", image, np.zeros((64, 64), dtype="uint8"))

# validate the whole hierarchy (metadata *and* the arrays it points to)
group = validate_zarr_store("example.ome.zarr")
image = group.ome_metadata()  # a v05.Image

# or read just this group's metadata, without validating the hierarchy
image = open_group("example.ome.zarr").ome_metadata()

array = group["0"]  # a ZarrArray, without reading any data
print(array.shape, array.dtype, array.dimension_names)
```

| ome-zarr-models | yaozarrs |
| --------------- | -------- |
| `open_ome_zarr(group)` | `validate_zarr_store(group).ome_metadata()` |
| `Image.from_zarr(group)` | `validate_zarr_store(group).ome_metadata()`, then check that it's an `Image` |
| `image.ome_attributes` | the model returned by `.ome_metadata()` |
| `image.attributes` | `group.attrs` |
| `image.labels` | `group["labels"].ome_metadata()` |
| `hcs.get_well_group(i)`, `hcs.well_groups` | `group["A/1"].ome_metadata()` (no typed traversal yet) |
| an array in `image.members` | `group["0"]`: `.shape`, `.dtype`, `.dimension_names`, and the full zarr metadata in `.metadata` (there's no `.chunks`) |

Like ome-zarr-models, `validate_zarr_store` also checks the arrays the metadata
points to (that they exist, and match the axes).  `open_group(group).ome_metadata()`
is cheaper, but only checks the metadata document itself.

To get the data, call `.to_zarr_python()` or `.to_tensorstore()` on the array.  For
writing, see [Writing OME-Zarr](writing.md).

## Class names

An ome-zarr-models class like `Image` models the whole zarr group, and its
`...Attrs` class models the group's OME metadata.  A yaozarrs model is only the
metadata, so it corresponds to the `...Attrs` class, even where the names match.
(The group itself is a [`ZarrGroup`][yaozarrs.ZarrGroup].)  The names are the
same in v0.4, v0.5 and v0.6.

| ome-zarr-models group | ome-zarr-models attributes | yaozarrs |
| --------------------- | -------------------------- | -------- |
| `Image` | `ImageAttrs` | `Image` |
| `ImageLabel` | `ImageLabelAttrs` | `LabelImage` |
| `Labels` | `LabelsAttrs` | `LabelsGroup` |
| `HCS` | `HCSAttrs` | `Plate` |
| `Well` | `WellAttrs` | `Well` |
| `BioFormats2Raw` | `BioFormats2RawAttrs` | `Bf2Raw` |
| v0.6 `Scene` | v0.6 `SceneAttrs` | v0.6 `Scene` |

The objects nested inside those documents:

| ome-zarr-models | yaozarrs |
| --------------- | -------- |
| `Label` (the `image-label` object) | `ImageLabel` |
| `Plate` (the `plate` object) | `PlateDef` |
| `WellMeta` (the `well` object) | `WellDef` |
| v0.6 `Scale`, `Affine`, ... | `ScaleTransformation`, `AffineTransformation`, ... |
| v0.6 `CoordinateSystemIdentifier` | `CoordinateSystemRef` |

Watch out for `ImageLabel` and `Plate`, which mean different things in the two
libraries.

## Label images are images

A yaozarrs `LabelImage` is a subclass of `Image` (in ome-zarr-models, `ImageLabel`
and `Image` are unrelated).  So `isinstance(obj, v05.Image)` is also true for
label images.  If you branch on type, check for `LabelImage` first.

## Omero metadata

ome-zarr-models' `Omero` model only declares `channels`, and its `Channel` only
`color` and `window`.  Everything else (the omero `name` and `rdefs`, a channel's
`label` and `active`, ...) ends up in `model_extra`.  On a v0.5 `Image` there's no
`omero` field at all, so the whole `omero` object is a plain dict in `model_extra`.

yaozarrs declares all of those as fields, in v0.4 and v0.5, so they're no longer
in `model_extra`:

| ome-zarr-models | yaozarrs |
| --------------- | -------- |
| `channel.model_extra["label"]` | `channel.label` |
| `channel.model_extra["active"]` | `channel.active` |
| `omero.model_extra["rdefs"]` | `omero.rdefs` |
| v0.5: `image.model_extra["omero"]` (a dict) | `image.omero` (an `Omero` model) |

Code that looks these up in `model_extra` won't fail, it just won't find them, so
check for it when porting.

## Coordinate transformations (v0.6)

ome-zarr-models' v0.6 transforms have `as_affine()`, `transform_point()` and
`get_inverse()`, and share a `Transform` base class.  yaozarrs' transforms only
describe the metadata: there's no transform math, so you'll need your own (e.g.
building the affine matrix from a `ScaleTransformation`'s `scale`).  There's no
public base class either: `CoordinateTransformation` is a union type, for
annotations and parsing.  To branch on the kind of transform, check `t.type` or
the concrete classes (`isinstance(t, v06.ScaleTransformation)`).

## Exceptions

ome-zarr-models' `open_ome_zarr` raises `RuntimeError` when a group doesn't
validate as any OME-Zarr group.  In yaozarrs, every validation failure is a
`ValueError`:

- `validate_zarr_store` raises a
  [`StorageValidationError`][yaozarrs.StorageValidationError] (a `ValueError`)
  for invalid metadata or stores, and a `ValueError` for a group that isn't
  OME-Zarr at all.
- `open_group(...).ome_metadata()` raises pydantic's `ValidationError` (also a
  `ValueError`) for invalid metadata, and returns `None` for a group with no OME
  metadata.

## Models are mutable

ome-zarr-models' models are frozen (and so hashable).  yaozarrs models can be
changed after creation (each assignment is validated), so they aren't hashable.
If you used models as dict keys, set members, or graph nodes, key them by
something else.

ome-zarr-models also uses tuples for `Multiscale.datasets` and for the
`coordinateTransformations` of a `Multiscale` and a `Dataset`.  yaozarrs uses
lists everywhere.

## Dumping to JSON

yaozarrs' `model_dump()` and `model_dump_json()` leave out fields that are `None`
(pass `exclude_none=False` to keep them).  So dumped metadata no longer has the
`"name": null`, `"type": null` and `"metadata": null` entries that ome-zarr-models
wrote.

## Validation results

Some stores that validated with ome-zarr-models now fail, and the reverse.  For
example:

| store | ome-zarr-models | yaozarrs |
| ----- | --------------- | -------- |
| v0.5 array without `dimension_names` | error | warning |
| plate listing a well that doesn't exist | valid | error |
| group with both `plate` and `multiscales` metadata | a plate | an image |
| plate with empty `rows`, `columns` and `wells` | valid | error |

A v0.5 `plate` without a `version` is valid in both, but ome-zarr-models fills in
`plate.version = "0.5"` (with a warning), and yaozarrs doesn't.  So
`plate.plate.version` raises `AttributeError` there.  Use the `version` of the
`Plate` itself (`plate.version`) instead.

Findings that aren't errors (e.g. spec "SHOULD"s) are reported as
[`StorageValidationWarning`][yaozarrs.StorageValidationWarning]s, through Python's
`warnings` module.  To collect them (for a report, say) instead of having them
printed:

```python
import warnings

from yaozarrs import StorageValidationWarning, validate_zarr_store

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", StorageValidationWarning)
    group = validate_zarr_store("example.ome.zarr")

findings = [
    detail  # a dict with "type", "loc", "msg" and "ctx"
    for w in caught
    if isinstance(w.message, StorageValidationWarning)
    for detail in w.message.warnings()
]
```

## Writing

yaozarrs writes v0.5 and v0.6 (zarr v3) only.  There's no v0.4 (zarr v2) writer
like ome-zarr-models' `v04.Image.new(...).to_zarr(...)`.  To write v0.4, create the
arrays with zarr-python and write the metadata yourself:

```python
import zarr

from yaozarrs import DimSpec, v04

dims = [DimSpec(name="y"), DimSpec(name="x")]
image = v04.Image(multiscales=[v04.Multiscale.from_dims(dims)])
root = zarr.open_group("example_v04.ome.zarr", mode="w", zarr_format=2)
root.attrs.update(image.model_dump())
for dataset in image.multiscales[0].datasets:
    root.create_array(dataset.path, shape=(64, 64), dtype="uint8", chunks=(64, 64))
```

ome-zarr-models' `Image.new` takes pydantic-zarr `ArraySpec`s, which carry their
own codecs.  yaozarrs' `compression` only picks one of `"blosc-zstd"`,
`"blosc-lz4"`, `"zstd"` or `"none"`.  For other codecs, pass your own `writer`
function (a [`CreateArrayFunc`][yaozarrs.write._common._arrays.CreateArrayFunc]),
which creates each array however you like.
