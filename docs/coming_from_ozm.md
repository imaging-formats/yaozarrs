---
icon: material/swap-horizontal
title: Coming from ome-zarr-models
---

# Coming from ome-zarr-models

This page is for porting code from
[ome-zarr-models-py](https://github.com/ome-zarr-py/ome-zarr-models-py) to
yaozarrs.

!!! warning "Not a drop-in replacement"
    yaozarrs and ome-zarr-models were developed independently, and each interprets
    the OME-Zarr specification and its JSON schemas on its own.  Some metadata and
    stores that ome-zarr-models accepts are invalid in yaozarrs, and the reverse.
    Re-validate your data and test fixtures after porting.
    [Validation results](#validation-results) lists the known differences.

Most of the differences come from one design choice.  An ome-zarr-models class
(`Image`, `HCS`, ...) is a whole zarr *group*: its attributes plus its members.  A
yaozarrs model is only the OME *metadata document* (the value of the `"ome"` key
in `zarr.json`, or the `.zattrs` of a v0.4 group).  The store itself is read
through [`open_group`][yaozarrs.open_group] and
[`validate_zarr_store`][yaozarrs.validate_zarr_store], which accept a path, a URL,
or a zarr-python `Group` or `Store`.

Some features require optional extras (see [Installation](installation.md)).

## Reading a store

The reading API is described in [Open and validate a store](index.md#open-and-validate-a-store)
and [Read array data](index.md#read-array-data).

In the table, `zarr_group` is what you passed to ome-zarr-models, and `group` is
the [`ZarrGroup`][yaozarrs.ZarrGroup] that `validate_zarr_store` returns.

| ome-zarr-models | yaozarrs |
| --------------- | -------- |
| `open_ome_zarr(zarr_group)` | `validate_zarr_store(zarr_group).ome_metadata()` |
| `Image.from_zarr(zarr_group)` | `validate_zarr_store(zarr_group).ome_metadata()`, then check that it's an `Image` (see [Label images are images](#label-images-are-images)) |
| `image.ome_attributes` | the model returned by `.ome_metadata()` |
| `image.attributes` | `group.attrs` |
| `image.ome_zarr_version` | `group.ome_version()` |
| `image.labels` | `group.get("labels")` (`None` if absent), then `.ome_metadata()` |
| `hcs.get_well_group(i)`, `hcs.well_groups` | `group[plate.plate.wells[i].path].ome_metadata()`, where `plate = group.ome_metadata()` |
| `scene.images`, or any listing of child groups | `group.to_zarr_python().group_keys()` (a `ZarrGroup` can't list its children) |
| an array in `image.members` | `group["0"]`: `.shape`, `.dtype`, `.dimension_names`, and `.metadata` (the array's `zarr.json` or `.zarray`, with the same field names as pydantic-zarr's `ArraySpec`, e.g. `chunk_grid` and `codecs`, or `chunks` for zarr v2) |

Like ome-zarr-models, `validate_zarr_store` checks the arrays the metadata points to
(that they exist, and match the axes).  `open_group(zarr_group).ome_metadata()` is
cheaper, but only checks the metadata document itself.

## Class names

A yaozarrs model corresponds to an ome-zarr-models `*Attrs` class, even where
the names match.  These names are the same in v0.4, v0.5 and v0.6, and each
version's fields are listed in the API reference
([v0.4](API_Reference/yaozarrs.v04.md), [v0.5](API_Reference/yaozarrs.v05.md),
[v0.6](API_Reference/yaozarrs.v06.md)).

| ome-zarr-models group | ome-zarr-models attributes | yaozarrs |
| --------------------- | -------------------------- | -------- |
| `Image` | `ImageAttrs` | `Image` |
| `ImageLabel` | `ImageLabelAttrs` | `LabelImage` |
| `Labels` | `LabelsAttrs` | `LabelsGroup` |
| `HCS` | `HCSAttrs` | `Plate` |
| `Well` | `WellAttrs` | `Well` |
| `BioFormats2Raw` | `BioFormats2RawAttrs` | `Bf2Raw` |
| v0.6 `Scene` | v0.6 `BaseSceneAttrs` | v0.6 `Scene` |

The objects nested inside those documents:

| ome-zarr-models | yaozarrs |
| --------------- | -------- |
| `Label` (the `image-label` object) | `ImageLabel` |
| `Plate` (the `plate` object) | `PlateDef` |
| `WellMeta` (the `well` object) | `WellDef` |
| v0.6 `SceneAttrs` (the `scene` object) | `SceneDef` |
| v0.4/v0.5 `VectorScale`, `PathScale` | `ScaleTransformation` (with `scale` or `path` set) |
| v0.4/v0.5 `VectorTranslation`, `PathTranslation` | `TranslationTransformation` (with `translation` or `path` set) |
| v0.6 `Scale`, `Affine`, ... | `ScaleTransformation`, `AffineTransformation`, ... |
| v0.6 `CoordinateSystemIdentifier` | `CoordinateSystemRef` |

`ImageLabel`, `Plate` and `SceneAttrs` name different objects in the two
libraries.

## Label images are images

A yaozarrs `LabelImage` is a subclass of `Image` (in ome-zarr-models, `ImageLabel`
and `Image` are unrelated).  So `isinstance(obj, v05.Image)` is also true for
label images.  If you branch on type, check for `LabelImage` first.

## Omero metadata

ome-zarr-models' `Omero` model only declares `channels`, and its `Channel` only
`color` and `window`.  Everything else (the omero `name` and `rdefs`, a channel's
`label` and `active`, ...) is in `model_extra`.  A v0.5 or v0.6 `ImageAttrs` has
no `omero` field, so `image.omero` there is a plain dict.

yaozarrs declares all of these as fields (see
[`Omero`][yaozarrs.v05.Omero] and [`OmeroChannel`][yaozarrs.v05.OmeroChannel]), in
v0.4, v0.5 and v0.6, so `model_extra` no longer contains them:

| ome-zarr-models | yaozarrs |
| --------------- | -------- |
| `channel.model_extra["label"]` | `channel.label` |
| `channel.model_extra["active"]` | `channel.active` |
| `omero.model_extra["rdefs"]` | `omero.rdefs` |
| v0.5/v0.6: `image.omero["channels"]` (a dict) | `image.omero.channels` (an `Omero` model) |

`model_extra["label"]` raises `KeyError` in yaozarrs, and `model_extra.get("label")`
returns `None`.

## Coordinate transformations (v0.6)

ome-zarr-models' v0.6 transforms have `as_affine()`, `transform_point()` and
`get_inverse()`, and share a `Transform` base class.  yaozarrs' transforms only
describe the metadata, with no transform math.

There's no public base class.  `CoordinateTransformation` is an annotated union:
use it in annotations, or parse with
`pydantic.TypeAdapter(CoordinateTransformation)`, but not with `isinstance`
(it raises `TypeError`).  To branch on the kind of transform, check `t.type` or
the [concrete classes](API_Reference/yaozarrs.v06.md#transformations)
(`isinstance(t, v06.ScaleTransformation)`).

## Exceptions

ome-zarr-models' `open_ome_zarr` raises `RuntimeError` when a group doesn't
validate as any OME-Zarr group.  In yaozarrs, every validation failure is a
`ValueError`, so replace `except RuntimeError` with `except ValueError`:

- `validate_zarr_store` raises a
  [`StorageValidationError`][yaozarrs.StorageValidationError] (a `ValueError`)
  for invalid metadata or stores, and a `ValueError` for a group that isn't
  OME-Zarr at all.
- `open_group(...).ome_metadata()` raises a `ValueError` (usually pydantic's
  `ValidationError`) for invalid metadata, and returns `None` for a group with no
  OME metadata.

## Models are mutable

ome-zarr-models' models are frozen, and a few are hashable (`Axis`, the omero
`Channel`, and the v0.6 coordinate systems and transforms).  yaozarrs models can be
changed after creation (each assignment is validated), and none are hashable.

ome-zarr-models uses tuples for `Multiscale.datasets`, for `coordinateTransformations`,
and in v0.6 also for `coordinateSystems` and the transform parameters.  yaozarrs
uses lists.

## Dumping to JSON

yaozarrs' `model_dump()` and `model_dump_json()` leave out fields that are `None`
(pass `exclude_none=False` to keep them).  So dumped metadata no longer has the
`null` entries (`"name": null`, `"unit": null`, ...) that ome-zarr-models wrote.

## Validation results

| store | ome-zarr-models | yaozarrs `validate_zarr_store` |
| ----- | --------------- | ------------------------------ |
| v0.5 array without `dimension_names` | error | warning |
| plate listing a well that doesn't exist | valid | error |
| plate with empty `rows`, `columns` and `wells` | valid | error |
| v0.6 scene transform whose `input` or `output` has no `name`, or has no `path` and names a coordinate system that isn't in the scene's `coordinateSystems` | valid | error |
| v0.6 scene with an empty `coordinateTransformations` | valid | error |

A v0.5 `plate` object without a `version` is valid in both, but ome-zarr-models
fills in `plate.version = "0.5"` (with a warning), and yaozarrs doesn't.  So in
yaozarrs, `plate.plate.version` raises `AttributeError` when the `plate` object has
no `version`.  Use the `version` of the `Plate` itself (`plate.version`).

Findings that aren't errors (e.g. spec "SHOULD"s) are reported as
[`StorageValidationWarning`][yaozarrs.StorageValidationWarning]s, through Python's
`warnings` module.  To collect them instead of having them printed:

<!-- no-test -->
```python
import warnings

from yaozarrs import StorageValidationWarning, validate_zarr_store

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", StorageValidationWarning)
    group = validate_zarr_store("example.ome.zarr")

findings = [
    detail  # a dict with "type", "loc", "msg" (and "ctx", when there is one)
    for w in caught
    if isinstance(w.message, StorageValidationWarning)
    for detail in w.message.warnings()
]
```

## Writing

yaozarrs writes v0.5 and v0.6 (zarr v3) only.  There's no v0.4 (zarr v2) writer
like ome-zarr-models' `v04.Image.new(...).to_zarr(...)`.  To write v0.4, write the
metadata yourself (`image.model_dump()` is the `.zattrs` content) and create the
arrays with any zarr v2 library.  With zarr-python:

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
own codecs.  The `compression` argument of yaozarrs' writers
([`write_image`][yaozarrs.write.v05.write_image],
[`prepare_image`][yaozarrs.write.v05.prepare_image], ...) only accepts
`"blosc-zstd"`, `"blosc-lz4"`, `"zstd"` or `"none"`.  For other codecs, pass a
`writer=` function that creates each array (see
[Custom writers](writing.md#custom-writers)).
