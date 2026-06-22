# v0.6 implementation — tricky decisions & validation gaps

Tracking record for the OME-NGFF **v0.6** (`0.6.dev4`) support added in
`yaozarrs.v06`. Source of truth: the `main` branch of
<https://github.com/ome/ngff-spec> and its open PRs (per issue #47 and
jo-mueller's note that the only *pending* modelling change is the
coordinates/displacements transforms).

Legend: 🟢 decided & implemented · 🟡 needs your call · 🔴 known gap vs spec.

---

## Big-picture decisions (please sanity-check)

- 🟢 **Version string is `"0.6.dev4"` only** (your call). `Literal["0.6.dev4"]`
  everywhere. This is a *dev* string and **will need bumping** when 0.6 is
  released (likely to `"0.6"`). Grep for `0.6.dev4` to find all sites.
- 🟢 **`coordinateSystems` replaces `axes`** (the RFC-5 redesign). On a
  `v06.Multiscale` there is **no `axes` field**; instead there is a required
  `coordinateSystems: list[CoordinateSystem]`, each holding `axes`. For v0.5
  parity we expose a **read-only `Multiscale.axes` property** that returns the
  *intrinsic* coordinate system's axes.
- 🟢 **Dataset transforms now carry `input`/`output`.** Each dataset has exactly
  one transform (`scale`, `identity`, or a `sequence` of scale+translation) with
  `input={"path": <level>}` and `output={"name": <coordinate-system>}`.
- 🟢 **Full transform zoo modeled** (`identity, mapAxis, scale, translation,
  affine, rotation, bijection, sequence, byDimension, displacements,
  coordinates`) as a discriminated union on `type`, with **v0.5-style pragmatic
  validation**. Gaps listed below.
- 🟢 **New `Scene` object** modeled (it's in the `ome_zarr` root union) but is
  **experimental** — see below.

---

## Per-field / per-model changes from v0.5

- 🟢 **`scale` values must be `> 0`** in v0.6 (`exclusiveMinimum: 0`). v0.5 placed
  no constraint. Modeled with `PositiveFloat`. *Behavior change*: a v0.5 doc with
  a `0` (or negative) scale factor will now fail.
- 🟢 **`scale` length constraint dropped** in v0.6 (v0.5 had `minItems: 2`).
  Length is instead validated against the axes by the containing `Multiscale`.
- 🟢 **Axis `type` is now optional and free-form** (no enum). New optional axis
  fields **`longName`** and **`discrete`**. New `type: "array"` axes exist for
  array/displacement coordinate systems.
- 🟢 **`image-label` no longer has its own `version`** (the inner version field
  was removed; version lives once at the `ome` level). `v06.ImageLabel` has no
  `version` field (v05 did).
- 🟢 **`image-label` is now *required*** on a label image (`label.schema`
  `required: ["image-label", "version"]`). `v06.LabelImage.image_label` is
  required (same as v05's model, but now schema-mandated).
- 🟢 **Plate / Well / Bf2Raw / Series / Omero are structurally identical** to
  v0.5 — only the `version` string changed. These modules are near-verbatim
  copies of the v05 ones.

---

## 🔴 Where we do NOT fully enforce the spec (pragmatic gaps)

These are deliberate (per "use v0.5 pragmatism as the guide"). Candidates for a
future "strict mode".

1. **Axes structural rules are only applied when fully typed.** The 2-3 space /
   ≤1 time / ≤1 channel / ordering rules are enforced **only when every axis has
   a recognized `space`/`time`/`channel` type.** Real v0.6 examples (e.g.
   `label_strict/colors_properties.json`) use *type-less* axes
   (`{"name":"x","unit":"micrometer"}`); hard-failing those would reject valid
   spec examples. (RFC-3 / PR #75 "Unconstrained Axes" is moving this way.)
2. **Affine / rotation matrix shapes are not validated.** We only check that
   exactly one of the inline matrix or `path` is given. The spec restricts
   `rotation` to a square NxN matrix (N in 2..5) and ties affine/rotation
   dimensionality to the axes — not enforced.
3. **Coordinate-system graph connectivity is not validated.** The spec says the
   graph of coordinate systems (via `coordinateTransformations`) MUST be fully
   connected. We don't check this.
4. **Multiscale-level `coordinateTransformations` are weakly validated.** We
   accept the transform list but do **not** verify that referenced
   `input.name`/`output.name` coordinate systems exist, that dimensionality
   matches, or that an `output.path` obeys the downward-only relative-path rule
   (`^(\.\./|/)` forbidden).
5. **Scene transforms are weakly validated.** We require `input.name` and
   `output.name` to be present, but do **not** check that they reference declared
   coordinate systems, nor connectivity, nor path constraints.
6. **`byDimension` axis references** (`input_axes`/`output_axes`) are typed as
   `list[float]` (the *schema* types them as `number`, even though the prose
   describes them as axis names/indices). We don't validate that they reference
   real axes. (Recorded discrepancy between schema and prose.)
7. **Coordinate-system `name` uniqueness is only structural.** `coordinateSystems`
   uses `UniqueList` (whole-object uniqueness). Two coordinate systems with the
   **same `name` but different `axes`** would currently pass; the spec wants
   names unique across the whole document.
8. **Dataset `sequence` ordering is stricter than the schema.** We require
   exactly one `scale` *followed by* one `translation` (matching the prose
   description). The schema's `sequence` variant only requires two items, each a
   `oneOf[scale, translation]` (no order enforced). So we are slightly *stricter*
   here, not looser.

---

## Tolerances (we accept MORE than the schema)

- 🟢 **Bare-string `input`/`output` shorthand.** Some spec examples write
  `"input": "in"` instead of `"input": {"name": "in"}` (e.g.
  `transformations/mapAxis1.json`, inconsistently within one file). The schema
  requires the object form. We coerce a bare string to `{"name": <str>}` on input
  and always *emit* the object form.

---

## Experimental / unstable surface

- 🟡 **`Scene`** is brand-new and the spec around it is actively changing:
  - `arrayCoordinateSystem` is slated for **removal** (PR #151). We model it
    (loosely) only to round-trip existing docs.
  - `scene.schema` **omits `version`** under its `ome` object (likely an
    oversight). For consistency with every other v0.6 doc we keep a `version`
    field on `v06.Scene`, defaulting to `"0.6.dev4"`. **If you'd rather mirror
    the schema exactly, drop it.**
  - `input.path`/`output.path` constraints are being reworked (PRs #149, #137).
- 🔴 **xarray-like dict-form axes** (`"axes": {"t": {...}, "c": {...}}`, seen in
  `multiscales_example_relative.json` and `xarrayLike.json`/`byDimensionXarray.json`)
  are **not supported** — they contradict `axes.schema` (which requires a JSON
  array). Treated as an experimental proposal.
- 🔴 **displacements/coordinates "as multiscales"** (PR #145, prose-only, no
  schema change) — not modeled specially; we model the transform objects per the
  current schema.
- Pending transform **`projectAxis`** (PR #130) is not yet in the main schema, so
  not modeled.

---

## Things validated against the real spec examples

All of `ome/ngff-spec/examples/**` (downloaded from `main`) parse/validate as
expected, except the intentionally-experimental / intentionally-invalid ones
noted above:
`multiscales_strict/*`, `label_strict/colors_properties`, `plate_strict/*`,
`well_strict/*`, `scene/*`, `ome/series-2`, and 12 `transformations/*`.
