# v0.6 implementation — tricky decisions & validation gaps

Tracking record for the OME-NGFF **v0.6** (`0.6.dev4`) support added in
`yaozarrs.v06`. Source of truth: the `main` branch of
<https://github.com/ome/ngff-spec> and its open PRs (per issue #47 and
jo-mueller's note that the only *pending* modelling change is the
coordinates/displacements transforms).

Legend: 🟢 decided & implemented · 🟡 needs your call · 🔴 known gap vs spec.

## Validation policy

The **JSON schema is the first source of truth**, supplemented by the normative
prose in `index.md`. Per the ngff-spec developers, **any spec _example_ that
conflicts with the schema or prose is wrong** (the examples will be fixed
upstream). So we validate to the schema+prose and do *not* relax rules to make a
non-conforming example pass. Where our local test fixtures violated the schema we
fixed our copies (typed the axes; dropped two fabricated `transformations/`
files; moved the `reference_to_label` labels link from `input` to `output`).

---

## Big-picture decisions (please sanity-check)

- 🟢 **Dataset transforms now carry `input`/`output`.** Each dataset has exactly
  one transform (`scale`, `identity`, or a `sequence` of scale+translation) with
  `input={"path": <level>}` and `output={"name": <coordinate-system>}`. All
  datasets in a multiscale **MUST** share the *same* `output.name` (enforced in
  `Multiscale._post_validate`; the spec requires one intrinsic coordinate system
  per multiscale).
- 🟢 **Full transform zoo modeled** (`identity, mapAxis, scale, translation,
  affine, rotation, bijection, sequence, byDimension, displacements,
  coordinates`) as a discriminated union on `type`. Remaining gaps listed below.
- 🟢 **New `Scene` object** modeled (it's in the `ome_zarr` root union) but is
  **experimental** — see below.

---

## Per-field / per-model changes from v0.5

- 🟢 **`scale` values must be `> 0`** in v0.6 (`exclusiveMinimum: 0`). v0.5 placed
  no constraint. Modeled with `PositiveFloat`. *Behavior change*: a v0.5 doc with
  a `0` (or negative) scale factor will now fail.
- 🟢 **`scale` length constraint dropped** in v0.6 (v0.5 had `minItems: 2`).
  Length is instead validated against the axes by the containing `Multiscale`.
- 🟢 **An individual axis `type` is a free-form string** (no enum in the
  non-strict `axes.schema`). New optional axis fields **`longName`** and
  **`discrete`**. But the *array* as a whole MUST satisfy the `axes.schema`
  `oneOf`: **either 2-3 `space` axes or ≥2 `array` axes** — this is enforced for
  *every* coordinate system (`_axes._validate_axes_list`), so a fully type-less
  axes array is invalid. The `space` branch also enforces the prose rules (≤1
  time, ≤1 channel/custom, and time→channel/custom→space ordering).
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

## Enforced (no longer hedged)

- 🟢 **Axes structural rules** — the `axes.schema` `oneOf` (2-3 `space` or ≥2
  `array`) plus the prose rules (≤1 time, ≤1 channel/custom, ordering) are now
  enforced for **every** coordinate system at the field level
  (`_axes._validate_axes_list`). Type-less axes arrays are rejected.
- 🟢 **Coordinate-system `name` uniqueness** — validated directly on
  `CoordinateSystems` (both `Multiscale` and `Scene`), replacing whole-object
  `UniqueList` (which missed same-`name`/different-`axes` collisions). *Document*-
  wide uniqueness (names shared between a multiscale and a sibling scene) is still
  not cross-checked.
- 🟢 **`input`/`output` object form required** — the schema requires the object
  form (`{"name": "in"}`); the bare-string shorthand (`"input": "in"`) is no
  longer accepted (was an example-driven tolerance; the example is wrong).
- 🟢 **Multiscale-level `coordinateTransformations` `input.path` SHOULD warning**
  — restored. (The `reference_to_label` example that contradicted this is wrong
  and our fixture copy is fixed: the labels link moved to `output`.)

## 🔴 Where we do NOT fully enforce the spec (remaining gaps)

1. **Affine / rotation matrix shapes are not validated.** We only check that
   exactly one of the inline matrix or `path` is given. The spec restricts
   `rotation` to a square NxN matrix (N in 2..5) and ties affine/rotation
   dimensionality to the axes — not enforced.
2. **Coordinate-system graph connectivity is not validated.** The spec says the
   graph of coordinate systems (via `coordinateTransformations`) MUST be fully
   connected. We don't check this.
3. **Multiscale-level `coordinateTransformations` — partially validated.** We DO
   enforce that `input.name == intrinsic`, that `output` references a declared CS
   (or a labels link via `name`+`path`), and that labels-link transforms are
   identity/scale/translation. We do **not** verify dimensionality matching or
   that an `output.path` obeys the downward-only relative-path rule
   (`^(\.\./|/)` forbidden).
4. **Scene transforms — partially validated.** We require `input.name` and
   `output.name` to be present (field-level on `SceneDef.coordinateTransformations`)
   but do **not** check that they reference declared coordinate systems, nor
   connectivity, nor path constraints.
5. **`byDimension` axis references** (`input_axes`/`output_axes`) are typed as
   `list[float]` (the *schema* types them as `number`, even though the prose
   describes them as axis names/indices). We don't validate that they reference
   real axes. (Recorded discrepancy between schema and prose.)
6. **`mapAxis` does not enforce `uniqueItems`.** The schema marks the index list
   `uniqueItems: true`; we enforce the bounds/length but allow duplicate indices.

Note: **dataset `sequence` ordering is intentionally *stricter* than the schema**
— we require exactly one `scale` *followed by* one `translation` (per the prose);
the schema's `sequence` variant only requires two `oneOf[scale, translation]`
items. Stricter, not looser, so it stays.

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

Our fixture copies under `tests/data/v06/examples/**` were taken from
`ome/ngff-spec/examples` on `main`, then **corrected where the upstream example
violated the schema** (per the policy above): the type-less axes in
`label_strict/colors_properties`, `multiscales_strict/multiscales_reference_to_label`,
and the `transformations/*` coordinate systems were given `type: "space"`;
`mapAxis1`'s bare-string `input`/`output` were objectified; the labels link in
`reference_to_label` moved to `output`; and the two fabricated 1-D
`coordinates1d`/`displacement1d` fixtures (never in upstream) were removed
(`coordinates`/`displacements` transforms are covered by unit tests instead).
These corrections should be pushed upstream too.
