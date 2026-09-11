# v0.6 review: action items

Status legend: ✅ fixed · 🟡 partially fixed · ⬜ open

## A. Bugs (current behavior is wrong)

1. ✅ **`v06/_storage.py:287` — label dataset-count check is one-sided.** Spec says label datasets MUST have the *same* number of entries as the parent image (index.md:1494); the code used `n_lbl_ds < n_img_ds`, so a label with *more* levels passed. **Fixed: now `!=`.**

2. ✅ **`Scene` documents crash storage validation.** `Scene` is in the `OMEMetadata` union but the storage dispatch had no branch for it and raised `NotImplementedError`. **Fixed: dispatch now routes to a new `visit_scene` (see C3).**

3. ✅ **FOV path validation is stale for v0.6.** v0.6 *changed* the well-image path rule (index.md:1678–1681, `well.schema` pattern `^[A-Za-z0-9_.-]+$` plus `not ^\.+$` / `not ^__`). **Fixed: `v06/_plate.py` now has its own `FOVPathName` implementing the v0.6 rule (allows `._-` without warning, rejects all-periods paths and the `__` prefix); tests updated. The stale `RelaxedFOVPathName`/env-var machinery remains for v04/v05 where it is correct.**

4. ✅ **`ByDimensionItem.input_axes/output_axes: list[float]`** accepted `0.5` as an axis index; prose says integers (index.md:1171). **Fixed: now `list[int]` with `ge=0` (pydantic still coerces `1.0`, so schema-"number" docs keep parsing).**

5. ✅ **`_version.py` validator was looser than its own docstring** (`"0.6.x"`, `"0.6.dev4.junk"`, `"0.6.0.1"` all passed). **Fixed: now an explicit regex `^0\.6(\.0|\.dev\d+)?$`.**

6. ✅ **Test fixtures are not what they claim.** `tests/v06/test_examples_v06.py` said fixtures were "taken verbatim" from the spec examples, but `"type": "space"` was added to axes (the real examples are type-less and fail the `axes.schema` `oneOf`). **Fixed: docstring now says "adapted" and explains the axis-type modification.**

7. ✅ **Docstrings referenced `TRICKY_NOTES_v06.md`** (removed from git in a42a787). **Fixed: references removed from `_transforms.py` and `_scene.py`; surviving pragmatic-validation notes reworded in place.**

## B. Missing model-level validators

**Transforms (`v06/_transforms.py`)**:
- 🟡 **byDimension**: **Fixed locally:** duplicate output axes across children now rejected (`byDimensionInvalid2`-style docs), and `input_axes`/`output_axes` length is checked against the child's parameter array for scale/translation children (index.md:1177–1178). **Still open:** document-level range/coverage check of axis indices against the input/output CS dims (`byDimensionInvalid1`-style docs) — belongs in `Multiscale._post_validate`/`SceneDef` where the CSs are in scope.
- ✅ **mapAxis**: duplicate indices now rejected (schema `uniqueItems: true`, prose index.md:750). (Note: the spec's own `mapAxis2.json` contradicts this; see D.)
- 🟡 **affine**: **Fixed locally:** inline matrix must now be non-empty and rectangular. **Still open:** document-level rows == M, cols == N+1 against the input/output CS dims (index.md:884–885).
- 🟡 **rotation**: **Fixed:** inline matrix must now be square N×N with N∈2..5 (hard schema constraint). Determinant-1/orthonormality (prose MUST, index.md:963) not checked numerically.
- ✅ **sequence**: `transformations: []` now rejected (`min_length=1`; prose index.md:1003–1004).
- ✅ **Transformation `name` uniqueness** within a list (index.md:416–417 MUST): new `_validate_unique_transform_names` applied to `Multiscale.coordinateTransformations` and `SceneTransformList`.
- ⬜ **Dimensionality vs. coordinate systems** for *non-dataset* transforms: a length-3 `scale` between two declared 2-D systems under `multiscales > coordinateTransformations` still passes (index.md:418, 806–807, 833–834). Same for mapAxis length==N==M (index.md:747–749), bijection forward/inverse dims (index.md:1228–1230). `Multiscale._post_validate` has the CSs in scope — that's the natural home. (Path-backed affine/rotation matrices and vector fields DO now get dimensional checks, at the storage layer — see C1/C2 — but inline parameters are still unchecked against CS dims.)
- ✅ **Transform-graph connectedness** (index.md:517–522, not in the original review): the coordinate systems + transformations MUST form a fully connected graph. **Fixed in two layers: `Multiscale._post_validate` checks it document-locally (declared CSs + dataset arrays as nodes; dataset transforms and multiscale CTs as edges), and `visit_scene` checks it at the storage layer for scenes (where the referenced CSs live in child image groups).**

**Image (`v06/_image.py`)**:
- ✅ **`input.path` never compared to `Dataset.path`** (index.md:1319): new `Dataset._input_path_matches` model validator enforces equality.
- ✅ **Translation length unchecked in dataset sequences**: `_validate_dataset_transform` now requires the scale and translation arrays to have equal length (index.md:806–807).
- ✅ **`multiscales > coordinateTransformations` output.path** must be a relative downward path (image.schema forbids `^(\.\./|/)`): now rejected in `_post_validate`.
- ⬜ **Only the intrinsic CS is checked** against dataset dimensionality; index.md:1284 applies the axes-length rule to *all* coordinate systems in the multiscale.
- ⬜ **`input.path` on multiscale-level transforms**: yaozarrs warns (following prose SHOULD, index.md:1356), but image.schema's `additionalProperties: false` makes it schema-invalid. The spec conflicts with itself here (its own strict example `multiscales_reference_to_label.json` includes `input: {path, name}`) — the warning is defensible; leave a comment recording the divergence.

**Plate/Well (`v06/_plate.py`)**:
- ✅ **`path` ↔ `rowIndex`/`columnIndex` consistency** (index.md:1639–1649): `_validate_well_indices` now checks `path == rows[rowIndex].name + "/" + columns[columnIndex].name`.
- ✅ **Acquisition `id` uniqueness** within the plate (index.md:1594): new `_validate_unique_acquisition_ids`.
- ✅ **FOV `path` uniqueness** (index.md:1677): new `WellDef._validate_unique_paths` (path-level, stronger than whole-object `UniqueList`).

**Labels (`v06/_labels.py`)**:
- ✅ **`label-value` uniqueness** in `colors` and `properties` (index.md:1504, 1514): new `_validate_unique_label_values` on both fields.
- ✅ **`LabelProperty` silently dropped arbitrary per-label metadata** (index.md:1517–1519; `LabelColor` extras allowed per index.md:1510): both models now use `extra="allow"` so extras round-trip.
- ✅ Awareness item recorded in code: `LabelColor.label_value: float` follows the schema ("number") while prose says integer — now documented with a comment; not tightened (would reject schema-valid docs).

**Axes (`v06/_axes.py`)** — possible *over*-validation:
- ⬜ The time/channel/ordering rules (index.md:1286–1289) are scoped by the prose to "coordinate systems **inside multiscales metadata**", but `_validate_axes_list` applies them to every `CoordinateSystem`, including scene-level ones. A scene CS ordered `[x, y, t]` violates nothing yet is rejected. Consider keeping only the schema `oneOf` in the generic `AxesList` and moving ordering/count into the multiscale context.

## C. `v06/_storage.py` — structural gaps

The file started as a verbatim v0.5 clone; the RFC-5 storage checks below have now been added:

1. ✅ **Path-backed transform parameters never resolved** (index.md:878–880, 967–969). **Fixed: new `_validate_transform_params` recursively walks `multiscales > coordinateTransformations` (and scene transforms), including nested `sequence`/`bijection`/`byDimension` children. Path-backed affine/rotation matrices must resolve to 2-D zarr arrays; shape is checked against the input/output coordinate-system dims where resolvable — affine M×(N+1), rotation N×N (byDimension children get dims from `input_axes`/`output_axes`; sequence interiors have unknown dims and get existence/2-D checks only). Non-downward paths (`/...`, `../...`) produce a warning instead of being silently skipped.**
2. ✅ **`displacements`/`coordinates` fields** (index.md:1094–1102). **Fixed: new `_check_vector_field` resolves `path` to a multiscale image group and checks: N+1 dimensionality, exactly one axis of type `displacement`/`coordinate`, length along that axis == N (displacements, with the M=N requirement) or M (coordinates), measured on the highest-resolution dataset.**
3. ✅ **`visit_scene`** (index.md:1721–1727, 517–522). **Fixed: new `visit_scene` resolves every transform endpoint — no `path` → coordinate system must be declared in the scene; with `path` → must resolve to a multiscale image subgroup declaring the named coordinate system — validates path-backed parameters (via 1 and 2), and checks the transform-**graph connectedness**: scene coordinate systems + endpoints form nodes, transforms form (direction-agnostic) edges, and everything must be one connected component (`transform_graph_disconnected`).**
4. ✅ **Same dtype across all datasets of a multiscale** (index.md:1309; explicitly added in the 0.6.dev4 changelog). **Fixed: `_visit_multiscale_no_prefetch` now compares each array's dtype against the first dataset's (new `StorageErrorType.dataset_dtype_mismatch`).**
5. ✅ **`multiscales > coordinateTransformations` output linking to a child labels group** (index.md:1359). **Fixed: `_visit_multiscale_transforms` resolves `output.path` on disk, requires Image metadata there, and confirms a coordinate system named `output.name` is declared.**
6. ✅ **Intermediate groups between `labels` and label images MUST NOT contain metadata** (index.md:1485–1486). **Fixed: `visit_labels_group` now checks every intermediate group of nested label paths for an `ome` attributes key (`labels_intermediate_metadata`).**
7. ⬜ **Version consistency within the hierarchy** (index.md:189): children are force-parsed with `version="0.6.dev4"`, which masks a 0.5 child inside a 0.6 tree instead of reporting it.
8. ✅ **Well `acquisition` ids vs plate acquisitions** (index.md:1682–1684). **Fixed: new `_validate_well_acquisitions` in `visit_plate` checks each field-of-view's `acquisition` against the plate's acquisition ids, and requires the key when the plate defines multiple acquisitions (`well_acquisition_invalid`).**
9. ⬜ **bf2raw numbered-group discovery `break`s at the first gap**, so `0,1,3` silently validates only `0,1` — a gap is itself a MUST violation (index.md:388). (Needs child-listing support in `ZarrGroup` to do cheaply.)
10. ✅ **`dimension_names` mismatch was a hard error with no v0.6 anchor** (the requirement existed in 0.5.2 but `dimension_names` doesn't appear anywhere in the 0.6 index.md — possibly a spec regression worth raising upstream). **Fixed: demoted to a warning in v06 (v05 keeps the error).**
11. ✅ **Label *multiscales-object* count equality was stricter than the spec**, which only ties the per-multiscale *dataset* counts (index.md:1494). **Fixed: the multiscales-count error was removed; per-index dataset counts are still compared.**
12. ⬜ Minor: nothing asserts `zarr_format == 3` for arrays (index.md:58–59) — though in practice a v2 array under a v3 group is unresolvable and surfaces as `dataset_path_not_found`.

## D. Spec-repo inconsistencies you're implicitly deciding around

Since these force silent judgment calls in yaozarrs, worth filing upstream (or fixing, given it's your checkout of `ngff-spec` main):

- `mapAxis2.json` violates its own schema three ways: length-1 `mapAxis` (schema `minItems: 2`), duplicate indices `[1,1,0]` (schema `uniqueItems`, prose "exactly once"), and `"input": "in"` as a bare string. The prose "length MUST equal dims of *both* input and output" also contradicts the projection examples.
- `byDimensionXarray.json`, `xarrayLike.json`, `sequenceSubspace1.json` use *string* axis names in `input_axes`/`output_axes` (and put them on `sequence` children); prose says integers, schema says number. Your models can't parse any of these three.
- Most transformation examples use type-less axes, which fail `axes.schema`'s `oneOf` (2–3 space XOR ≥2 array). Your test fixtures paper over this by adding `type: "space"` (now documented in the test module docstring).
- `examples/transformations/displacements/displacement_field.json` has a length-2 dataset scale against a 3-axis CS (rejected by your models, correctly).
- `multiscales_example_relative.json` uses axes-as-dict and `version` inside the multiscale — pre-current-schema format.
- `interpolation`: prose `bspline-cubic` vs schema enum `cubic`. Prose table offers scale/translation-by-`path`; schema doesn't. Labels-link transform types: index.md:1360 (identity/scale/translation) vs index.md:1479–1480 (also sequence-of-scale+translation) — you follow 1360.
- Schema nits: `axes.schema` has stray top-level `minContains`/`maxContains` (no sibling `contains`, no-ops); `ome.schema` `minContains: 1` likewise a no-op (yaozarrs' `MinLen(1)` is technically stricter); image.schema typo "a ingle scale".

## E. Test-coverage gaps

⬜ Missing from `tests/data/v06/examples`: the negative cases `byDimensionInvalid1/2.json` (spec ships them as must-fail; Invalid2 is now rejected by the new duplicate-output-axes check, Invalid1 still passes), `displacements/*`, `subspacePermute.json` (should parse today — good positive case), `plate_6wells.json`, `well_4fields.json`, `bf2raw/*`, `affine2d3d.json`, `affine2d2d_with_channel.json`, `byDimension2.json`, `bijection_verbose.json`. The unparseable ones (`mapAxis2`, `xarrayLike`, `byDimensionXarray`, `sequenceSubspace1`, `multiscales_example_relative`) deserve an explicit skip-with-reason list rather than silent omission.

⬜ The model-level validators added earlier (mapAxis uniqueness, affine/rotation shape, byDimension local checks, sequence non-empty, transform-name/label-value/acquisition-id/FOV-path uniqueness, well path↔index consistency, dataset input.path match) are covered indirectly by the existing suite but have no dedicated tests yet. The v0.6 storage checks (scene endpoints + graph, path-backed matrices, displacement fields, labels links, intermediate label metadata, acquisition cross-checks, dtype consistency, dimension_names demotion) DO have dedicated tests in `tests/v06/test_storage_v06.py`, and the multiscale graph-connectivity check is tested in `tests/v06/test_images_v06.py`.

---

**Remaining priorities:** document-level dimensionality checks for inline transform parameters in `Multiscale._post_validate` (closes the byDimension/affine/mapAxis "still open" halves in B), version-consistency reporting (C7), and the axes-ordering scoping question in B.
