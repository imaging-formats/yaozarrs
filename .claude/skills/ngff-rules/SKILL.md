---
name: ngff-rules
description: Audit, add, change, or retire yaozarrs rule codes (YZ-<AREA>-NNN for spec rules, YZ-NNN for yaozarrs-only checks). Use when a new NGFF spec version (release, patch, or dev tag) comes out, when the published spec changes (`ngff_rules.py site`), when adding or changing a validation check, or when asked to verify that existing rule codes, quotes, line numbers, examples, or coverage are correct.
---

# NGFF rule codes

Each normative statement in the NGFF spec has a permanent code, and a yaozarrs error
or warning for a spec violation should carry the code of the statement it enforces.
There are two kinds:

- `YZ-<AREA>-NNN` (e.g. `YZ-WL-002`): a spec rule, citing the spec's prose or JSON
  schemas.
- `YZ-NNN` (e.g. `YZ-007`): a check yaozarrs adds, with no source in the spec.

The codes belong to yaozarrs: they aren't part of the NGFF spec (see
<https://github.com/ome/ngff/issues/646>). If official codes are adopted later, a spec
rule records its official code; it doesn't get renamed.

The pieces are:

- `src/yaozarrs/_rules/rules.toml`: the registry and single source of truth. It holds
  the pinned spec files (`[spec."<version>"]`, whose keys are the tracked versions)
  and every rule (`[rules.<CODE>]`), with its quotes and line numbers.
- `src/yaozarrs/_rules/rule_examples.toml`: a minimal failing example for each rule
  that yaozarrs checks, keyed by code (`Rule.example`/`Rule.examples`, loaded on first
  use). The tests and the docs script use them.
- `src/yaozarrs/_rules/__init__.py`: `rule_error(code, msg)` (raise it inside a pydantic
  validator), `warn_rule(code, msg, stacklevel)`, `rewrap_error(e, prefix, suffix, code)`
  (add context to an error's message and keep its code), `error_code(e)`, and
  `RuleCodes` (field metadata that gives pydantic's own errors, such as a missing field
  or a wrong type, a rule code; see "Field-shape rules" below).
- `scripts/ngff_rules.py`: `fetch`, `check [--fix]`, `relocate`, `coverage`,
  `uncited`, and `docs`.
- `src/yaozarrs/_rules/skipped.toml`: spec lines with a keyword that deliberately have
  no rule, each with a `reason` (see "Every statement is catalogued" below).
- `tests/spec_sources/`: vendored spec files and cited JSON schemas at the pinned commits.
- `tests/test_rules.py`: integrity tests, and the coverage test. The coverage test runs
  every example against every version module and expects it to fire exactly in the
  versions whose spec states the rule, except for the `KNOWN_GAPS` and
  `KNOWN_OVERREACH` entries (which are also verified, so stale entries fail).
- `docs/rules/`: generated at docs-build time and gitignored. Never edit them.

A rule entry looks like this. A quote shared by several versions has one `lines`
entry per version. If the wording differs between versions, add another
`[[rules.<CODE>.sources]]` block.

```toml
[rules.YZ-AX-001]
slug = 'axis-name-unique'
level = 'MUST'
summary = 'Axis names must be unique within `axes`.'
introduced = '0.4'

[[rules.YZ-AX-001.sources]]
quote = 'The values MUST be unique across all "name" fields.'
lines = { "0.4" = [243, 243], "0.5" = [183, 183] }
```

and its example, in `rule_examples.toml`:

```toml
[YZ-AX-001.example]
model = 'Multiscale'   # class name in yaozarrs.v04 / yaozarrs.v05
json = '''
{"axes": [...], "datasets": [...]}
'''
```

Rules checked by the storage validator (`validate_zarr_store`) use a `store` example
instead of `model`/`json`. It is a JSON object that maps v0.5 metadata file paths to
their contents (`{"zarr.json": {...}, "s0/zarr.json": {...}}`); no chunk data is
needed. In code, storage checks pass the code to
`result.add_error(StorageErrorType.<type>, "<code>", loc, msg)`. Several storage
`type`s can share one code: `type` is the specific failure, `code` is the spec
statement.

`example` is written in the v0.5 format. `example_for` in `scripts/ngff_rules.py`
derives the other versions: `to_v04_store`, `to_v06_store`, and the `Multiscale` and
`Dataset` conversions to v0.6 (`axes` to a `coordinateSystems` entry named `physical`,
with `input` and `output` on each dataset's transformation). Where that doesn't work,
add `[<CODE>.examples."<version>"]`, in that version's own format. A rule that
only exists in one version has only that entry. Model examples are validated as JSON
(`model_validate_json`), so documents with a `version` field need one in the example.

A requirement that only a JSON schema in the spec repo states is still a spec rule.
Add `schema = '<file>.schema'` to its `sources` block, and quote the schema (the file is
found in that version's `schemas/` folder, and `fetch` vendors it). A base schema means
`level = 'MUST'`, and a `strict_*.schema` means `level = 'SHOULD'`. If a schema and the
prose disagree, keep yaozarrs' behavior (use `severity` if it only warns), and describe
the disagreement in `notes`.

Optional fields: `notes`, `severity` (default: `warning` for SHOULD, else `error`),
`status` (`stable` by default), `removed`, `supersedes`, `aliases`, and
`checkable = false`.

## Every statement is catalogued

Every RFC 2119 keyword (MUST, SHOULD, MAY, REQUIRED, RECOMMENDED, OPTIONAL, SHALL) in
a tracked spec version is inside a rule's quote, or inside the quote of an entry in
`skipped.toml`. This is checked per keyword, not per line: a quote that covers one
clause of a line doesn't account for a second keyword on the same line. A rule
doesn't need yaozarrs to check it: a rule without an example is listed as not
reported (yaozarrs may still reject such input, under another code or none).
`test_every_statement_catalogued` enforces this, and `scripts/ngff_rules.py uncited`
lists the keywords still missing.

- Statements about software behavior (what readers, writers, or viewers do) are rules
  with `checkable = false`: no document can violate them.
- MAY: a MAY that sets a limit ("MAY contain one additional entry of `type:time`") is a
  rule with level `MAY`. A MAY that offers a choice whose value is constrained ("MAY
  contain `longName`. The value MUST be a string") is covered by the MUST's rule. A MAY
  that only grants permission ("The interpolation attributes MAY be provided") is
  skipped as `permission`.
- Skip reasons (`[[skip]]` entries have `version`, `quote`, `lines`, `reason`, `note`):
  `permission`, `boilerplate` (RFC 2119 definitions and other text about the document
  itself), `duplicate` (restates a requirement that has a rule: name its code in
  `note`), `illustration` (the keyword is in an example, a diagram, a code comment, or
  a version history table). Skip quotes are checked and line-fixed like rule quotes.

## Field-shape rules

Statements about a field's presence or value type ("MUST contain the field X", "whose
value MUST be a string") are enforced by the pydantic model itself. To give those errors
a code, add one `RuleCodes` to the field's `Annotated` metadata, mapping pydantic error
types to codes:

```python
id: Annotated[
    NonNegativeInt,
    RuleCodes(missing="YZ-PL-009", int="YZ-PL-010", greater_than_equal="YZ-PL-010"),
] = Field(...)
```

- Keys are pydantic error types (`missing`, `string_pattern_mismatch`, `too_short`,
  `literal_error`, `listItemsNotUnique`, ...), or the shorthands `int`, `str`, `list`,
  `object`. Only map the errors the statement covers. A constraint from a JSON schema
  (e.g. `MinLen(1)` from `minItems`) gets its own rule citing that schema. If neither
  the prose nor a schema states it, leave that error untagged and say so in `notes`.
- Use exactly one `RuleCodes` per field, at the top level of the type
  (`Annotated[X | None, RuleCodes(...)]`, not `Annotated[X, RuleCodes(...)] | None`).
  pydantic < 2.6 keeps only one metadata object of each type.
- Only errors on the field itself or on its items are tagged; errors inside a nested
  model are tagged by that model. Don't tag fields of models used as members of a
  non-discriminated union (e.g. a transformation's `type` in v0.4 or v0.5): pydantic
  reports errors from every member, so a valid `translation` would be reported as
  breaking the `scale` member's rule. Discriminated unions (v0.6 transformations) are
  fine, and `union_tag_invalid` can be tagged.
- Presence and value type are separate statements, so separate codes
  (`well-path-present`, `well-path-string`). If a statement has no RFC 2119 keyword in a
  version (e.g. 0.4's "whose value is a string"), cite it with `implied = true` (see
  invariant 3).
  `src/yaozarrs/_common/` (including `_axis.py`) is shared by several versions, so
  a tag there applies to all of them (record any overreach in `KNOWN_OVERREACH`).
- Don't tag `version` fields: their `Literal` selects the model for a version.

## Invariants (never break these)

1. **Codes are permanent.** Never renumber, reuse, or delete a code. A new code takes
   the next unused number in its area (look at the highest existing number in
   `rules.toml`, including withdrawn rules).
2. **One rule = one (subject, constraint, RFC 2119 level).** The level is the
   statement's strength: `MUST`, `SHOULD`, or `MAY`. "MUST NOT" is level `MUST` and
   "SHOULD NOT" is level `SHOULD`, so rewording "MUST NOT be a duplicate" as "MUST be
   unique" is only a change of wording. A sentence with two keywords is two rules. If
   a later version changes any of the three, the old rule gets `removed="<version>"`
   and a new code is minted with `supersedes=("<old>",)`. If only the wording changes,
   keep the code and add a source for the new version.
3. **Quotes are verbatim** (whitespace-insensitive) and must contain the rule's
   `level` keyword. Never type line numbers by hand. Write `[0, 0]` and run
   `uv run python scripts/ngff_rules.py check --fix`, which fills them in when a quote
   occurs exactly once and leaves the rest of the file untouched. If a quote is
   ambiguous, extend it until it is unique.

   Exception: some versions state a requirement without a keyword ("contains the key
   `labels`", "Required: yes" in a field table, a lowercase "must"). If the spec text
   implies the requirement, it is still a spec rule: cite the quote with
   `implied = true` on its `sources` block, and explain it in `notes`. If no text in
   that version implies it, don't cite the version, and list any check yaozarrs does
   there in `KNOWN_OVERREACH`.
4. **Spec rules only cite the spec's prose or JSON schemas.** Checks with no source
   there go in `YZ-NNN`
   (no `level`, `introduced`, or `sources`, and `severity` must be set).
5. **Coverage is computed, never declared.** Every code that yaozarrs emits has an
   `example`, and `scripts/ngff_rules.py coverage` shows where each one fires. If the
   coverage test fails, find out why before touching `KNOWN_GAPS` or
   `KNOWN_OVERREACH`. A new entry in either set is a behavior decision, so ask the
   user.
6. **Drafts:** a rule first seen in a dev version (e.g. `0.9.dev2`, PEP 440 form;
   normalize upstream tags like `0.6dev1` to `0.6.dev1`) gets a permanent code with
   `status="draft"`. A draft may be edited in place. When the version is released,
   switch it to `status="stable"`, point its sources at the release `SpecDoc`, and set
   `introduced` to the release version. If it is dropped before release, use
   `status="withdrawn"` and `removed="<dev version where it disappeared>"`.

Area prefixes: `AX` axes, `CS` coordinateSystems, `MS` multiscales, `DS` datasets, `CT`
coordinateTransformations, `SC` scene, `LB` labels, `IL` image-label, `PL` plate, `WL`
well, `BF` bioformats2raw, `OM` omero, `ZR` zarr/storage layout, `GN` general. Ask the
user before adding a new area.

## Workflow: audit existing rules

1. `uv run python scripts/ngff_rules.py check`: every quote must be at its lines.
2. For each rule, read the quote **in context**: open the vendored spec file in
   `tests/spec_sources/` about 10 lines around `lines`. Confirm that:
   - the `level` matches the strength of the keyword that governs the constraint;
   - the `summary` says what the spec says, no more and no less;
   - the yaozarrs check (grep `src/` for the code) enforces that statement. Record any
     deliberate divergence (relaxed, stricter, partial) in `notes`.
3. Run `scripts/ngff_rules.py coverage`, and review `KNOWN_GAPS` and
   `KNOWN_OVERREACH` in `tests/test_rules.py`. Check that each example is minimal: it
   should violate only the rule it illustrates, apart from errors that can't be
   avoided. Report gaps and overreach rather than fixing them silently.
4. Report what you found. Don't change validation behavior without asking.

For a review by a person, `scripts/ngff_rules.py review <version> -o <file>` writes a
Markdown checklist of every rule source and skip entry for that version, in the order
the spec states them, with the cited lines and the quote in bold.

yaozarrs is deliberately lenient when input is not strictly correct but can be
interpreted unambiguously: lax type coercion (`"1"` or `1.0` for an integer), defaults
(a transformation without `type`), or `str()` on a path. That isn't a gap, so don't
propose errors or warnings for it. The statements still get rules (every statement is
catalogued): say in `notes` that yaozarrs accepts the interpretable input.

## Workflow: the published spec changed

Each pin in `rules.toml` (`[spec.*]`) is the commit that ngff.openmicroscopy.org
publishes, not a release tag. The site is built (Read the Docs) from `ome/ngff` main,
which includes each version as a submodule `specifications/<v>` of `ome/ngff-spec`,
rendered from its `index.md`. Fixes land on those branches without new tags, so the
pins need checking. `.github/workflows/spec-pins.yml` runs `site` weekly and opens (or
comments on) an issue when it fails. Also check before every release:

1. `uv run python scripts/ngff_rules.py site` compares each pin with the submodule
   commit on `ome/ngff` main, and exits 1 if any moved.
2. For a moved pin, update that version's `sha` (and `ref`, if the branch changed),
   run `fetch` (`git rm` the old vendored file if its name changed), then `check --fix`.
   Read the upstream diff (`<old sha>...<new sha>`), and investigate every quote that
   is no longer found: a reworded statement keeps its code, a changed meaning gets a
   new one (see step 3 below).
3. `uncited <version>`, and catalogue every keyword it lists.

## Workflow: a new spec version is published

1. Find the commit the site publishes: run `site` after adding the version (it reads
   the `specifications/<v>` submodule on `ome/ngff` main), or before the site has it,
   use the release tag (`git ls-remote --tags https://github.com/ome/ngff-spec`, the
   `^{}` peeled sha).
2. A patch release of a tracked version follows "the published spec changed" above.
3. New minor or dev version:
   1. Add a `[spec."<version>"]` table to `rules.toml`, then run `fetch`.
      If yaozarrs gains a module for the version, add it to `MODULES` in
      `scripts/ngff_rules.py`.
   2. `uv run python scripts/ngff_rules.py relocate <repo> <sha> <path>` lists which
      current quotes still appear verbatim. A format change (Bikeshed → MyST,
      changed quoting or backticks) makes many quotes come up NOT FOUND even when the
      statement is unchanged, so search for each missing statement by hand.
   3. For each existing rule, decide one of the following. Read the surrounding text,
      because a sentence can stay identical while its meaning shifts.
      - **unchanged**: add `"<version>" = [0, 0]` to the `lines` of the matching
        quote, or add a new `sources` block if the wording differs;
      - **removed**: set `removed="<new version>"`;
      - **changed meaning**: set `removed`, and mint a new code with `supersedes`.
      Ask the user when it's unclear whether the meaning changed.
   4. Run `scripts/ngff_rules.py uncited <version>`, and catalogue every keyword it
      lists (see "Every statement is catalogued"): a new rule, a new source on an
      existing rule, or a `skipped.toml` entry. Rules that yaozarrs doesn't check get
      no `example`.
   5. `check --fix`, then the closing steps below.

## Workflow: add or change a check in code

1. Find the governing statement in the spec. If none exists, use a `YZ-NNN` code.
2. Add the rule to `rules.toml` (next free number in the area; `[0, 0]` lines), then
   run `check --fix`.
3. In the validator, `raise rule_error("<code>", msg)` or
   `warn_rule("<code>", msg, stacklevel=...)`. Keep existing message text, because
   tests and users match on it. For an error that pydantic raises itself, add the code
   to the field's `RuleCodes` instead (see "Field-shape rules").
4. Add an `example` to `rule_examples.toml`: the model class name plus a minimal failing JSON document, or a
   `store` for storage checks. Then run `scripts/ngff_rules.py coverage` to confirm where
   it fires. An example should trigger only its own rule, apart from errors that can't
   be avoided; check what it emits with `emitted_codes()`.

## Closing steps (always)

```sh
uv run python scripts/ngff_rules.py check
uv run pytest tests/test_rules.py
just docs                # optional: render docs/rules/ and build the site
uv run pytest            # message changes can affect other tests
uv run ruff format && uv run ruff check --fix
```
