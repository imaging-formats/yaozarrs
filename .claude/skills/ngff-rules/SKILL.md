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
- `YZ-NNN` (e.g. `YZ-006`): a check yaozarrs adds, with no source in the spec.

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
  (add context to an error's message and keep its code), and `error_code(e)`.
- `scripts/ngff_rules.py`: `fetch`, `check [--fix]`, `relocate`, `coverage`, `site`,
  `uncited`, `docs`, and `review`.
- `src/yaozarrs/_rules/skipped.toml`: spec lines with a keyword that deliberately have
  no rule, each with a `reason` (see "Every statement is catalogued" below).
- `tests/spec_sources/`: vendored spec files and cited JSON schemas at the pinned
  commits (each file name includes its commit).
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
model = 'Multiscale'   # class name in yaozarrs.v04 / v05 / v06
json = '''
{"axes": [...], "datasets": [...]}
'''
```

Rules checked by the storage validator (`validate_zarr_store`) use a `store` example
instead of `model`/`json`. It is a JSON object that maps v0.5 metadata file paths to
their contents (`{"zarr.json": {...}, "s0/zarr.json": {...}}`); no chunk data is
needed.

`example` is written in the v0.5 format. `example_for` in `scripts/ngff_rules.py`
derives the other versions: `to_v04_store`, `to_v06_store`, and the `Multiscale` and
`Dataset` conversions to v0.6 (`axes` to a `coordinateSystems` entry named `physical`,
with `input` and `output` on each dataset's transformation). Where that doesn't work,
add `[<CODE>.examples."<version>"]`, in that version's own format. A rule that
only exists in one version has only that entry. Model examples are validated as JSON
(`model_validate_json`), so documents with a `version` field need one in the example.

A rule cites one of the spec's JSON schemas in exactly two cases:

- **the prose is silent**, and only the schema states the constraint (it is still a
  spec rule);
- **the prose and the schema both constrain it, but disagree** (e.g. 0.4's prose says
  a `colors` entry's `label-value` MUST be an integer, `label.schema` says `number`): add
  the schema source next to the prose one, as a record of the conflict, and say in
  `notes` which one yaozarrs follows (the prose, unless it's silent).

Don't cite a schema that says the same as the prose, or for prose the schema doesn't
encode. A schema source points at the keyword, rather than quoting text:

```toml
[[rules.YZ-PL-040.sources]]
schema = 'plate.schema'      # in that version's `schemas/` folder (`fetch` vendors it)
pointer = '/properties/ome/properties/plate/properties/wells/uniqueItems'
value = true                 # optional: the keyword's value, checked by the tests
lines = { "0.5" = [0, 0], "0.6" = [0, 0] }
```

One block per pointer: if a later version moves the keyword (0.4's plate schema has no
`ome` object), it gets its own block. `check --fix` fills in the keyword's lines, and
`check` verifies the pointer and the value. A base schema means `level = 'MUST'`, and a
`strict_*.schema` means `level = 'SHOULD'`.

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
   keep the code and add a source for the new version. `supersedes` is only for a new
   rule about the same subject: if a requirement's job moves to a different or broader
   statement (0.4's `multiscales` `version` became the `ome` object's `version`), the
   old rule just gets `removed`, and the notes of both say where it went.
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
   run `fetch` (it downloads the new files and deletes the old ones; stage both with
   `git add -A tests/spec_sources`), then `check --fix`.
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
   tests and users match on it.
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
