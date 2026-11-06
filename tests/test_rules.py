"""Integrity checks for the rule-code registry (`yaozarrs._rules`)."""

from __future__ import annotations

import copy
import importlib.util
import json
import pickle
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import get_args

import pytest
from packaging.version import Version
from pydantic_core import PydanticCustomError

from yaozarrs import validate_zarr_store
from yaozarrs._rules import (
    RULES,
    Rule,
    RuleWarning,
    SpecSource,
    _model,
    error_code,
    get_rule,
    rewrap_error,
    rule_error,
    warn_rule,
)
from yaozarrs._rules._model import Example, Level, _load_examples
from yaozarrs._rules._registry import SPEC_DOCS

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "ngff_rules", ROOT / "scripts" / "ngff_rules.py"
)
assert _spec and _spec.loader
ngff_rules = importlib.util.module_from_spec(_spec)
sys.modules["ngff_rules"] = ngff_rules  # (for its dataclasses)
_spec.loader.exec_module(ngff_rules)

# RFC 2119 keywords that have the same strength as each level
TRACKED_VERSIONS = tuple(SPEC_DOCS)
SYNONYMS = {
    "MUST": "MUST|REQUIRED|SHALL",
    "SHOULD": "SHOULD|RECOMMENDED",
    "MAY": "MAY|OPTIONAL",
}
CODE_RE = re.compile(r"^YZ-([A-Z]{2}-)?\d{3}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# Where yaozarrs behavior knowingly differs from "checked in every version whose
# spec states the rule". Both sets are verified, so stale entries fail the test.
# Spec states the rule, but yaozarrs doesn't check it (yet).
KNOWN_GAPS: set[tuple[str, str]] = set()
# yaozarrs checks the rule in a version whose spec doesn't state it.
KNOWN_OVERREACH: set[tuple[str, str]] = set()


# ---------------------------------------------------------------------------
# registry integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", RULES.values(), ids=list(RULES))
def test_rule_fields(rule: Rule) -> None:
    assert CODE_RE.match(rule.code)
    assert SLUG_RE.match(rule.slug)
    if rule.status == "withdrawn":
        assert rule.removed, "withdrawn rules must set `removed`"
    for code in rule.supersedes:
        assert code in RULES and code != rule.code

    if not rule.is_spec_rule:
        assert rule.level is None and rule.introduced is None and not rule.sources
        assert rule.severity is not None
        return

    assert rule.level in get_args(Level)
    # sources for exactly the tracked versions from `introduced` up to `removed`
    assert rule.introduced in TRACKED_VERSIONS
    if rule.removed:
        assert Version(rule.introduced) < Version(rule.removed)
    stated = [
        v
        for v in TRACKED_VERSIONS
        if Version(rule.introduced) <= Version(v)
        and not (rule.removed and Version(v) >= Version(rule.removed))
    ]
    assert {s.doc.version for s in rule.sources} == set(stated)
    for src in rule.sources:
        assert src.doc.version in TRACKED_VERSIONS
        if src.is_schema:
            # schemas have no keywords: base schemas are MUST, strict schemas SHOULD
            strict = src.doc.path.rsplit("/", 1)[-1].startswith("strict_")
            assert rule.level == ("SHOULD" if strict else "MUST")
        elif src.implied:
            assert rule.notes, "explain in `notes` why a keyword-less quote counts"
        else:
            assert rule.level
            assert re.search(rf"\b({SYNONYMS[rule.level]})\b", src.quote), (
                f"quote for {rule.code} does not contain {rule.level!r}"
            )


def test_unique_slugs() -> None:
    slugs = [r.slug for r in RULES.values()]
    assert len(slugs) == len(set(slugs))


def test_tracked_versions_ordered() -> None:
    versions = [Version(v) for v in TRACKED_VERSIONS]
    assert versions == sorted(versions)


@pytest.mark.parametrize(
    "src",
    [
        pytest.param(src, id=f"{r.code}-{src.doc.version}")
        for r in RULES.values()
        for src in r.sources
    ],
)
def test_quote_at_lines(src: SpecSource) -> None:
    assert ngff_rules.check_quote(src.doc, src.quote, src.lines) is None


def test_quote_must_be_unique() -> None:
    problem = ngff_rules.check_quote(SPEC_DOCS["0.5"], "MUST", (1, 1))
    assert problem and "times" in problem


def _src_codes() -> set[str]:
    """Rule codes used in `src/`."""
    pattern = re.compile(r"\"(YZ-(?:[A-Z]{2}-)?\d{3})\"")
    used: set[str] = set()
    for path in (ROOT / "src" / "yaozarrs").rglob("*.py"):
        used.update(pattern.findall(path.read_text(encoding="utf-8")))
    return used


def test_codes_in_source_exist() -> None:
    """Every code used in `src/` is registered and has an example."""
    used = _src_codes()
    assert used - set(RULES) == set()
    assert {c for c in used if not (RULES[c].example or RULES[c].examples)} == set()


@pytest.mark.parametrize("version", TRACKED_VERSIONS)
def test_every_statement_catalogued(version: str) -> None:
    """Every spec line with a keyword is cited by a rule, or skipped with a reason."""
    found = ngff_rules.uncited(version)
    assert not found, "\n".join(f"L{n}: {text}" for n, _, text in found)


def test_uncited_checks_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    # a MUST or SHOULD counts only if a rule of that level (or a skip) quotes it
    real = ngff_rules._citations
    as_may = [(q, lines, "MAY" if lvl else None) for q, lines, lvl in real("0.5")]
    monkeypatch.setattr(ngff_rules, "_citations", lambda version: as_may)
    found = ngff_rules.uncited("0.5")
    assert found and all("(only quoted by MAY rules)" in text for *_, text in found)


@pytest.mark.parametrize(
    "skip",
    ngff_rules.load_skips(),
    ids=lambda s: f"{s.version}-L{s.lines[0]}",
)
def test_skip_entries(skip: object) -> None:
    assert isinstance(skip, ngff_rules.Skip)
    assert skip.reason in ngff_rules.SKIP_REASONS
    doc = ngff_rules.SPEC_DOCS[skip.version]
    assert ngff_rules.check_quote(doc, skip.quote, skip.lines) is None
    assert ngff_rules.KEYWORD.search(skip.quote)
    if skip.reason == "duplicate":
        cited = re.findall(r"YZ-(?:[A-Z]{2}-)?\d{3}", skip.note)
        assert cited and all(c in RULES for c in cited), skip.note
        # (the rules it restates are stated in this version too)
        for code in cited:
            assert skip.version in {s.doc.version for s in RULES[code].sources}, code


@pytest.mark.parametrize("version", TRACKED_VERSIONS)
def test_review(version: str) -> None:
    """The review checklist has an item for every source and skip entry."""
    text = ngff_rules.review(version)
    sources = [s for r in RULES.values() for s in r.sources if s.doc.version == version]
    skips = [s for s in ngff_rules.load_skips() if s.version == version]
    assert text.count("\n- [ ] ") == len(sources) + len(skips)


def test_examples_have_rules() -> None:
    """Every example in `rule_examples.toml` is for a code that yaozarrs emits."""
    assert set(_load_examples()) - _src_codes() == set()


# ---------------------------------------------------------------------------
# coverage: each example fires exactly where it should
# ---------------------------------------------------------------------------


def _expected(rule: Rule, version: str) -> bool:
    if (rule.code, version) in KNOWN_OVERREACH:
        return True
    if (rule.code, version) in KNOWN_GAPS:
        return False
    if not rule.is_spec_rule:
        return ngff_rules.example_for(rule, version) is not None
    return version in {s.doc.version for s in rule.sources}


@pytest.mark.parametrize(
    "rule, version",
    [
        pytest.param(r, v, id=f"{r.code}-{v}")
        for r in RULES.values()
        if r.example or r.examples
        for v in ngff_rules.MODULES
    ],
)
def test_example_coverage(rule: Rule, version: str) -> None:
    example = ngff_rules.example_for(rule, version)
    if example and example.store is not None:
        pytest.importorskip("fsspec")
    emitted = rule.code in ngff_rules.emitted_codes(rule, version)
    assert emitted == _expected(rule, version)


def test_known_exceptions_have_examples() -> None:
    for code, version in KNOWN_GAPS | KNOWN_OVERREACH:
        assert version in ngff_rules.MODULES
        assert ngff_rules.example_for(RULES[code], version)


# ---------------------------------------------------------------------------
# messages and docs
# ---------------------------------------------------------------------------


def test_prose_permalink() -> None:
    # GitHub renders Markdown, without line numbers, unless asked for the plain view
    src = next(s for s in get_rule("YZ-GN-002").sources if s.doc.version == "0.5")
    assert src.permalink.endswith("index.md?plain=1#L43-L44")


def test_schema_source() -> None:
    src = next(s for s in get_rule("YZ-CT-045").sources)
    assert src.is_schema
    assert src.doc.path == "schemas/coordinate_transformations.schema"
    assert f"/blob/{src.doc.sha}/schemas/coordinate" in src.permalink
    assert "?plain=1" not in src.permalink
    v4 = next(s for s in get_rule("YZ-PL-037").sources if s.doc.version == "0.4")
    assert v4.doc.path == "schemas/plate.schema"
    assert v4.doc.cache_name == f"ome-ngff-spec-0.4-{v4.doc.sha[:9]}-plate.schema"


def test_rule_warning_filterable() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=r"\[YZ-AX-006\]")
        warn_rule("YZ-AX-006", "ignored")
        warn_rule("YZ-AX-007", "kept")
    (kept,) = w
    assert isinstance(kept.message, RuleWarning)
    assert kept.message.code == "YZ-AX-007"
    assert kept.message.url == get_rule("YZ-AX-007").url


def test_rule_warning_pickles() -> None:
    warning = RuleWarning("YZ-AX-007", "msg")
    for clone in (pickle.loads(pickle.dumps(warning)), copy.copy(warning)):
        assert isinstance(clone, RuleWarning)
        assert (clone.code, clone.msg, str(clone)) == ("YZ-AX-007", "msg", str(warning))


def test_rule_error() -> None:
    err = rule_error("YZ-CT-004", "bad {braces}")
    assert error_code(err) == "YZ-CT-004"
    assert str(err) == f"bad {{braces}}\n  See {get_rule('YZ-CT-004').url}"
    assert error_code(ValueError("bad")) is None


def test_rewrap_error_keeps_code() -> None:
    # the code survives the context that callers add to a message
    original = rule_error("YZ-CT-004", "bad")
    assert error_code(rewrap_error(original, "a: ", ": b")) == "YZ-CT-004"
    assert str(rewrap_error(original, "a: ", ": b")).startswith("a: bad: b\n  See ")
    assert error_code(rewrap_error(original, code="YZ-CT-041")) == "YZ-CT-041"
    plain = rewrap_error(ValueError("bad"), "a: ")
    assert str(plain) == "a: bad" and error_code(plain) is None
    # a custom error without a rule code keeps its type and context
    custom = PydanticCustomError("listItemsNotUnique", "{n} duplicates", {"n": 2})
    rewrapped = rewrap_error(custom, "a: ")
    assert isinstance(rewrapped, PydanticCustomError)
    assert rewrapped.type == "listItemsNotUnique"
    assert rewrapped.context == {"n": 2}
    assert str(rewrapped) == "a: 2 duplicates"


def test_docs_render() -> None:
    pytest.importorskip("fsspec")  # coverage of store examples
    docs = ngff_rules.expected_docs()
    assert set(docs) == {"index.md"} | {f"{c}.md" for c in RULES}
    for name, page in docs.items():
        # links to anchors in the spec would be broken here; other pages must exist
        assert "](#" not in page, name
        for target in re.findall(r"\]\(([^):]+\.md)\)", page):
            assert target in docs, (name, target)
    removed = next(r for r in RULES.values() if r.removed)
    assert f"(removed in {removed.removed})" in docs[f"{removed.code}.md"]


def test_heading_ignores_code_blocks() -> None:
    lines = ["# Section", "", "```python", "# not a heading", "```", "text"]
    assert ngff_rules._heading(lines, 6) == "Section"


def test_review_bold_per_line() -> None:
    assert ngff_rules._bold("- one MUST\n- two") == "- **one MUST**\n- **two**"


# ---------------------------------------------------------------------------
# example conversion (each version's format), with a stand-in example
# ---------------------------------------------------------------------------

MULTISCALE = {
    "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
    "datasets": [
        {
            "path": "s0",
            "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}],
        }
    ],
}
ARRAY = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4, 4],
    "data_type": "uint8",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4, 4]}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 0,
    "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    "dimension_names": ["y", "x"],
}


def _with_example(monkeypatch: pytest.MonkeyPatch, example: Example) -> Rule:
    rule = get_rule("YZ-AX-001")
    monkeypatch.setattr(_model, "_load_examples", lambda: {rule.code: (example, {})})
    return rule


@pytest.mark.parametrize("version", ngff_rules.MODULES)
def test_model_example_conversion(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    # a valid v0.5 document converts to a valid document of every version
    valid = Example(model="Multiscale", json=json.dumps(MULTISCALE))
    rule = _with_example(monkeypatch, valid)
    assert ngff_rules.emitted_codes(rule, version) == set()
    # and an invalid one stays invalid
    bad = {**MULTISCALE, "axes": [MULTISCALE["axes"][0]] * 2}
    rule = _with_example(monkeypatch, Example(model="Multiscale", json=json.dumps(bad)))
    assert ngff_rules.emitted_codes(rule, version)


@pytest.mark.parametrize("version", ngff_rules.MODULES)
def test_store_example_conversion(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    pytest.importorskip("fsspec")
    ome = {"version": "0.5", "multiscales": [MULTISCALE]}
    group = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}}
    files = {"zarr.json": group, "s0/zarr.json": ARRAY}
    rule = _with_example(monkeypatch, Example(store=json.dumps(files)))
    example = ngff_rules.example_for(rule, version)
    assert example and example.store
    assert ngff_rules._store_codes(example.files) == set()
    # (validate it directly too: no codes could also mean an unexpected error)
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmp:
        for path, content in example.files.items():
            Path(tmp, path).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp, path).write_text(json.dumps(content), encoding="utf-8")
        validate_zarr_store(tmp)
