"""Integrity checks for the rule-code registry (`yaozarrs._rules`)."""

from __future__ import annotations

import importlib.util
import re
import sys
import warnings
from pathlib import Path
from typing import get_args

import pytest
from packaging.version import Version

from yaozarrs._rules import (
    RULES,
    Rule,
    RuleWarning,
    SpecSource,
    error_code,
    get_rule,
    rewrap_error,
    rule_error,
    warn_rule,
)
from yaozarrs._rules._model import Level, _load_examples
from yaozarrs._rules._registry import TRACKED_VERSIONS

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "ngff_rules", ROOT / "scripts" / "ngff_rules.py"
)
assert _spec and _spec.loader
ngff_rules = importlib.util.module_from_spec(_spec)
sys.modules["ngff_rules"] = ngff_rules  # (for its dataclasses)
_spec.loader.exec_module(ngff_rules)

# RFC 2119 keywords that have the same strength as each level
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

    assert rule.level in get_args(Level) and rule.introduced
    assert rule.sources or not rule.checkable
    versions = [Version(s.doc.version) for s in rule.sources]
    assert Version(rule.introduced) <= versions[0]
    if rule.removed:
        assert Version(rule.introduced) < Version(rule.removed)
        assert all(v < Version(rule.removed) for v in versions)
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
    assert v4.doc.cache_name == "ome-ngff-spec-0.4-plate.schema"


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


def test_docs_render() -> None:
    pytest.importorskip("fsspec")  # coverage of store examples
    docs = ngff_rules.expected_docs()
    assert set(docs) == {"index.md"} | {f"{c}.md" for c in RULES}
