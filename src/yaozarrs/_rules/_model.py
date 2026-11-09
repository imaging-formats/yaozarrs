"""Data model for rule codes (see `yaozarrs._rules`)."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any, Literal

# the strength of a statement: "MUST NOT" is a MUST, and "SHOULD NOT" a SHOULD
Level = Literal["MUST", "SHOULD", "MAY"]
Status = Literal["draft", "stable", "withdrawn"]
Severity = Literal["error", "warning"]

DOCS_URL = "https://imaging-formats.github.io/yaozarrs/rules/{code}/"
EXAMPLES_FILE = Path(__file__).parent / "rule_examples.toml"


@dataclass(frozen=True)
class SpecDoc:
    """One version of the spec source file, pinned to an immutable commit."""

    version: str
    ref: str
    """The tag or branch the commit belongs to (for display; `sha` is the pin)."""
    repo: str
    sha: str
    """The commit the published spec (ngff.openmicroscopy.org) is built from."""
    path: str

    @property
    def raw_url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.sha}/{self.path}"

    @property
    def cache_name(self) -> str:
        """File name of the vendored copy in `tests/spec_sources/`."""
        # (with the commit, so that moving a pin can't leave an old copy in place)
        stem = f"{self.repo.replace('/', '-')}-{self.ref}-{self.sha[:9]}"
        if self.path.endswith(".schema"):
            return f"{stem}-{self.path.rsplit('/', 1)[-1]}"
        return f"{stem}.{self.path.rsplit('.', 1)[-1]}"

    def line_url(self, lines: tuple[int, int]) -> str:
        """GitHub permalink to `lines` of this file, at the pinned commit."""
        start, end = lines
        anchor = f"L{start}" if start == end else f"L{start}-L{end}"
        # a rendered (Markdown) file only shows line numbers in its plain view
        plain = "" if self.path.endswith(".schema") else "?plain=1"
        return f"https://github.com/{self.repo}/blob/{self.sha}/{self.path}{plain}#{anchor}"

    def schema(self, name: str) -> SpecDoc:
        """Return the JSON schema file `name` of this spec version."""
        folder = self.path.rpartition("/")[0]
        path = f"{folder}/schemas/{name}" if folder else f"schemas/{name}"
        return replace(self, path=path)


@dataclass(frozen=True)
class SpecSource:
    """Where a `SpecDoc` states a rule: a verbatim prose quote, or a schema keyword."""

    doc: SpecDoc
    lines: tuple[int, int]
    quote: str = ""
    """The verbatim prose (whitespace-insensitive); empty for a JSON schema source."""
    # the statement is a requirement, but this version states it without an
    # RFC 2119 keyword (the rule's `notes` say why it counts)
    implied: bool = False
    pointer: str = ""
    """For a JSON schema source: the JSON Pointer of the keyword that states it."""
    value: Any = None
    """For a JSON schema source: the keyword's value, if checked (None: not checked)."""

    @property
    def is_schema(self) -> bool:
        """Whether this quotes one of the spec's JSON schemas, rather than its prose."""
        return self.doc.path.endswith(".schema")

    @property
    def permalink(self) -> str:
        return self.doc.line_url(self.lines)


@dataclass(frozen=True)
class Example:
    """A minimal input that violates a rule.

    Either a JSON document validated as `model`, or a `store`: a JSON object mapping
    metadata file paths (`zarr.json`, `s0/zarr.json`, ...) to their contents, validated
    with `validate_zarr_store`.  A rule's `example` is in the v0.5 format, and is
    converted for other versions (see `example_for` in `scripts/ngff_rules.py`); its
    `examples` are in the format of the version they are for.
    """

    model: str | None = None  # name of a model class in a version module
    json: str | None = None
    store: str | None = None

    @property
    def data(self) -> object:
        return json.loads(self.json or "null")

    @property
    def files(self) -> dict[str, dict]:
        return json.loads(self.store or "{}")


@dataclass(frozen=True)
class Rule:
    """A single normative statement (or yaozarrs-specific check)."""

    code: str
    """Permanent code: `YZ-<AREA>-NNN` (spec rule) or `YZ-NNN` (yaozarrs check)."""
    slug: str
    """Short kebab-case name, unique among rules (e.g. `axis-name-unique`)."""
    summary: str
    """One-sentence description of the requirement."""
    level: Level | None = None
    """Strength of the statement ("MUST NOT" is a MUST); None for yaozarrs checks."""
    sources: tuple[SpecSource, ...] = ()
    """Where each spec version states it (prose or JSON schema), in version order."""
    introduced: str | None = None
    """First spec version that states the rule; None for yaozarrs checks."""
    removed: str | None = None
    """First spec version that no longer states the rule, if any."""
    status: Status = "stable"
    """`draft` (first seen in a dev version), `stable`, or `withdrawn`."""
    severity: Severity | None = None
    """Overrides the default severity (`warning` for SHOULD, otherwise `error`)."""
    checkable: bool = True
    """False for statements about software behavior, which no document can violate."""
    supersedes: tuple[str, ...] = ()
    """Codes of the retired rules that this one replaces."""
    aliases: tuple[str, ...] = ()
    """Other names for the rule, e.g. another validator's identifier."""
    notes: str = ""
    """How yaozarrs applies the rule, or why a quote without a keyword counts."""

    @property
    def example(self) -> Example | None:
        """Minimal failing example (v0.5 format); None if the code is never reported."""
        return _load_examples().get(self.code, (None, {}))[0]

    @property
    def examples(self) -> dict[str, Example]:
        """Examples that replace `example` for specific versions."""
        return _load_examples().get(self.code, (None, {}))[1]

    @property
    def is_spec_rule(self) -> bool:
        """Whether this cites the spec (`YZ-<AREA>-NNN`), not a yaozarrs check."""
        return self.code.count("-") == 2

    @property
    def url(self) -> str:
        return DOCS_URL.format(code=self.code)

    @property
    def default_severity(self) -> Severity:
        if self.severity is not None:
            return self.severity
        return "warning" if self.level == "SHOULD" else "error"


@cache
def _load_examples() -> dict[str, tuple[Example | None, dict[str, Example]]]:
    """Load `rule_examples.toml` (on first use: only the tests and docs need it)."""
    data = tomllib.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
    return {
        code: (
            Example(**e["example"]) if "example" in e else None,
            {v: Example(**x) for v, x in e.get("examples", {}).items()},
        )
        for code, e in data.items()
    }
