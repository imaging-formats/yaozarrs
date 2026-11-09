"""Maintenance tool for the rule registry in `src/yaozarrs/_rules/rules.toml`.

Usage (from the repo root):

    uv run python scripts/ngff_rules.py fetch          # vendor pinned spec sources
    uv run python scripts/ngff_rules.py check [--fix]  # verify quotes are at `lines`
    uv run python scripts/ngff_rules.py relocate <repo> <ref> <path>
    uv run python scripts/ngff_rules.py coverage       # where each example fires
    uv run python scripts/ngff_rules.py uncited [v]    # keywords without a rule
    uv run python scripts/ngff_rules.py site           # pins match the website?
    uv run python scripts/ngff_rules.py docs           # generate docs/rules/
    uv run python scripts/ngff_rules.py review <v> [-o FILE]  # checklist, spec order
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import sys
import tempfile
import textwrap
import tomllib
import urllib.request
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from yaozarrs import v04, v05, v06, validate_zarr_store
from yaozarrs._errors import StorageValidationError, StorageValidationWarning
from yaozarrs._rules import RULES, Rule, RuleWarning, SpecDoc, SpecSource
from yaozarrs._rules._registry import REGISTRY_FILE, SPEC_DOCS

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    from yaozarrs._rules._model import Example

ROOT = Path(__file__).resolve().parent.parent
SPEC_SOURCES = ROOT / "tests" / "spec_sources"
DOCS_DIR = ROOT / "docs" / "rules"
TRACKED = tuple(SPEC_DOCS)  # the tracked spec versions, in order
SKIPPED_FILE = REGISTRY_FILE.parent / "skipped.toml"

# yaozarrs model module for each spec version it implements
MODULES: dict[str, ModuleType] = {"0.4": v04, "0.5": v05, "0.6": v06}
# env vars that change which rules fire; coverage is computed without them
BEHAVIOR_ENV_VARS = (
    "YAOZARRS_ALLOW_RISKY_NODE_NAMES",
    "YAOZARRS_IGNORE_RISKY_FOV_NAMES",
    "YAOZARRS_STRICT_FOV_NAMES",
)


# ---------------------------------------------------------------------------
# quote lookup
# ---------------------------------------------------------------------------


def _normalized(text: str) -> tuple[str, list[int]]:
    """`text` with each whitespace run as one space, and the line of each character."""
    chars: list[str] = []
    line_of: list[int] = []
    prev_space = True
    for lineno, line in enumerate(text.splitlines(), 1):
        for ch in line + "\n":
            if ch.isspace():
                if prev_space:
                    continue
                ch, prev_space = " ", True
            else:
                prev_space = False
            chars.append(ch)
            line_of.append(lineno)
    return "".join(chars), line_of


def _quote_spans(
    norm: str, line_of: list[int], quote: str
) -> list[tuple[int, int, tuple[int, int]]]:
    """(start, end, lines) of each occurrence of `quote` in a `_normalized` text."""
    needle = " ".join(quote.split())
    found = []
    idx = norm.find(needle)
    while idx != -1:
        end = idx + len(needle)
        found.append((idx, end, (line_of[idx], line_of[end - 1])))
        idx = norm.find(needle, idx + 1)
    return found


def find_quote(text: str, quote: str) -> list[tuple[int, int]]:
    """Return the (start, end) line spans where `quote` occurs, ignoring whitespace."""
    return [lines for *_, lines in _quote_spans(*_normalized(text), quote)]


def pointer_spans(text: str) -> dict[str, tuple[int, int]]:
    """The (first, last) lines of every node of a JSON document, by JSON Pointer.

    A member's lines start at its key, an array item's at its value.
    """
    newlines = [i for i, ch in enumerate(text) if ch == "\n"]
    decoder = json.JSONDecoder()
    spans: dict[str, tuple[int, int]] = {}

    def line(i: int) -> int:
        return bisect.bisect_right(newlines, i - 1) + 1

    def skip(i: int) -> int:
        while i < len(text) and text[i].isspace():
            i += 1
        return i

    def node(i: int, ptr: str, first: int) -> int:
        i = skip(i)
        if text[i] in "{[":
            close, n = ("}" if text[i] == "{" else "]"), 0
            i = skip(i + 1)
            while text[i] != close:
                if close == "}":
                    key, j = decoder.raw_decode(text, i)
                    j = skip(j)
                    assert text[j] == ":", f"expected ':' at line {line(j)}"
                    name = key.replace("~", "~0").replace("/", "~1")
                    i = node(j + 1, f"{ptr}/{name}", line(i))
                else:
                    i = node(i, f"{ptr}/{n}", line(i))
                    n += 1
                i = skip(i)
                if text[i] == ",":
                    i = skip(i + 1)
            i += 1
        else:
            _, i = decoder.raw_decode(text, i)
        spans[ptr] = (first, line(i - 1))
        return i

    node(0, "", line(skip(0)))
    return spans


def resolve_pointer(data: Any, ptr: str) -> Any:
    """The node of `data` at JSON Pointer `ptr` (KeyError or IndexError if none)."""
    for part in ptr.split("/")[1:] if ptr else ():
        part = part.replace("~1", "/").replace("~0", "~")
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def spec_text(doc: SpecDoc) -> str:
    """Return the vendored text of `doc`."""
    path = SPEC_SOURCES / doc.cache_name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `scripts/ngff_rules.py fetch`")
    return path.read_text(encoding="utf-8")


def check_pointer(src: SpecSource) -> str | None:
    """Return a problem, or None if `src`'s schema keyword is at its lines."""
    text = spec_text(src.doc)
    span = pointer_spans(text).get(src.pointer)
    if span is None:
        return f"pointer {src.pointer!r} not found"
    if src.value is not None:
        if (found := resolve_pointer(json.loads(text), src.pointer)) != src.value:
            return f"value at {src.pointer!r} is {found!r}, not {src.value!r}"
    if span != src.lines:
        return f"pointer found at {span}, not at {src.lines}"
    return None


def check_source(src: SpecSource) -> str | None:
    """Return a problem description, or None if `src` is where it says it is."""
    if src.is_schema:
        return check_pointer(src)
    return check_quote(src.doc, src.quote, src.lines)


def check_quote(doc: SpecDoc, quote: str, lines: tuple[int, int]) -> str | None:
    """Return a problem description, or None if `quote` is at `lines` of `doc`."""
    spans = find_quote(spec_text(doc), quote)
    if spans == [lines]:
        return None
    if not spans:
        return "quote not found"
    if len(spans) > 1:
        return f"quote found {len(spans)} times, at {spans}: extend it to be unique"
    return f"quote found at {spans}, not at {lines}"


# ---------------------------------------------------------------------------
# completeness: every keyword in the spec is quoted by a rule, or skipped
# ---------------------------------------------------------------------------

KEYWORD = re.compile(r"\b(MUST|SHOULD|MAY|REQUIRED|RECOMMENDED|OPTIONAL|SHALL)\b")
# the level each keyword states
KEYWORD_LEVEL = {
    **dict.fromkeys(("MUST", "REQUIRED", "SHALL"), "MUST"),
    **dict.fromkeys(("SHOULD", "RECOMMENDED"), "SHOULD"),
    **dict.fromkeys(("MAY", "OPTIONAL"), "MAY"),
}
# why a keyword in the spec has no rule
SKIP_REASONS = {
    "permission": "a MAY that only grants permission: nothing can violate it",
    "boilerplate": "RFC 2119 definitions and other text about the document itself",
    "duplicate": "restates a requirement that has a rule (name it in `note`)",
    "illustration": "the keyword is in an example, diagram, or code comment",
}


@dataclass(frozen=True)
class Skip:
    """Spec text with a keyword that deliberately has no rule (`skipped.toml`)."""

    version: str
    quote: str
    lines: tuple[int, int]
    reason: str
    note: str = ""


def load_skips() -> list[Skip]:
    """Load `skipped.toml`."""
    data = tomllib.loads(SKIPPED_FILE.read_text(encoding="utf-8"))
    return [
        Skip(
            version=s["version"],
            quote=s["quote"],
            lines=(s["lines"][0], s["lines"][1]),
            reason=s["reason"],
            note=s.get("note", ""),
        )
        for s in data.get("skip", [])
    ]


def _citations(version: str) -> list[tuple[str, tuple[int, int], str | None]]:
    """(quote, lines, rule level) of each rule source and skip entry (level None)."""
    cited: list[tuple[str, tuple[int, int], str | None]] = [
        (src.quote, src.lines, rule.level)
        for rule in RULES.values()
        for src in rule.sources
        if src.doc.version == version and not src.is_schema
    ]
    cited += [(s.quote, s.lines, None) for s in load_skips() if s.version == version]
    return cited


def _heading(lines: list[str], n: int) -> str:
    """The heading of the section that contains line `n` (1-based)."""
    heading, fence = "", ""
    for i, raw in enumerate(lines[:n]):
        line = raw.strip()
        if line.startswith(("```", "~~~")):  # (a `#` in a code block isn't a heading)
            fence = "" if line[:3] == fence else (fence or line[:3])
            continue
        if fence:
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        elif line and nxt and set(nxt) <= {"-", "="} and len(nxt) >= 3:
            heading = line
    return heading


def uncited(version: str) -> list[tuple[int, str, str]]:
    """(line, section, context) of each keyword that no rule or skip entry quotes.

    A MUST or SHOULD that only rules of other levels quote counts as uncited too (a
    MAY inside a MUST rule's quote is fine: see the MAY policy in the skill).
    """
    text = spec_text(SPEC_DOCS[version])
    norm, line_of = _normalized(text)
    # a quote counts only at its recorded lines (`check` reports any that moved)
    quoted = [
        (start, end, level)
        for quote, lines, level in _citations(version)
        for start, end, at in _quote_spans(norm, line_of, quote)
        if at == lines
    ]
    found = []
    for m in KEYWORD.finditer(norm):
        levels = {lvl for start, end, lvl in quoted if start <= m.start() < end}
        level = KEYWORD_LEVEL[m[0]]
        if not levels:
            why = ""
        elif level != "MAY" and None not in levels and level not in levels:
            why = f"(only quoted by {'/'.join(sorted(map(str, levels)))} rules) "
        else:
            continue
        n = line_of[m.start()]
        context = f"{norm[max(0, m.start() - 60) : m.start()]}<<{m[0]}>>"
        context += norm[m.end() : m.end() + 40]
        found.append((n, _heading(text.splitlines(), n), why + context.strip()))
    return found


# ---------------------------------------------------------------------------
# coverage: run each rule's example against every version module
# ---------------------------------------------------------------------------


@contextmanager
def _default_behavior() -> Generator[None]:
    saved = {k: os.environ.pop(k) for k in BEHAVIOR_ENV_VARS if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


# numpy-style dtype strings for zarr v2 `.zarray` files
_V2_DTYPES = {
    "bool": "|b1",
    "int8": "|i1",
    "uint8": "|u1",
    **{f"{k}{n}": f"<{k[0]}{n // 8}" for k in ("int", "uint") for n in (16, 32, 64)},
    "float16": "<f2",
    "float32": "<f4",
    "float64": "<f8",
    "complex64": "<c8",
    "complex128": "<c16",
}


def to_v04_store(files: dict[str, dict]) -> dict[str, dict]:
    """Convert a v0.5 (zarr v3) store example into the equivalent v0.4 (zarr v2)."""
    out: dict[str, dict] = {}
    for path, meta in files.items():
        prefix = path.removesuffix("zarr.json")
        if meta["node_type"] == "array":
            shape = meta["shape"]
            out[f"{prefix}.zarray"] = {
                "zarr_format": 2,
                "shape": shape,
                "chunks": shape,
                "dtype": _V2_DTYPES[meta["data_type"]],
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
            }
            continue
        attrs = dict(meta.get("attributes", {}))
        ome = dict(attrs.pop("ome", {}))
        ome.pop("version", None)
        # in v0.4 the version lives on each of these objects instead
        if "multiscales" in ome:
            ome["multiscales"] = [{**ms, "version": "0.4"} for ms in ome["multiscales"]]
        for key in ("plate", "well", "image-label"):
            if key in ome:
                ome[key] = {**ome[key], "version": "0.4"}
        out[f"{prefix}.zgroup"] = {"zarr_format": 2}
        out[f"{prefix}.zattrs"] = {**attrs, **ome}
    return out


def _dataset_to_v06(ds: dict) -> dict:
    """Convert a v0.5 `datasets` entry into the equivalent v0.6."""
    ts = ds.get("coordinateTransformations")
    if not isinstance(ts, list) or not ts:
        return ds
    ref = {"input": {"path": ds.get("path")}, "output": {"name": "physical"}}
    # v0.6 has a single transformation: several become a sequence
    t = ts[0] if len(ts) == 1 else {"type": "sequence", "transformations": ts}
    return {**ds, "coordinateTransformations": [{**t, **ref}]}


def _multiscale_to_v06(ms: dict) -> dict:
    """Convert a v0.5 `multiscales` entry into the equivalent v0.6."""
    out = {}
    if "axes" in ms:
        out["coordinateSystems"] = [{"name": "physical", "axes": ms["axes"]}]
    out.update({k: v for k, v in ms.items() if k != "axes"})
    if isinstance(ms.get("datasets"), list):
        out["datasets"] = [_dataset_to_v06(ds) for ds in ms["datasets"]]
    return out


def to_v06_store(files: dict[str, dict]) -> dict[str, dict]:
    """Convert a v0.5 (zarr v3) store example into the equivalent v0.6."""
    out: dict[str, dict] = {}
    for path, meta in files.items():
        ome = meta.get("attributes", {}).get("ome")
        if ome is None:
            out[path] = meta
            continue
        ome: dict[str, Any] = {**ome, "version": "0.6"}
        if "multiscales" in ome:
            ome["multiscales"] = [_multiscale_to_v06(m) for m in ome["multiscales"]]
        attrs = {**meta["attributes"], "ome": ome}
        out[path] = {**meta, "attributes": attrs}
    return out


def example_for(rule: Rule, version: str) -> Example | None:
    """Return `rule`'s example for `version`, in that version's format."""
    if version in rule.examples:
        return rule.examples[version]
    if (ex := rule.example) is None:
        return None
    if ex.store is not None:
        convert = {"0.4": to_v04_store, "0.6": to_v06_store}.get(version)
        files = convert(ex.files) if convert else ex.files
        return replace(ex, store=json.dumps(files, indent=2))
    if version == "0.6" and ex.model in ("Multiscale", "Dataset"):
        fn = _multiscale_to_v06 if ex.model == "Multiscale" else _dataset_to_v06
        return replace(ex, json=json.dumps(fn(cast("dict", ex.data)), indent=2))
    return ex


def _store_codes(files: dict[str, dict]) -> set[str]:
    """Write `files` to a temporary store, validate it, and return emitted codes."""
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmp:
        for path, content in files.items():
            dest = Path(tmp, path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(content), encoding="utf-8")
        codes: set[str] = set()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                validate_zarr_store(tmp)
            except StorageValidationError as e:
                codes.update(c for d in e.errors() if (c := d.get("code")))
            except ValidationError as e:
                codes.update(err["type"] for err in e.errors())
            except ValueError:
                pass  # e.g. the OME-Zarr version could not be determined
        for x in w:
            if isinstance(x.message, StorageValidationWarning):
                codes.update(c for d in x.message.warnings() if (c := d.get("code")))
            elif isinstance(x.message, RuleWarning):
                codes.add(x.message.code)
        return codes


def emitted_codes(rule: Rule, version: str) -> set[str]:
    """Codes emitted when validating `rule`'s example as spec `version`."""
    if (ex := example_for(rule, version)) is None:
        return set()
    with _default_behavior():
        if ex.store is not None:
            return _store_codes(ex.files)
        if (model := getattr(MODULES[version], ex.model or "", None)) is None:
            return set()
        codes: set[str] = set()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                model.model_validate_json(ex.json or "null")
            except ValidationError as e:
                codes.update(err["type"] for err in e.errors())
    codes.update(x.message.code for x in w if isinstance(x.message, RuleWarning))
    return codes


def coverage(rule: Rule) -> list[str]:
    """Spec versions in which yaozarrs emits `rule.code` for its example."""
    return [v for v in MODULES if rule.code in emitted_codes(rule, v)]


# ---------------------------------------------------------------------------
# review: every rule and skip entry of one version, in the order of the spec
# ---------------------------------------------------------------------------


def _unlink(text: str) -> str:
    """`text` without its links to anchors in the spec (they'd be broken here)."""
    return re.sub(r"\[([^\]]*)\]\(#[^)]*\)", r"\1", text)


def _bold(text: str) -> str:
    """`text` in bold, line by line (bold can't span list items or blank lines)."""
    out = []
    for line in text.split("\n"):
        m = re.match(r"(\s*(?:[-*+]|\d+\.|:)?\s*)(.*?)(\s*)$", line)
        assert m
        out.append(f"{m[1]}**{m[2]}**{m[3]}" if m[2] else line)
    return "\n".join(out)


def _excerpt(lines: list[str], span: tuple[int, int], quote: str) -> list[str]:
    """The spec lines at `span`, with `quote` in bold, as a Markdown blockquote."""
    text = "\n".join(lines[span[0] - 1 : span[1]]).strip()
    pattern = r"\s+".join(map(re.escape, quote.split()))
    text = _unlink(re.sub(pattern, lambda m: _bold(m[0]), text, count=1))
    return [f"> {line}".rstrip() for line in text.splitlines()]


def _review_item(head: str, body: list[str]) -> list[str]:
    return [f"- [ ] {head}", "", *(f"  {b}".rstrip() for b in body), ""]


def review(version: str) -> str:
    """Markdown checklist of `version`'s rules and skip entries, in spec order."""
    doc = SPEC_DOCS[version]
    text = spec_text(doc)
    lines = text.splitlines()
    norm, line_of = _normalized(text)

    def offset(quote: str, span: tuple[int, int]) -> int:
        found = [s for s, _, at in _quote_spans(norm, line_of, quote) if at == span]
        if not found:
            raise ValueError(f"{quote!r} is not at {span}: run `check --fix`")
        return found[0]

    def link(span: tuple[int, int], permalink: str) -> str:
        start, end = span
        return f"[L{start}{f'-{end}' if end != start else ''}]({permalink})"

    prose: list[tuple[int, tuple[int, int], list[str]]] = []
    schema: list[tuple[str, int, list[str]]] = []
    n_rules = 0
    for rule in RULES.values():
        here = [s for s in rule.sources if s.doc.version == version]
        if not here:
            continue
        n_rules += 1
        checked = version in MODULES and rule.code in emitted_codes(rule, version)
        about = [rule.summary, ""]
        if rule.notes:
            # rule pages aren't next to this file: keep the code, drop the link
            notes = re.sub(r"\[(YZ-[A-Z0-9-]+)\]\(\1\.md\)", r"\1", rule.notes)
            about += [f"*Notes:* {notes}", ""]
        flags = ["code reported" if checked else "code not reported"]
        if not rule.checkable:
            flags = ["about software behavior"]
        if rule.removed:
            flags.append(f"removed in {rule.removed}")
        for src in here:
            kind = "schema" if src.is_schema else "implied" if src.implied else ""
            head = f"**{rule.code}** · {rule.level}" + (f" · {kind}" if kind else "")
            head += f" · `{rule.slug}` · {link(src.lines, src.permalink)}"
            tail = [f"*{'; '.join(flags)}*"]
            if src.is_schema:
                name = src.doc.path.rsplit("/", 1)[-1]
                cited = spec_text(src.doc).splitlines()[src.lines[0] - 1 : src.lines[1]]
                cited = textwrap.dedent("\n".join(cited)).rstrip(",").splitlines()
                body = [f"`{name}` `{src.pointer}`:", "", "```json", *cited, "```"]
                body += ["", *about]
                schema.append((name, src.lines[0], _review_item(head, body + tail)))
            else:
                body = [*_excerpt(lines, src.lines, src.quote), "", *about, *tail]
                pos = offset(src.quote, src.lines)
                prose.append((pos, src.lines, _review_item(head, body)))
    skips = [s for s in load_skips() if s.version == version]
    for skip in skips:
        head = f"skipped · {skip.reason} · {link(skip.lines, doc.line_url(skip.lines))}"
        body = _excerpt(lines, skip.lines, skip.quote)
        body += ["", skip.note or SKIP_REASONS[skip.reason]]
        prose.append(
            (offset(skip.quote, skip.lines), skip.lines, _review_item(head, body))
        )

    out = [
        f"# NGFF {version}: rule review",
        "",
        f"Every rule and skip entry for {version}, in the order the spec states them, "
        f"from [`{doc.repo}@{doc.ref}` `{doc.path}`]"
        f"(https://github.com/{doc.repo}/blob/{doc.sha}/{doc.path}) "
        f"(commit `{doc.sha[:9]}`, as published on ngff.openmicroscopy.org).",
        "",
        "For each item, check that the bold quote is the statement, that the level "
        "(MUST / SHOULD / MAY) is the one that governs it, that the summary says what "
        "the spec says (no more, no less), and for a skip, that its reason holds. "
        "Rules from the JSON schemas are at the end.",
        "",
        REPORTED_NOTE,
        "",
        f"{n_rules} rules ({len(prose) - len(skips)} prose quotes, "
        f"{len(schema)} schema quotes), {len(skips)} skip entries.",
        "",
    ]
    section = None
    for _, span, item in sorted(prose, key=lambda e: e[0]):
        heading = _heading(lines, span[0])
        if heading != section:
            out += [f"## {heading or '(top)'}", ""]
            section = heading
        out += item
    if schema:
        out += ["## JSON schemas", ""]
        for *_, item in sorted(schema, key=lambda e: e[:2]):
            out += item
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _download(url: str) -> str:
    # a token (e.g. in CI) avoids GitHub's unauthenticated API rate limit
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as resp:
        return resp.read().decode("utf-8")


def cmd_fetch(_: argparse.Namespace) -> int:
    """Download the pinned spec sources into `tests/spec_sources/`."""
    SPEC_SOURCES.mkdir(parents=True, exist_ok=True)
    cited = {src.doc for rule in RULES.values() for src in rule.sources}
    docs = sorted({*SPEC_DOCS.values(), *cited}, key=lambda d: d.cache_name)
    # (each file name includes its commit, so a moved pin leaves the old file behind)
    for stale in sorted(
        {p.name for p in SPEC_SOURCES.glob("*.*")}
        - {d.cache_name for d in docs}
        - {"README.md"}
    ):
        (SPEC_SOURCES / stale).unlink()
        print(f"removed: {stale}")
    for doc in docs:
        dest = SPEC_SOURCES / doc.cache_name
        if dest.exists():
            print(f"exists: {dest.relative_to(ROOT)}")
            continue
        dest.write_text(_download(doc.raw_url), encoding="utf-8", newline="\n")
        print(f"fetched: {dest.relative_to(ROOT)}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Verify every quote is at its recorded lines (optionally fix the lines)."""
    problems = 0
    for rule in RULES.values():
        for src in rule.sources:
            if msg := check_source(src):
                problems += 1
                print(f"{rule.code} [{src.doc.version}]: {msg}")
    for skip in load_skips():
        if msg := check_quote(SPEC_DOCS[skip.version], skip.quote, skip.lines):
            problems += 1
            print(f"skip {skip.quote[:40]!r} [{skip.version}]: {msg}")
    if not (args.fix and problems):
        return 1 if problems else 0

    # tomlkit preserves comments and formatting; it is only in the `dev` group
    import tomlkit

    unfixed = 0
    doc = tomlkit.parse(REGISTRY_FILE.read_text(encoding="utf-8"))
    for code, entry in doc["rules"].items():  # type: ignore[union-attr]
        for src in entry.get("sources", []):
            for version in list(src["lines"]):
                spec = SPEC_DOCS[version]
                if "schema" in src:
                    span = pointer_spans(spec_text(spec.schema(src["schema"])))
                    spans = [span[p]] if (p := src["pointer"]) in span else []
                else:
                    spans = find_quote(spec_text(spec), src["quote"])
                if len(spans) == 1:
                    src["lines"][version] = list(spans[0])
                else:
                    unfixed += 1
                    print(f"{code} [{version}]: cannot fix, found at {spans}")
    REGISTRY_FILE.write_text(tomlkit.dumps(doc), encoding="utf-8")
    skipped = tomlkit.parse(SKIPPED_FILE.read_text(encoding="utf-8"))
    for skip in skipped.get("skip", []):
        spans = find_quote(spec_text(SPEC_DOCS[skip["version"]]), skip["quote"])
        if len(spans) == 1:
            skip["lines"] = list(spans[0])
        else:
            unfixed += 1
            print(f"skip {skip['quote'][:40]!r}: cannot fix, quote found at {spans}")
    SKIPPED_FILE.write_text(tomlkit.dumps(skipped), encoding="utf-8")
    print("updated line numbers where the quote was found exactly once")
    return 1 if unfixed else 0


def cmd_uncited(args: argparse.Namespace) -> int:
    """List spec lines with a keyword that no rule or skip entry covers."""
    total = 0
    for version in args.versions or TRACKED:
        found = uncited(version)
        total += len(found)
        print(f"== {version}: {len(found)} uncited")
        for n, section, text in found:
            print(f"L{n} [{section[:40]}] {text}")
    return 1 if total else 0


def cmd_relocate(args: argparse.Namespace) -> int:
    """Look for the latest quote of every rule in a new version of the spec."""
    text = _download(
        f"https://raw.githubusercontent.com/{args.repo}/{args.ref}/{args.path}"
    )
    for rule in RULES.values():
        prose = [src for src in rule.sources if not src.is_schema]
        if not prose or rule.removed:
            continue
        for src in (s for s in prose if s.doc.version == prose[-1].doc.version):
            spans = find_quote(text, src.quote)
            status = f"found at {spans}" if spans else "NOT FOUND (changed or removed?)"
            print(f"{rule.code} ({src.doc.version} quote): {status}")
    return 0


# the repo that ngff.openmicroscopy.org is built from, and its spec submodules
SITE_REPO = "ome/ngff"
SITE_SUBMODULE = "specifications/{version}"


def published_pins() -> dict[str, tuple[str, str]]:
    """{version: (repo, sha)} that the website's `ome/ngff` main pins, per version."""
    pins = {}
    for version in SPEC_DOCS:
        path = SITE_SUBMODULE.format(version=version)
        url = f"https://api.github.com/repos/{SITE_REPO}/contents/{path}?ref=main"
        info = json.loads(_download(url))
        if info.get("type") != "submodule":
            raise ValueError(
                f"{SITE_REPO}:{path} is a {info.get('type')}, not a submodule"
            )
        repo = info["submodule_git_url"].removeprefix("https://github.com/")
        pins[version] = (repo.removesuffix(".git"), info["sha"])
    return pins


def cmd_site(_: argparse.Namespace) -> int:
    """Check that each pinned spec is the commit ngff.openmicroscopy.org publishes."""
    problems = 0
    for version, (repo, sha) in published_pins().items():
        doc = SPEC_DOCS[version]
        ok = (doc.repo, doc.sha) == (repo, sha)
        problems += not ok
        status = "ok" if ok else f"MOVED: the site publishes {repo}@{sha}"
        print(f"{version}: {doc.repo}@{doc.sha[:9]} {status}")
    return 1 if problems else 0


def cmd_coverage(_: argparse.Namespace) -> int:
    """Print the versions in which each rule's example fires."""
    for rule in RULES.values():
        checked = ", ".join(coverage(rule)) or "-"
        print(f"{rule.code:<12} {checked}")
    return 0


# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------


def _versions(rule: Rule, short: bool = False) -> str:
    if rule.introduced is None:
        return "—"
    if rule.removed:
        versions = sorted({s.doc.version for s in rule.sources}, key=TRACKED.index)
        if short:  # e.g. 0.4-0.5
            return "-".join(dict.fromkeys((versions[0], versions[-1])))
        return f"{', '.join(versions)} (removed in {rule.removed})"
    return f"{rule.introduced}+"


def _rendered_examples(rule: Rule, checked: list[str]) -> list[tuple[str, str]]:
    """(introduction, JSON) of `rule`'s examples, one for each distinct document."""
    by_doc: dict[tuple[str | None, str], list[str]] = {}
    for version in checked or list(MODULES):
        if (ex := example_for(rule, version)) is not None:
            is_store = ex.store is not None
            body = ex.store if is_store else ex.json
            key = (None if is_store else ex.model, (body or "").strip())
            by_doc.setdefault(key, []).append(version)
        if not checked and by_doc:
            break
    out = []
    for (model, body), versions in by_doc.items():
        if model is None:
            names = ", ".join(f"v{v}" for v in versions)
            intro = (
                f"This {names} store (metadata files only) fails "
                "`yaozarrs.validate_zarr_store`:"
            )
        else:
            names = " or ".join(f"`{MODULES[v].__name__}.{model}`" for v in versions)
            intro = f"This document fails validation as {names}:"
        out.append((intro, body))
    return out


# what "Checked in" means, for the docs
CHECKED_NOTE = (
    "*Checked in* lists the spec versions in which yaozarrs reports a rule's code "
    "when a document breaks the rule. If none are listed, yaozarrs may still reject "
    "such a document, but not under this code (yet)."
)
# and for the review checklist, whose readers know about the examples
REPORTED_NOTE = (
    "*Code reported* means yaozarrs reports the rule's code for its example (in "
    "`rule_examples.toml`). A rule whose code isn't reported may still be enforced, "
    "under another code or none."
)


# hover text for the labels in a rule page's facts box
FACT_TIPS = {
    "Level": "How strongly the NGFF spec states the rule (its RFC 2119 keyword)",
    "Versions": "The spec versions that state the rule",
    "Severity": "What yaozarrs reports when a document breaks the rule",
    "Checked in": "The spec versions in which yaozarrs reports this code",
}
NOT_YET_TIP = (
    "yaozarrs doesn't report this code yet. It may still reject such a document, "
    "under another code or none."
)


def _facts_group(title: str, facts: list[tuple[str, str]], empty: str = "") -> str:
    """One labeled group (`NGFF spec` or `yaozarrs`) of a rule page's facts box."""
    rows = [f'<dt title="{FACT_TIPS[k]}">{k}</dt><dd>{v}</dd>' for k, v in facts]
    body = (
        ["<dl>", *rows, "</dl>"]
        if rows
        else [f'<p class="rule-facts__none">{empty}</p>']
    )
    return "\n".join(
        [
            '<div class="rule-facts__group">',
            f'<p class="rule-facts__title">{title}</p>',
            *body,
            "</div>",
        ]
    )


def _rule_header(rule: Rule, checked: list[str]) -> list[str]:
    """The summary and facts box at the top of a rule page (`.rule-` CSS classes)."""
    lines = [rule.summary, "{ .rule-summary }", ""]
    if rule.status == "draft":
        lines += [
            '!!! note "Draft"',
            "    This rule comes from a development version of the spec, so it may "
            "still change.",
            "",
        ]
    elif rule.status == "withdrawn":
        lines += [
            '!!! warning "Withdrawn"',
            "    This statement was dropped from the spec before it was released. "
            "The code is kept so that it is never reused.",
            "",
        ]
    if rule.is_spec_rule:
        level = f'<span class="rule-level">{rule.level}</span>'
        spec = _facts_group(
            "NGFF spec", [("Level", level), ("Versions", _versions(rule))]
        )
    else:
        spec = _facts_group(
            "NGFF spec", [], "Not in the spec; this is a yaozarrs check."
        )
    if rule.checkable:
        sev = rule.default_severity
        severity = f'<span class="rule-severity rule-severity--{sev}">{sev}</span>'
        seen = ", ".join(checked) or f'<abbr title="{NOT_YET_TIP}">not yet</abbr>'
        ours = _facts_group("yaozarrs", [("Severity", severity), ("Checked in", seen)])
    else:
        ours = _facts_group(
            "yaozarrs",
            [],
            "Not checked, since it is about software behavior and no document "
            "can violate it.",
        )
    links = []
    if rule.supersedes:
        links.append(
            "Supersedes " + ", ".join(f"[{c}]({c}.md)" for c in rule.supersedes)
        )
    if by := [r.code for r in RULES.values() if rule.code in r.supersedes]:
        links.append("Superseded by " + ", ".join(f"[{c}]({c}.md)" for c in by))
    if rule.aliases:
        links.append("Also known as " + ", ".join(f"`{a}`" for a in rule.aliases))
    box = ['<div class="rule-facts" markdown="block">', spec, ours]
    if links:
        box.append(
            f'<p class="rule-facts__extra" markdown="span">{" · ".join(links)}</p>'
        )
    return [*lines, *box, "</div>", ""]


# a rule code, or a GitHub reference like `ome/ngff#637`, in a rule's notes; skipping
# code spans and existing links
_NOTE_REFS = re.compile(
    r"(`[^`]*`|\[[^\]]*\]\([^)]*\))"
    r"|\b(YZ-(?:[A-Z]{2}-)?\d{3})\b"
    r"|\b([\w.-]+/[\w.-]+)#(\d+)\b"
)


def _link_notes(rule: Rule) -> str:
    """`rule.notes`, with links to the other rules and GitHub issues it mentions."""

    def link(m: re.Match[str]) -> str:
        if (code := m[2]) and code in RULES and code != rule.code:
            return f"[{code}]({code}.md)"
        if m[3]:
            return f"[{m[0]}](https://github.com/{m[3]}/issues/{m[4]})"
        return m[0]

    return _NOTE_REFS.sub(link, rule.notes)


def _spec_quote(src: SpecSource) -> list[str]:
    """`src`'s quote: the prose, or the cited lines of a JSON schema as code."""
    if not src.is_schema:
        return [_unlink(" ".join(src.quote.split()))]
    start, end = src.lines
    if start == end:  # (a one-line keyword: its pointer and value say it all)
        value = resolve_pointer(json.loads(spec_text(src.doc)), src.pointer)
        return [f"`{src.pointer}`: `{json.dumps(value)}`"]
    cited = "\n".join(spec_text(src.doc).splitlines()[start - 1 : end])
    cited = textwrap.dedent(cited).rstrip(",")  # (the keyword, not what follows)
    return [f"`{src.pointer}`", "", "```json", *cited.splitlines(), "```"]


def _spec_source(src: SpecSource) -> str:
    """The version of `src`, and a permalink to its lines."""
    start, end = src.lines
    file = src.doc.path.rsplit("/", 1)[-1]
    where = f"line {start}" if start == end else f"lines {start}-{end}"
    tip = f"{src.doc.repo}@{src.doc.ref}, commit {src.doc.sha[:7]}"
    return (
        f'<span class="rule-source__item"><span class="rule-source__version">'
        f"{src.doc.version}</span> "
        f'[{file}, {where}]({src.permalink} "{tip}"){{ .external }}</span>'
    )


def _spec_texts(rule: Rule, note_schema: bool = True) -> list[str]:
    """Each distinct quote of `rule`, newest first, with the versions that state it."""
    groups: dict[tuple[tuple[str, ...], bool, bool], list[SpecSource]] = {}
    for src in reversed(rule.sources):
        key = (tuple(_spec_quote(src)), src.implied, src.is_schema)
        groups.setdefault(key, []).append(src)
    lines = []
    for (quote, implied, is_schema), srcs in groups.items():
        caption = " ".join(_spec_source(s) for s in srcs)
        if implied:
            caption += (
                ' <span class="rule-source__note">'
                "Stated without an RFC 2119 keyword; see Notes.</span>"
            )
        if is_schema and note_schema:  # (unless the heading says so already)
            caption += (
                ' <span class="rule-source__note">Stated in the JSON schema.</span>'
            )
        lines += [
            *(f"> {line}".rstrip() for line in quote),
            "",
            caption,
            "{ .rule-source }",
            "",
        ]
    return lines


def render_rule(rule: Rule, checked: list[str]) -> str:
    """Markdown page for one rule."""
    lines = [f"# {rule.code}: `{rule.slug}`", ""]
    lines += _rule_header(rule, checked)
    if rule.notes:
        lines += ["## Notes", "", _link_notes(rule), ""]
    if examples := _rendered_examples(rule, checked):
        lines += ["## Example", ""]
        for intro, body in examples:
            lines += [intro, "", "```json", body, "```", ""]
    if rule.sources:
        schema = {s.is_schema for s in rule.sources}
        heading = {
            frozenset({False}): "Spec text",
            frozenset({True}): "JSON schema",  # (the spec's prose is silent)
        }.get(frozenset(schema), "Spec text and JSON schema")
        lines += [f"## {heading}", "", *_spec_texts(rule, note_schema=len(schema) > 1)]
    return "\n".join(lines)


# what each code's area prefix covers (for the filter on the docs index)
AREAS = {
    "AX": "axes",
    "CS": "coordinateSystems",
    "MS": "multiscales",
    "DS": "datasets",
    "CT": "coordinateTransformations",
    "SC": "scene",
    "LB": "labels",
    "IL": "image-label",
    "PL": "plate",
    "WL": "well",
    "BF": "bioformats2raw",
    "OM": "omero",
    "ZR": "zarr / storage layout",
    "GN": "general",
}


def _filter_data() -> str:
    """JSON for `docs/javascripts/rule_filter.js`: each rule's level, versions, area."""
    rules = {
        r.code: {
            "level": r.level,
            "versions": sorted({s.doc.version for s in r.sources}, key=TRACKED.index),
            "area": r.code.split("-")[1] if r.is_spec_rule else "YZ",
        }
        for r in RULES.values()
    }
    areas = {**AREAS, "YZ": "yaozarrs checks"}
    data = {"versions": TRACKED, "areas": areas, "rules": rules}
    # (`</` can't appear inside a <script>)
    return json.dumps(data, separators=(",", ":")).replace("</", "<\\/")


def render_index(checked: dict[str, list[str]]) -> str:
    """Markdown table of all rules."""
    tracked = ", ".join(TRACKED)

    def code(r: Rule) -> str:  # (like ruff's rules page, only the name is a link)
        return f"`{r.code}`"

    def seen(r: Rule) -> str:
        return ", ".join(checked[r.code]) or "—"

    def about(r: Rule) -> str:  # (the name, with the summary under it)
        return f"[`{r.slug}`]({r.code}.md)<br>{r.summary}"

    lines = [
        "---",
        "hide: [toc]",  # (the tables need the width)
        "---",
        "",
        "# Rule codes",
        "",
        "Each code identifies one normative statement in the "
        "[NGFF specification](https://ngff.openmicroscopy.org/), and links to the "
        "exact text in the spec that states it, for each spec version. yaozarrs "
        "errors and warnings for spec violations carry these codes once they are "
        "tagged.",
        "",
        "- `YZ-<AREA>-<NNN>` codes refer to statements in the NGFF spec or its JSON "
        "schemas.  These are codes we hope to upstream to an official spec someday.",
        "- `YZ-<NNN>` codes are checks that yaozarrs adds beyond the spec.",
        "",
        "Codes are permanent: a code is never renumbered or reused. If a statement "
        "changes meaning in a later spec version, its code is retired and a new one "
        "supersedes it.",
        "",
        "!!! note",
        "    These codes are assigned by yaozarrs; they are not (yet) part of the NGFF "
        "specification itself (see "
        "[ome/ngff#646](https://github.com/ome/ngff/issues/646)). If official codes "
        "are adopted, each spec rule will record its official code. Spec versions "
        f"audited so far: {tracked}. Every RFC 2119 keyword in those versions is "
        "inside the quote of a rule here, or of an entry (with a reason) in "
        "`skipped.toml`.",
        "",
        CHECKED_NOTE,
        "",
        f'<script type="application/json" id="rule-data">{_filter_data()}</script>',
        "",
        # (one table, like ruff's: the yaozarrs-only checks are an area in the filter;
        # the summary states the level: "must", "should", or "may")
        *_rule_table(
            ("Code", "Rule", "Versions", "Checked in"),
            [
                (code(r), about(r), _versions(r, short=True), seen(r))
                for r in RULES.values()
            ],
        ),
    ]
    return "\n".join(lines)


def _rule_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    """A Markdown table, styled by `.rule-table` (`docs/stylesheets/extra.css`)."""
    out = ['<div class="rule-table" markdown>', ""]
    out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return [*out, "", "</div>", ""]


def expected_docs() -> dict[str, str]:
    """Map of file name -> content for `docs/rules/`."""
    checked = {code: coverage(rule) for code, rule in RULES.items()}
    files = {"index.md": render_index(checked)}
    files.update({f"{c}.md": render_rule(r, checked[c]) for c, r in RULES.items()})
    return files


def cmd_review(args: argparse.Namespace) -> int:
    """Write a review checklist of one version's rules, in spec order."""
    content = review(args.version)
    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(content)
    return 0


def cmd_docs(_: argparse.Namespace) -> int:
    """Write the generated docs to `docs/rules/` (gitignored)."""
    expected = expected_docs()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in {p.name for p in DOCS_DIR.glob("*.md")} - set(expected):
        (DOCS_DIR / stale).unlink()
    for name, content in expected.items():
        (DOCS_DIR / name).write_text(content, encoding="utf-8")
    print(f"wrote {len(expected)} files to {DOCS_DIR.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    sub.add_parser("fetch").set_defaults(func=cmd_fetch)
    p = sub.add_parser("check")
    p.add_argument("--fix", action="store_true", help="rewrite line numbers")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("relocate")
    p.add_argument("repo", help="e.g. ome/ngff-spec")
    p.add_argument("ref", help="tag or commit sha")
    p.add_argument("path", help="spec file path in the repo, e.g. index.md")
    p.set_defaults(func=cmd_relocate)
    sub.add_parser("coverage").set_defaults(func=cmd_coverage)
    sub.add_parser("site").set_defaults(func=cmd_site)
    p = sub.add_parser("uncited")
    p.add_argument("versions", nargs="*", help="spec versions (default: all)")
    p.set_defaults(func=cmd_uncited)
    sub.add_parser("docs").set_defaults(func=cmd_docs)
    p = sub.add_parser("review")
    p.add_argument("version", choices=list(SPEC_DOCS))
    p.add_argument("-o", "--output", help="file to write (default: stdout)")
    p.set_defaults(func=cmd_review)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
