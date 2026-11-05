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
from yaozarrs._rules import RULES, Rule, RuleWarning, SpecDoc
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


def spec_text(doc: SpecDoc) -> str:
    """Return the vendored text of `doc`."""
    path = SPEC_SOURCES / doc.cache_name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `scripts/ngff_rules.py fetch`")
    return path.read_text(encoding="utf-8")


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


def _citations(version: str) -> list[tuple[str, tuple[int, int]]]:
    """(quote, lines) of each rule source and skip entry for `version`'s spec prose."""
    cited = [
        (src.quote, src.lines)
        for rule in RULES.values()
        for src in rule.sources
        if src.doc.version == version and not src.is_schema
    ]
    cited += [(s.quote, s.lines) for s in load_skips() if s.version == version]
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
    """(line, section, context) of each keyword that no rule or skip entry quotes."""
    text = spec_text(SPEC_DOCS[version])
    norm, line_of = _normalized(text)
    # a quote counts only at its recorded lines (`check` reports any that moved)
    quoted = [
        (start, end)
        for quote, lines in _citations(version)
        for start, end, at in _quote_spans(norm, line_of, quote)
        if at == lines
    ]
    found = []
    for m in KEYWORD.finditer(norm):
        if not any(start <= m.start() < end for start, end in quoted):
            n = line_of[m.start()]
            context = f"{norm[max(0, m.start() - 60) : m.start()]}<<{m[0]}>>"
            context += norm[m.end() : m.end() + 40]
            found.append((n, _heading(text.splitlines(), n), context.strip()))
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
                cited = textwrap.dedent("\n".join(cited)).splitlines()
                body = [f"`{name}`:", "", "```json", *cited, "```", "", *about]
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
        dest.write_text(_download(doc.raw_url), encoding="utf-8")
        print(f"fetched: {dest.relative_to(ROOT)}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Verify every quote is at its recorded lines (optionally fix the lines)."""
    problems = 0
    for rule in RULES.values():
        for src in rule.sources:
            if msg := check_quote(src.doc, src.quote, src.lines):
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
                    spec = spec.schema(src["schema"])
                spans = find_quote(spec_text(spec), src["quote"])
                if len(spans) == 1:
                    src["lines"][version] = list(spans[0])
                else:
                    unfixed += 1
                    print(f"{code} [{version}]: cannot fix, quote found at {spans}")
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


def _versions(rule: Rule) -> str:
    if rule.introduced is None:
        return "—"
    if rule.removed:
        versions = sorted({s.doc.version for s in rule.sources}, key=TRACKED.index)
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


# what "reported" means, for the docs and the review checklist
REPORTED_NOTE = (
    "*Reported* means yaozarrs reports the rule's code for its example. A rule "
    "whose code isn't reported may still be enforced, under another code or none."
)


def _checked_in(rule: Rule, checked: list[str]) -> str:
    if checked:
        return ", ".join(checked)
    if not rule.checkable:
        return "— (about software behavior: no document can violate it)"
    return "—"


def render_rule(rule: Rule, checked: list[str]) -> str:
    """Markdown page for one rule."""
    lines = [f"# {rule.code}: `{rule.slug}`", ""]
    lines += [f"> {rule.summary}", ""]
    meta = [
        ("Spec level", rule.level or "— (not from the NGFF spec)"),
        ("yaozarrs severity", rule.default_severity),
        ("Status", rule.status),
        ("Spec versions", _versions(rule)),
        ("Reported by yaozarrs in", _checked_in(rule, checked)),
    ]
    if rule.supersedes:
        meta.append(
            ("Supersedes", ", ".join(f"[{c}]({c}.md)" for c in rule.supersedes))
        )
    superseded_by = [r.code for r in RULES.values() if rule.code in r.supersedes]
    if superseded_by:
        meta.append(
            ("Superseded by", ", ".join(f"[{c}]({c}.md)" for c in superseded_by))
        )
    if rule.aliases:
        meta.append(("Aliases", ", ".join(f"`{a}`" for a in rule.aliases)))
    lines += ["| | |", "|---|---|"]
    lines += [f"| **{k}** | {v} |" for k, v in meta]
    lines.append("")
    if rule.notes:
        lines += ["## Notes", "", rule.notes, ""]
    if examples := _rendered_examples(rule, checked):
        lines += ["## Example", ""]
        for intro, body in examples:
            lines += [intro, "", "```json", body, "```", ""]
    if rule.sources:
        lines += ["## Spec text", ""]
        for src in reversed(rule.sources):
            start, end = src.lines
            where = f"L{start}" if start == end else f"L{start}-L{end}"
            lines += [
                f"### {src.doc.version}",
                "",
                f"> {_unlink(' '.join(src.quote.split()))}",
                "",
                *(
                    ["*Stated without an RFC 2119 keyword; see Notes.*", ""]
                    if src.implied
                    else []
                ),
                *(["*Stated in the JSON schema.*", ""] if src.is_schema else []),
                f"[`{src.doc.repo}@{src.doc.ref}` · `{src.doc.path}` {where}]"
                f"({src.permalink})",
                "",
            ]
    return "\n".join(lines)


def render_index(checked: dict[str, list[str]]) -> str:
    """Markdown table of all rules."""
    tracked = ", ".join(TRACKED)
    lines = [
        "# Rule codes",
        "",
        "Each code identifies one normative statement in the "
        "[NGFF specification](https://ngff.openmicroscopy.org/), and links to the "
        "exact text in the spec that states it, for each spec version. yaozarrs "
        "errors and warnings for spec violations carry these codes once they are "
        "tagged: *Reported in* shows where yaozarrs reports each one.",
        "",
        "- `YZ-<AREA>-<NNN>` codes refer to statements in the NGFF spec or its JSON "
        "schemas.",
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
        REPORTED_NOTE,
        "",
        "## Spec rules",
        "",
        "| Code | Name | Level | Severity | Versions | Reported in | Summary |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in RULES.values():
        if r.is_spec_rule:
            lines.append(
                f"| [{r.code}]({r.code}.md) | `{r.slug}` | {r.level} | "
                f"{r.default_severity} | {_versions(r)} | "
                f"{', '.join(checked[r.code]) or '—'} | {r.summary} |"
            )
    lines += [
        "",
        "## yaozarrs checks",
        "",
        "| Code | Name | Severity | Reported in | Summary |",
        "|---|---|---|---|---|",
    ]
    for r in RULES.values():
        if not r.is_spec_rule:
            lines.append(
                f"| [{r.code}]({r.code}.md) | `{r.slug}` | {r.default_severity} | "
                f"{', '.join(checked[r.code]) or '—'} | {r.summary} |"
            )
    lines.append("")
    return "\n".join(lines)


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
