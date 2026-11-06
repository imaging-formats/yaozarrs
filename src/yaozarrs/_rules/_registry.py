"""Load the rule registry from `rules.toml`."""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Iterator, Mapping
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

from yaozarrs._rules._model import Rule, SpecDoc, SpecSource

REGISTRY_FILE = Path(__file__).parent / "rules.toml"
_T = TypeVar("_T")


def _load_rule(code: str, entry: dict[str, Any], docs: dict[str, SpecDoc]) -> Rule:
    order = list(docs)
    sources = tuple(
        sorted(
            (
                SpecSource(
                    doc=docs[version].schema(src["schema"])
                    if "schema" in src
                    else docs[version],
                    lines=(start, end),
                    quote=src["quote"],
                    implied=src.get("implied", False),
                )
                for src in entry.pop("sources", ())
                for version, (start, end) in src["lines"].items()
            ),
            key=lambda s: order.index(s.doc.version),
        )
    )
    for key in ("supersedes", "aliases"):
        if key in entry:
            entry[key] = tuple(entry[key])
    return Rule(code=code, sources=sources, **entry)


@cache
def _load() -> tuple[dict[str, SpecDoc], dict[str, Rule]]:
    data = tomllib.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    docs = {v: SpecDoc(version=v, **d) for v, d in data["spec"].items()}
    rules = {code: _load_rule(code, e, docs) for code, e in data["rules"].items()}
    return docs, rules


class _Lazy(Mapping[str, _T]):
    """A read-only view of part of `rules.toml`, which is parsed on first use."""

    def __init__(self, part: Callable[[], dict[str, _T]]) -> None:
        self._part = part

    def __getitem__(self, key: str) -> _T:
        return self._part()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._part())

    def __len__(self) -> int:
        return len(self._part())

    def __repr__(self) -> str:
        return f"<{len(self)} entries from {REGISTRY_FILE.name}>"


# (lazy, so that importing yaozarrs doesn't parse the registry)
SPEC_DOCS: Mapping[str, SpecDoc] = _Lazy(lambda: _load()[0])
"""The pinned spec file of each tracked version, in version order."""
RULES: Mapping[str, Rule] = _Lazy(lambda: _load()[1])
"""Every rule, by code."""
