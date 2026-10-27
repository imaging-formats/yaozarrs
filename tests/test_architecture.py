"""Enforce import boundaries between spec versions and shared code.

- A module belonging to a spec version (any module with a `vNN` component in its
  dotted path, e.g. `yaozarrs.v05._image` or `yaozarrs.write.v05._write`) must
  never directly import a module belonging to a *different* version.
  Older versions must not change behavior as a side effect of work on newer ones.
- `yaozarrs._common` holds building blocks shared across versions.  It must not
  import any version module (or `yaozarrs.write`), and must not contain version
  string literals (i.e. it may not branch on version: "fork, don't flag").
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
PY_FILES = sorted((SRC / "yaozarrs").rglob("*.py"))
VERSION_PART = re.compile(r"v\d\d")
VERSION_LITERAL = re.compile(r"0\.\d+(\.\d+)?((rc|\.dev)\d+)?")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


COMMON_FILES = [p for p in PY_FILES if _module_name(p).startswith("yaozarrs._common")]


def _version_of(module: str) -> str | None:
    return next((p for p in module.split(".") if VERSION_PART.fullmatch(p)), None)


def _imported_modules(path: Path) -> list[str]:
    """Return absolute names of all modules imported by the file at `path`."""
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0]
                name = f"{base}.{node.module}" if node.module else base
            else:
                name = node.module or ""
            out.append(name)
            # `from pkg import submodule` imports the submodule too
            out.extend(f"{name}.{alias.name}" for alias in node.names)
    return [m for m in out if m.startswith("yaozarrs")]


@pytest.mark.parametrize("path", PY_FILES, ids=[_module_name(p) for p in PY_FILES])
def test_import_boundaries(path: Path) -> None:
    module = _module_name(path)
    imported = _imported_modules(path)

    if (version := _version_of(module)) is not None:
        bad = [m for m in imported if _version_of(m) not in (None, version)]
        assert not bad, f"{module} ({version}) imports other versions: {bad}"

    if module.startswith("yaozarrs._common"):
        bad = [m for m in imported if _version_of(m) or m.startswith("yaozarrs.write")]
        assert not bad, f"{module} imports version-specific modules: {bad}"


@pytest.mark.parametrize(
    "path", COMMON_FILES, ids=[_module_name(p) for p in COMMON_FILES]
)
def test_common_has_no_version_literals(path: Path) -> None:
    literals = [
        node.value
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and VERSION_LITERAL.fullmatch(node.value)
    ]
    assert not literals, f"{_module_name(path)} contains version literals: {literals}"
