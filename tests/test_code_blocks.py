"""Run the Python code blocks in the README and the docs."""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
FILES = [
    path
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    if "```python" in path.read_text(encoding="utf-8")
]
# a python block, possibly indented (in a tab or admonition), and maybe marked with
# `<!-- no-test -->` on the line before it (for schematic examples that can't run)
BLOCK = re.compile(
    r"^(?P<skip>[ \t]*<!-- no-test -->\s*\n)?"
    r"(?P<indent>[ \t]*)```python\n(?P<code>.*?)^(?P=indent)```",
    re.MULTILINE | re.DOTALL,
)


@pytest.mark.skipif(
    sys.version_info < (3, 11) or sys.platform == "win32", reason="requires python3.11+"
)
@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_code_blocks(
    path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run each file's blocks in order, sharing one namespace (like a reader would)."""
    pytest.importorskip("tensorstore")
    pytest.importorskip("zarr")
    pytest.importorskip("fsspec")
    # the examples write stores to relative paths (e.g. "example.ome.zarr")
    monkeypatch.chdir(tmp_path)

    text = path.read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    for block in BLOCK.finditer(text):
        if not block["skip"]:
            # pad with newlines, so that tracebacks point at the line in the file
            lines_before = text.count("\n", 0, block.start("code"))
            code = "\n" * lines_before + textwrap.dedent(block["code"])
            exec(compile(code, str(path), "exec"), namespace)
