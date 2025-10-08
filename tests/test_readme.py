"""Test that all Python code blocks in README.md execute successfully."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"


def test_readme_python_blocks(tmp_path: Path) -> None:
    """Extract and execute all Python code blocks from README.md."""
    pytest.importorskip("tensorstore")
    pytest.importorskip("zarr")
    pytest.importorskip("fsspec")

    readme_content = README.read_text()

    python_blocks = re.findall(r"```python\n(.*?)```", readme_content, re.DOTALL)
    namespace: dict[str, object] = {}
    for i, block in enumerate(python_blocks):
        try:
            # replace "zarr.json" with tmp_path / "zarr.json"
            block = block.replace("zarr.json", str(tmp_path / "zarr.json"))
            exec(block, namespace)
        except Exception as e:
            raise AssertionError(
                f"Python block {i + 1} failed to execute:\n{block}\n\nError: {e}"
            ) from e
