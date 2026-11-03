"""Fixtures for the doctests in `src/` (collected with `--doctest-modules`)."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

collect_ignore = [
    # a test helper that needs `ome-zarr`, and has no doctests
    "src/yaozarrs/_demo_data.py",
    # scripts for `just test-conformance`, which run when imported
    "tests/conformance",
]

SRC = Path(__file__).parent / "src"
WRITE = SRC / "yaozarrs" / "write"


@pytest.fixture(autouse=True)
def _doctest_env(request: pytest.FixtureRequest) -> None:
    if not request.path.is_relative_to(SRC):
        return
    # the examples write stores to relative paths (e.g. "example.ome.zarr")
    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.chdir(request.getfixturevalue("tmp_path"))
    if request.path.is_relative_to(WRITE):
        pytest.importorskip("numpy")
        if not (find_spec("zarr") or find_spec("tensorstore")):
            pytest.skip("the write examples need zarr or tensorstore")
