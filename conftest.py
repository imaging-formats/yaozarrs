"""Fixtures for the doctests in `src/` (collected with `--doctest-modules`)."""

from __future__ import annotations

import urllib.error
import urllib.request
from functools import cache
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
# doctests that read a public store over http
NETWORK = {"yaozarrs._zarr.ZarrGroup.tree": "https://uk1s3.embassy.ebi.ac.uk/idr/"}


@cache
def _reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=5).close()
    except OSError as e:
        # any http response (even an error) means the server is up
        return isinstance(e, urllib.error.HTTPError)
    return True


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
    if url := NETWORK.get(request.node.name):
        pytest.importorskip("fsspec")
        pytest.importorskip("aiohttp")
        if not _reachable(url):
            pytest.skip(f"{url} is not reachable")
