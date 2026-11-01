"""Shared fixtures for the CodSpeed benchmarks."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

# The benchmarks rely on the `benchmark` fixture from pytest-codspeed
# (`uv run --group bench ...`).  Skip them when it isn't installed, so that
# the regular test suite keeps working without it.
if importlib.util.find_spec("pytest_codspeed") is None:
    collect_ignore_glob = ["test_*.py"]

DATA = Path(__file__).parent.parent / "data"


def _array_meta(shape: list[int], dimension_names: list[str]) -> dict[str, Any]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape,
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shape}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "dimension_names": dimension_names,
    }


def _add_arrays(root: Path) -> None:
    """Add (metadata-only) zarr arrays for every multiscale dataset under `root`.

    The test data stores only contain the group metadata, so this makes them
    complete enough for structural validation.
    """
    for group_json in root.rglob("zarr.json"):
        doc = json.loads(group_json.read_text())
        ome = doc.get("attributes", {}).get("ome", {})
        for ms in ome.get("multiscales", []):
            # (v0.6 moved the axes into the coordinate systems)
            axes = ms.get("axes") or ms["coordinateSystems"][0]["axes"]
            names = [ax["name"] for ax in axes]
            types = [ax.get("type") for ax in axes]
            for level, ds in enumerate(ms["datasets"]):
                shape = [max(1, 256 >> level) if t == "space" else 2 for t in types]
                dest = group_json.parent / ds["path"] / "zarr.json"
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(json.dumps(_array_meta(shape, names)))


@pytest.fixture(scope="session")
def ome_store(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Path]:
    """Return a complete, valid local copy of a test data store (e.g. "v06/x.zarr").

    The array metadata missing from the test data is added (see `_add_arrays`).
    """
    dest = tmp_path_factory.mktemp("bench_stores")

    def _get(name: str) -> Path:
        out = dest / name
        if not out.exists():
            shutil.copytree(DATA / name, out)
            _add_arrays(out)
        return out

    return _get
