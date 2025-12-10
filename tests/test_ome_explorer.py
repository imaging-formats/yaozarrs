"""Test that OME explorer web component generates valid outputs.

These tests validate that the interactive OME-Zarr explorer
web component (docs/javascripts/ome_explorer.js) generates:
1. Valid JSON that validates against yaozarrs pydantic models
2. Valid Python code that executes without errors

The tests use Node.js to run the pure generation functions from
ome_generator.js and capture their outputs for validation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from yaozarrs import v04, v05, validate_ome_json

if TYPE_CHECKING:
    from collections.abc import Sequence

# Path to the Node.js test runner
RUNNER_PATH = Path(__file__).parent / "ome_explorer_runner.js"


def _dim(
    name,
    *,
    type_: str | None = None,
    unit=None,
    scale=None,
    translation=0.0,
    scaleFactor: float | None = None,
):
    if scale is None:
        scale = 0.5 if name in ["x", "y"] else (2.0 if name == "z" else 1.0)
    if type_ is None:
        if name in ["t", "time"]:
            type_ = "time"
        elif name in ["c", "channel"]:
            type_ = "channel"
        elif name in ["x", "y", "z"]:
            type_ = "space"
        else:
            type_ = ""
    if unit is None:
        if type_ == "time":
            unit = "second"
        elif type_ == "space":
            unit = "micrometer"
        else:
            unit = ""
    return {
        "name": name,
        "type": type_,
        "unit": unit,
        "scale": scale,
        "translation": translation,
        "scaleFactor": scaleFactor or (2.0 if type_ == "space" else 1.0),
    }


counter = count()


def _config(version: str, dims: Sequence[str | dict], levels: int):
    ndims = len(dims)
    if isinstance(dims, str):
        dims = [_dim(d) for d in dims]
    name = f"{next(counter)}_{version.replace('.', '')}_{ndims}d_{levels}levels"
    return {"name": name, "version": version, "numLevels": levels, "dimensions": dims}


TEST_CONFIGS = []
for versions in ["v0.4", "v0.5"]:
    for levels in [1, 3]:
        dims = ...
        name = f"{versions.replace('.', '')}_{levels}levels"
        TEST_CONFIGS.extend(
            [
                _config(versions, "yx", levels),
                _config(versions, "zyx", levels),
                _config(
                    versions,
                    [_dim("z"), _dim("y", translation=100), _dim("x", translation=200)],
                    levels,
                ),
                _config(versions, "czyx", levels),
                _config(versions, "tczyx", levels),
            ]
        )


def run_generator(config: dict) -> dict:
    """Call Node.js to generate JSON and Python code.

    Parameters
    ----------
    config : dict
        Configuration with dimensions, version, numLevels.

    Returns
    -------
    dict
        Dictionary with 'json' and 'python' keys containing generated strings.
    """
    # Strip the 'name' field as it's only for test identification
    config_for_runner = {k: v for k, v in config.items() if k != "name"}

    result = subprocess.run(
        ["node", str(RUNNER_PATH), json.dumps(config_for_runner)],
        capture_output=True,
        text=True,
        cwd=RUNNER_PATH.parent.parent,  # Run from repo root
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Node.js runner failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    return json.loads(result.stdout)


@pytest.fixture(params=TEST_CONFIGS, ids=lambda c: c.get("name", "unknown"))
def config(request: pytest.FixtureRequest) -> dict:
    """Parametrized fixture providing test configurations."""
    return request.param


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Node.js path handling differs on Windows",
)
def test_json_validates_against_pydantic(config: dict) -> None:
    """Validate generated JSON against yaozarrs pydantic models."""
    output = run_generator(config)

    # validate_ome_json auto-detects v04 vs v05
    result = validate_ome_json(output["json"])

    # Check it returned a valid model type
    assert result is not None
    if config["version"] == "v0.5":
        assert isinstance(result, v05.OMEZarrGroupJSON)
    else:
        # v04 output is just the .zattrs content (not wrapped in zarr.json)
        # which corresponds to Image type
        assert isinstance(result, v04.Image)

    namespace: dict[str, object] = {}
    try:
        exec(output["python"], namespace)
    except Exception as e:
        raise AssertionError(f"Python code failed to execute:\n\nError: {e}") from e

    # Verify expected objects were created
    assert isinstance(namespace["image"], (v04.Image, v05.Image))
    assert isinstance(namespace["multiscale"], (v04.Multiscale, v05.Multiscale))


def test_invalid():
    output = run_generator(_config(versions, "ztcyx", 1))

    with pytest.raises(ValidationError):
        exec(output["python"], {})

    with pytest.raises(ValidationError):
        validate_ome_json(output["json"])
