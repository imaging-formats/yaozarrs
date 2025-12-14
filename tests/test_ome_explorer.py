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

if sys.platform == "win32":
    pytest.skip("Node.js path handling differs on Windows", allow_module_level=True)

# Path to the Node.js test runner
RUNNER_PATH = Path(__file__).parent / "ome_explorer_runner.mjs"


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


def _plate_config(
    version: str,
    dims: Sequence[str | dict],
    levels: int,
    plate_type: str,
    selected_wells: list[dict[str, int]],
    num_fovs: int,
):
    """Helper to create a plate configuration."""
    ndims = len(dims)
    if isinstance(dims, str):
        dims = [_dim(d) for d in dims]
    name = (
        f"{next(counter)}_{version.replace('.', '')}_{ndims}d_{levels}levels_"
        f"{plate_type}_{len(selected_wells)}wells_{num_fovs}fovs"
    )
    return {
        "name": name,
        "version": version,
        "numLevels": levels,
        "dimensions": dims,
        "isPlate": True,
        "plateType": plate_type,
        "selectedWells": selected_wells,
        "numFOVs": num_fovs,
    }


PLATE_TEST_CONFIGS = []
for version in ["v0.4", "v0.5"]:
    for levels in [1, 2]:
        # Test different plate types
        for plate_type in ["12-well", "24-well", "96-well"]:
            # Single well, single FOV (simplest case)
            PLATE_TEST_CONFIGS.append(
                _plate_config(
                    version, "yx", levels, plate_type, [{"row": 0, "col": 0}], 1
                )
            )
            # Multiple wells, single FOV
            PLATE_TEST_CONFIGS.append(
                _plate_config(
                    version,
                    "czyx",
                    levels,
                    plate_type,
                    [{"row": 0, "col": 0}, {"row": 0, "col": 1}, {"row": 1, "col": 0}],
                    1,
                )
            )
            # Single well, multiple FOVs
            PLATE_TEST_CONFIGS.append(
                _plate_config(
                    version, "zyx", levels, plate_type, [{"row": 0, "col": 0}], 3
                )
            )

        # Test one complex case: 24-well, multiple wells, multiple FOVs, 5D
        PLATE_TEST_CONFIGS.append(
            _plate_config(
                version,
                "tczyx",
                levels,
                "24-well",
                [
                    {"row": 0, "col": 0},
                    {"row": 0, "col": 1},
                    {"row": 1, "col": 0},
                    {"row": 1, "col": 1},
                ],
                2,
            )
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


def test_json_validates_against_pydantic(config: dict) -> None:
    """Validate generated JSON against yaozarrs pydantic models."""
    output = run_generator(config)
    ngff_version = config["version"]

    # validate_ome_json auto-detects v04 vs v05
    result = validate_ome_json(output["json"])

    # Check it returned a valid model type
    assert result is not None
    if ngff_version == "v0.5":
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
    assert isinstance(namespace["multiscale1"], (v04.Multiscale, v05.Multiscale))
    assert isinstance(namespace["multiscale2"], (v04.Multiscale, v05.Multiscale))
    assert namespace["multiscale1"] == namespace["multiscale2"]

    # test the python code generates the same json as the js code
    json_dict = json.loads(output["json"])
    python_dict = namespace["image"].model_dump(exclude_unset=True, exclude_none=True)
    if ngff_version == "v0.5":
        json_dict = json_dict["attributes"]["ome"]
        assert json_dict.pop("version") == "0.5"
    assert json_dict == python_dict


def test_invalid():
    output = run_generator(_config(versions, "ztcyx", 1))

    with pytest.raises(ValidationError):
        exec(output["python"], {})

    with pytest.raises(ValidationError):
        validate_ome_json(output["json"])


@pytest.mark.parametrize(
    "plate_config",
    PLATE_TEST_CONFIGS,
    ids=lambda c: c.get("name", "unknown"),
)
def test_plate_validates_against_pydantic(plate_config: dict) -> None:
    """Validate generated plate JSON against yaozarrs pydantic models."""
    output = run_generator(plate_config)
    ngff_version = plate_config["version"]

    # Validate plate metadata
    plate_result = validate_ome_json(output["plateJson"])
    assert plate_result is not None
    if ngff_version == "v0.5":
        assert isinstance(plate_result, v05.OMEZarrGroupJSON)
        # Extract the plate metadata from attributes
        assert hasattr(plate_result.attributes, "ome")
        assert hasattr(plate_result.attributes.ome, "plate")
    else:
        # v0.4 output is just the .zattrs content
        assert isinstance(plate_result, v04.Plate)

    # Validate well metadata
    well_result = validate_ome_json(output["wellJson"])
    assert well_result is not None
    if ngff_version == "v0.5":
        assert isinstance(well_result, v05.OMEZarrGroupJSON)
        assert hasattr(well_result.attributes, "ome")
        assert hasattr(well_result.attributes.ome, "well")
    else:
        assert isinstance(well_result, v04.Well)

    # Execute Python code and verify it creates valid objects
    namespace: dict[str, object] = {}
    try:
        exec(output["python"], namespace)
    except Exception as e:
        raise AssertionError(f"Python code failed to execute:\n\nError: {e}") from e

    plate_def = namespace["plate_def"]
    well_def = namespace["well_def"]

    # Verify expected objects were created
    # Both versions create plate_def and well_def
    assert isinstance(plate_def, (v04.PlateDef, v05.PlateDef))
    assert isinstance(well_def, (v04.WellDef, v05.WellDef))
    assert isinstance(namespace["image"], (v04.Image, v05.Image))
    assert isinstance(namespace["multiscale"], (v04.Multiscale, v05.Multiscale))

    # v0.5 also wraps them in Plate and Well
    if ngff_version == "v0.5":
        assert isinstance(namespace["plate"], v05.Plate)
        assert isinstance(namespace["well"], v05.Well)

    # Verify the Python-generated plate_def matches the JS-generated plate
    plate_json_dict = json.loads(output["plateJson"])
    well_json_dict = json.loads(output["wellJson"])
    plate_python_dict = plate_def.model_dump(exclude_unset=True)
    well_python_dict = well_def.model_dump(exclude_unset=True)

    if ngff_version == "v0.5":
        plate_json_dict = plate_json_dict["attributes"]["ome"]["plate"]
        well_json_dict = well_json_dict["attributes"]["ome"]["well"]
    else:
        plate_json_dict = plate_json_dict["plate"]
        well_json_dict = well_json_dict["well"]

    assert plate_json_dict == plate_python_dict
    assert well_json_dict == well_python_dict
