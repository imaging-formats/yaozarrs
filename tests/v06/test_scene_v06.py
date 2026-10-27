from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from yaozarrs import v06

Y = {"name": "y", "type": "space"}
X = {"name": "x", "type": "space"}
WORLD = {"name": "world", "axes": [Y, X]}


def tile(i: int, offset: list[float] | None = None) -> dict:
    return {
        "type": "translation",
        "translation": offset or [0, 100 * i],
        "input": {"path": f"tile_{i}", "name": "physical"},
        "output": {"name": "world"},
    }


def scene(transforms: list[dict], systems: list[dict] | None = None) -> dict:
    sc: dict = {"coordinateTransformations": transforms}
    if systems is not None:
        sc["coordinateSystems"] = systems
    return {"version": "0.6", "scene": sc}


BIJECTION = {
    "type": "bijection",
    "forward": {"type": "displacements", "path": "fwd"},
    "inverse": {"type": "displacements", "path": "inv"},
    "input": {"path": "A", "name": "physical"},
    "output": {"path": "B", "name": "physical"},
}

VALID_SCENES = [
    # stitching: image subgroups -> a coordinate system of the scene itself
    scene([tile(0), tile(1)], [WORLD]),
    # registration between two image subgroups (no scene coordinate systems)
    scene([BIJECTION]),
    # between the scene's own coordinate systems
    scene(
        [
            tile(0),
            {
                "type": "scale",
                "scale": [2, 2],
                "input": {"name": "world"},
                "output": {"name": "other"},
            },
        ],
        [WORLD, {"name": "other", "axes": [Y, X]}],
    ),
    # explicit null path: refers to the scene's own coordinate systems (conflict N1)
    scene([{**tile(0), "output": {"name": "world", "path": None}}], [WORLD]),
]


@pytest.mark.parametrize("obj", VALID_SCENES)
def test_valid_scenes(obj: dict) -> None:
    s = v06.Scene.model_validate(obj)
    assert v06.Scene.model_validate_json(s.model_dump_json()) == s


INVALID_SCENES = [
    ({"version": "0.6", "scene": {}}, "Field required"),
    (scene([]), "at least 1 item"),
    ({"scene": scene([tile(0)], [WORLD])["scene"], "version": "0.5"}, "'0.6'"),
    (scene([{**tile(0), "input": None}], [WORLD]), "both an 'input' and an 'output'"),
    (
        scene([{**tile(0), "output": {"path": "tile_1"}}], [WORLD]),
        "must specify a 'name'",
    ),
    (scene([tile(0)]), "'world' has no 'path', so it must be one of"),
    (
        scene([{**tile(0), "output": "world"}], [WORLD]),
        "valid dictionary or instance of CoordinateSystemRef",
    ),
    # dimensionality against the scene's own coordinate systems
    (scene([tile(0, [1, 2, 3])], [WORLD]), "translation: has 3 values"),
    # connectivity
    (scene([tile(0)], [WORLD, {"name": "unused", "axes": [Y, X]}]), "'unused'"),
    (
        scene(
            [
                BIJECTION,
                {
                    **BIJECTION,
                    "input": {"path": "C", "name": "p"},
                    "output": {"path": "D", "name": "p"},
                },
            ]
        ),
        "not all linked",
    ),
]


@pytest.mark.parametrize("obj, msg", INVALID_SCENES)
def test_invalid_scenes(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(msg)):
        v06.Scene.model_validate(obj)


# ------------------------------------------------------------------------------
# upstream examples and conformance fixtures
# ------------------------------------------------------------------------------

SPEC = Path(__file__).parent.parent / "data" / "v06" / "spec"


def _scene_files() -> list[Path]:
    out = []
    for p in sorted(SPEC.rglob("*.json")):
        if p.name.startswith(".") or p.name == "SOURCE.json":
            continue
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:  # (examples with // comments; none are scenes)
            continue
        if isinstance(d, dict) and "scene" in d.get("attributes", d).get("ome", {}):
            out.append(p)
    return out


@pytest.mark.parametrize("path", _scene_files(), ids=lambda p: str(p.relative_to(SPEC)))
def test_upstream_scenes(path: Path) -> None:
    upstream_valid = "/invalid/" not in str(path)  # (examples are meant to be valid)
    d = json.loads(path.read_text())
    try:
        v06.Scene.model_validate(d.get("attributes", d)["ome"])
        valid = True
    except ValidationError:
        valid = False
    assert valid == upstream_valid
