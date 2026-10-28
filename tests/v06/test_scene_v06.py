from __future__ import annotations

import re

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
    # `path` "MAY be omitted or null when referencing a coordinate system defined in
    # the scene's own `coordinateSystems`"
    scene([{**tile(0), "output": {"name": "world", "path": None}}], [WORLD]),
    # two coordinate systems of image A are linked within A, so this is connected
    scene(
        [BIJECTION, {**tile(0), "input": {"path": "A", "name": "other"}}],
        [WORLD],
    ),
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
    # ... also when only one side is known
    (
        scene(
            [
                {
                    "type": "affine",
                    "affine": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "input": {"path": "tile_0", "name": "physical"},
                    "output": {"name": "world"},
                }
            ],
            [WORLD],
        ),
        "matrix is 3x3, expected 2x3 (input has 2, output has 2 axes); the input size "
        "here is inferred",
    ),
    (
        scene(
            [
                {
                    "type": "projectAxis",
                    "droppedInputs": [3],
                    "input": {"name": "world"},
                    "output": {"path": "tile_0", "name": "physical"},
                }
            ],
            [WORLD],
        ),
        "droppedInputs [3] out of range (input has 2, output has 1 axes); the output "
        "size here is inferred",
    ),
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
