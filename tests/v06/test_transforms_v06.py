import pytest
from pydantic import TypeAdapter, ValidationError

from yaozarrs import v06

TA = TypeAdapter(v06.Transformation)

VALID_TRANSFORMS = [
    {"type": "identity"},
    {"type": "scale", "scale": [1.0, 2.0]},
    {"type": "translation", "translation": [0.0, -5.0]},
    {"type": "mapAxis", "mapAxis": [1, 0]},
    {"type": "affine", "affine": [[1, 0, 0], [0, 1, 0]]},
    {"type": "affine", "path": "matrix"},
    {"type": "rotation", "rotation": [[0, -1], [1, 0]]},
    {"type": "rotation", "path": "rot"},
    {
        "type": "bijection",
        "forward": {"type": "identity"},
        "inverse": {"type": "identity"},
    },
    {
        "type": "sequence",
        "transformations": [
            {"type": "scale", "scale": [2, 2]},
            {"type": "translation", "translation": [1, 1]},
        ],
    },
    {
        "type": "byDimension",
        "transformations": [
            {
                "transformation": {"type": "scale", "scale": [2.0]},
                "input_axes": [0],
                "output_axes": [0],
            }
        ],
    },
    {"type": "displacements", "path": "disp"},
    {"type": "displacements", "path": "disp", "interpolation": "cubic"},
    {"type": "coordinates", "path": "coords"},
]


@pytest.mark.parametrize("data", VALID_TRANSFORMS, ids=lambda d: d["type"])
def test_valid_transforms(data: dict) -> None:
    t = TA.validate_python(data)
    assert t.type == data["type"]


def test_type_survives_exclude_unset_roundtrip() -> None:
    # discriminated union needs `type` -- it must survive exclude_unset dumps
    t = v06.ScaleTransformation(scale=[1.0, 1.0])
    dumped = t.model_dump_json(exclude_unset=True)
    assert '"type":"scale"' in dumped
    TA.validate_json(dumped)


def test_bare_string_io_shorthand() -> None:
    # spec examples use "input": "in" shorthand; we coerce to {"name": "in"}
    t = TA.validate_python(
        {"type": "scale", "scale": [2, 2], "input": "in", "output": "out"}
    )
    assert t.input.name == "in"
    assert t.output.name == "out"


@pytest.mark.parametrize(
    "data, msg",
    [
        ({"type": "scale", "scale": [0.0, 1.0]}, "greater than 0"),
        ({"type": "affine"}, "exactly one"),
        ({"type": "affine", "affine": [[1]], "path": "x"}, "exactly one"),
        ({"type": "rotation"}, "exactly one"),
        ({"type": "mapAxis", "mapAxis": [0]}, "at least 2"),
        ({"type": "mapAxis", "mapAxis": [0, 9]}, "less than or equal to 4"),
        (
            {"type": "displacements", "path": "p", "interpolation": "bspline"},
            "interpolation",
        ),
        ({"type": "bogus"}, "tag"),
    ],
)
def test_invalid_transforms(data: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        TA.validate_python(data)


def test_interpolation_default_linear() -> None:
    t = v06.DisplacementsTransformation(path="d")
    assert t.interpolation == "linear"
