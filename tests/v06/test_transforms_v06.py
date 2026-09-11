import pytest
from pydantic import TypeAdapter, ValidationError

from yaozarrs import v06
from yaozarrs._validation_warning import ValidationWarning

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


def test_bare_string_io_rejected() -> None:
    # the schema requires the object form {"name": "in"}; a bare string is invalid
    with pytest.raises(ValidationError):
        TA.validate_python(
            {"type": "scale", "scale": [2, 2], "input": "in", "output": "out"}
        )


@pytest.mark.parametrize(
    "data, msg",
    [
        ({"type": "scale", "scale": [0.0, 1.0]}, "greater than 0"),
        ({"type": "affine"}, "exactly one"),
        ({"type": "affine", "affine": [[1]], "path": "x"}, "exactly one"),
        ({"type": "rotation"}, "exactly one"),
        ({"type": "mapAxis", "mapAxis": [0]}, "at least 2"),
        ({"type": "mapAxis", "mapAxis": [0, 9]}, "less than or equal to 4"),
        ({"type": "bogus"}, "tag"),
        ({"type": "projectAxis"}, "at least one"),
    ],
)
def test_invalid_transforms(data: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        TA.validate_python(data)


def test_interpolation_default_linear() -> None:
    t = v06.DisplacementsTransformation(path="d")
    assert t.interpolation == "linear"


def test_interpolation_unknown_value_warns_not_rejects() -> None:
    # spec: the interpolation method list is explicitly non-exhaustive/
    # non-normative (prose also mentions "bspline-cubic"), so an unrecognized
    # value is accepted with a warning, not rejected.
    with pytest.warns(ValidationWarning, match="Unrecognized interpolation"):
        t = v06.DisplacementsTransformation(path="p", interpolation="bspline-cubic")
    assert t.interpolation == "bspline-cubic"


def test_project_axis() -> None:
    t = v06.ProjectAxisTransformation(droppedInputs=[0])
    assert t.model_dump()["droppedInputs"] == [0]
    t2 = v06.ProjectAxisTransformation(createdOutputs=[1, 2])
    assert t2.model_dump()["createdOutputs"] == [1, 2]
    TA.validate_python({"type": "projectAxis", "droppedInputs": [0, 1]})
