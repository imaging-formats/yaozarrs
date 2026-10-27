from __future__ import annotations

import math
import re

import pytest
from pydantic import TypeAdapter, ValidationError

from yaozarrs import v06
from yaozarrs._validation_warning import ValidationWarning
from yaozarrs.v06._transforms import check_transform_ndim

TRANSFORM = TypeAdapter(v06.CoordinateTransformation)

c, s = math.cos(0.3), math.sin(0.3)
VALID_TRANSFORMS = [
    {"type": "identity"},
    {"type": "mapAxis", "mapAxis": [1, 0]},
    {"type": "projectAxis", "droppedInputs": [0]},
    {"type": "projectAxis", "createdOutputs": [0, 1], "droppedInputs": [2]},
    {"type": "translation", "translation": [1.5, -2]},
    {"type": "scale", "scale": [0.5, 2]},
    {"type": "scale", "scale": [-1, 1]},  # only "SHOULD be non-zero"
    {"type": "affine", "affine": [[1, 0, 3], [0, 1, 4]]},
    {"type": "affine", "path": "coordinateTransformations/affine"},
    {"type": "rotation", "rotation": [[c, -s], [s, c]]},
    # 30 degrees, rounded to 5 decimals
    {"type": "rotation", "rotation": [[0.86603, -0.5], [0.5, 0.86603]]},
    {"type": "rotation", "path": "coordinateTransformations/rot"},
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
                "transformation": {"type": "scale", "scale": [2]},
                "inputAxes": [0],
                "outputAxes": [0],
            },
            {
                "transformation": {"type": "translation", "translation": [3]},
                "inputAxes": [1],
                "outputAxes": [1],
            },
        ],
    },
    {
        "type": "bijection",
        "forward": {"type": "coordinates", "path": "fwd"},
        "inverse": {"type": "coordinates", "path": "inv"},
    },
    {"type": "displacements", "path": "field"},
    {"type": "coordinates", "path": "field", "interpolation": "bspline-cubic"},
    # "The interpolation methods listed in this specification ... are not exhaustive"
    {"type": "coordinates", "path": "field", "interpolation": "windowed-sinc"},
    # optional name, input and output
    {
        "type": "scale",
        "scale": [1, 1],
        "name": "s",
        "input": {"name": "a"},
        "output": {"name": "b", "path": "labels/x"},
    },
]


@pytest.mark.parametrize("obj", VALID_TRANSFORMS, ids=lambda o: o["type"])
def test_valid_transforms(obj: dict) -> None:
    t = TRANSFORM.validate_python(obj)
    assert t.type == obj["type"]
    dumped = TRANSFORM.dump_python(t, mode="json", exclude_unset=True)
    assert TRANSFORM.validate_python(dumped) == t


INVALID_TRANSFORMS = [
    ({"scale": [1, 1]}, "Unable to extract tag using discriminator 'type'"),
    ({"type": "shear", "shear": [1]}, "does not match any of the expected tags"),
    ({"type": "scale", "scale": []}, "at least 1 item"),
    ({"type": "scale", "path": "s"}, "Field required"),  # `scale` is required
    ({"type": "translation", "translation": []}, "at least 1 item"),
    ({"type": "mapAxis", "mapAxis": [0, 0]}, "must be a permutation of 0..1"),
    ({"type": "mapAxis", "mapAxis": [0, 5]}, "must be a permutation of 0..1"),
    ({"type": "mapAxis", "mapAxis": [-1, 0]}, "greater than or equal to 0"),
    ({"type": "projectAxis"}, "at least one of 'droppedInputs'"),
    ({"type": "projectAxis", "droppedInputs": [1, 1]}, "must be unique"),
    ({"type": "affine"}, "exactly one of 'affine'"),
    ({"type": "affine", "affine": [[1, 0, 0]], "path": "a"}, "exactly one of"),
    ({"type": "affine", "affine": [[1, 0, 0], [0, 1]]}, "same length"),
    ({"type": "rotation", "rotation": [[1, 0, 0], [0, 1, 0]]}, "square"),
    ({"type": "rotation", "rotation": []}, "at least 1 item"),
    ({"type": "affine", "affine": [[1]]}, "at least 2 items"),
    ({"type": "rotation", "rotation": [[2, 0], [0, 2]]}, "orthonormal rows"),
    ({"type": "rotation", "rotation": [[1, 0], [0, -1]]}, "determinant 1"),
    # "A non-empty array of transformations."
    ({"type": "sequence", "transformations": []}, "at least 1 item"),
    ({"type": "byDimension", "transformations": []}, "at least 1 item"),
    (
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {"type": "scale", "scale": [2]},
                    "inputAxes": [0.5],  # "arrays of integers"
                    "outputAxes": [0],
                }
            ],
        },
        "valid integer",
    ),
    (
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {"type": "scale", "scale": [2, 2]},
                    "inputAxes": [0],
                    "outputAxes": [0],
                }
            ],
        },
        "scale: has 2 values",
    ),
    (
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {"type": "scale", "scale": [2]},
                    "inputAxes": [0],
                    "outputAxes": [1],
                },
                {
                    "transformation": {"type": "scale", "scale": [2]},
                    "inputAxes": [1],
                    "outputAxes": [1],
                },
            ],
        },
        "output axes [1] appear in more than one",
    ),
    (
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {"type": "identity"},
                    "inputAxes": [0, 1],
                    "outputAxes": [1, 1],
                }
            ],
        },
        "transformations[0].outputAxes has duplicate axes",
    ),
    ({"type": "displacements"}, "Field required"),
    ({"type": "bijection", "forward": {"type": "identity"}}, "Field required"),
]


@pytest.mark.parametrize("obj, msg", INVALID_TRANSFORMS)
def test_invalid_transforms(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(msg)):
        TRANSFORM.validate_python(obj)


def test_integral_float_axes_coerce() -> None:
    t = TRANSFORM.validate_python(
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {"type": "identity"},
                    "inputAxes": [1.0],
                    "outputAxes": [0],
                }
            ],
        }
    )
    assert t.transformations[0].inputAxes == [1]


def test_zero_scale_warns() -> None:
    with pytest.warns(ValidationWarning, match="SHOULD be non-zero"):
        v06.ScaleTransformation(scale=[0, 1])


def test_bijection_of_invertible_types_warns() -> None:
    obj = {
        "type": "bijection",
        "forward": {"type": "scale", "scale": [2]},
        "inverse": {"type": "coordinates", "path": "inv"},
    }
    with pytest.warns(ValidationWarning, match="forward transformation") as record:
        TRANSFORM.validate_python(obj)
    assert len(record) == 1  # (validators of recursive models must run only once)


def test_type_survives_exclude_unset() -> None:
    t = v06.ScaleTransformation(scale=[1, 2])
    assert t.model_dump(exclude_unset=True) == {"type": "scale", "scale": [1, 2]}
    assert isinstance(
        TRANSFORM.validate_python(t.model_dump()), v06.ScaleTransformation
    )


S2 = {"type": "scale", "scale": [2, 3]}
NDIM_OK = [
    ({"type": "identity"}, 3, 3),
    (S2, 2, 2),
    ({"type": "mapAxis", "mapAxis": [2, 0, 1]}, 3, 3),
    ({"type": "projectAxis", "droppedInputs": [0]}, 3, 2),
    ({"type": "projectAxis", "createdOutputs": [0, 1]}, 2, 4),
    ({"type": "affine", "affine": [[1, 0, 0], [0, 1, 0], [1, 1, 0]]}, 2, 3),
    ({"type": "affine", "path": "a"}, 2, 3),  # shape checked in storage validation
    ({"type": "rotation", "rotation": [[0, -1], [1, 0]]}, 2, 2),
    (
        {
            "type": "sequence",
            "transformations": [S2, {"type": "affine", "affine": [[1, 1, 0]]}],
        },
        2,
        1,
    ),
    ({"type": "displacements", "path": "d"}, 2, 2),
    ({"type": "coordinates", "path": "c"}, 2, 3),
]


@pytest.mark.parametrize("obj, n_in, n_out", NDIM_OK)
def test_check_transform_ndim_ok(obj: dict, n_in: int, n_out: int) -> None:
    check_transform_ndim(TRANSFORM.validate_python(obj), n_in, n_out)


NDIM_BAD = [
    ({"type": "identity"}, 2, 3, "same number of axes"),
    (S2, 3, 3, "has 2 values"),
    ({"type": "mapAxis", "mapAxis": [1, 0]}, 3, 3, "has 2 indices"),
    ({"type": "projectAxis", "droppedInputs": [3]}, 3, 2, "out of range"),
    ({"type": "projectAxis", "droppedInputs": [0]}, 3, 3, "drops 1 and creates 0"),
    ({"type": "affine", "affine": [[1, 0, 0], [0, 1, 0]]}, 3, 2, "expected 2x4"),
    ({"type": "rotation", "rotation": [[0, -1], [1, 0]]}, 3, 3, "matrix is 2x2"),
    ({"type": "displacements", "path": "d"}, 2, 3, "same number of axes"),
    (
        {"type": "sequence", "transformations": [S2, {"type": "identity"}]},
        2,
        3,
        "sequence.transformations[1] -> identity: input and output must have",
    ),
    # a byDimension step inside a sequence is checked too
    (
        {
            "type": "sequence",
            "transformations": [
                {
                    "type": "byDimension",
                    "transformations": [
                        {
                            "transformation": {"type": "scale", "scale": [2]},
                            "inputAxes": [9],
                            "outputAxes": [0],
                        }
                    ],
                }
            ],
        },
        2,
        2,
        "byDimension: transformations[0].inputAxes [9] out of range",
    ),
    # after a step of unknown output size, checking resumes at the next step
    (
        {
            "type": "sequence",
            "transformations": [{"type": "coordinates", "path": "c"}, S2],
        },
        2,
        3,
        "sequence.transformations[1] -> scale: has 2 values",
    ),
    (
        {"type": "sequence", "transformations": [S2, S2]},
        3,
        3,
        "sequence.transformations[0] -> scale: has 2 values",
    ),
    (
        {
            "type": "byDimension",
            "transformations": [
                {"transformation": S2, "inputAxes": [0, 1], "outputAxes": [0, 1]}
            ],
        },
        2,
        3,
        "output axes [2] are not covered",
    ),
    (
        {
            "type": "bijection",
            "forward": {"type": "coordinates", "path": "f"},
            "inverse": {"type": "coordinates", "path": "i"},
        },
        2,
        3,
        "same number of axes",
    ),
]


@pytest.mark.parametrize("obj, n_in, n_out, msg", NDIM_BAD)
def test_check_transform_ndim_bad(obj: dict, n_in: int, n_out: int, msg: str) -> None:
    t = TRANSFORM.validate_python(obj)
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_transform_ndim(t, n_in, n_out)
