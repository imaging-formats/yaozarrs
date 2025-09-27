import pytest
from pydantic import ValidationError

from yaozarrs import v05, validate_ome_object

# Helper data
COLOR_1 = {"label-value": 1, "rgba": [255, 0, 0, 255]}  # red
COLOR_2 = {"label-value": 2, "rgba": [0, 255, 0, 255]}  # green
COLOR_3 = {"label-value": 3.5}  # no color specified

PROPERTY_1 = {"label-value": 1}
PROPERTY_2 = {"label-value": 2}

SOURCE_1 = {"image": "../../"}

X_AXIS = {"name": "x", "type": "space", "unit": "millimeter"}
Y_AXIS = {"name": "y", "type": "space", "unit": None}
SCALE_TFORM_2D = {"type": "scale", "scale": [1, 1]}
MULTISCALES_2D = [
    {
        "name": "image",
        "axes": [X_AXIS, Y_AXIS],
        "datasets": [
            {"path": "0", "coordinateTransformations": [SCALE_TFORM_2D]},
            {"path": "1", "coordinateTransformations": [SCALE_TFORM_2D]},
        ],
    }
]

V05_VALID_LABEL_IMAGES = [
    # Minimal label (no colors, properties, or source)
    {
        "version": "0.5",
        "multiscales": MULTISCALES_2D,
        "image-label": {},
    },
    # Label with colors only
    {
        "version": "0.5",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "colors": [COLOR_1, COLOR_2],
        },
    },
    # Label with colors (no rgba for some)
    {
        "version": "0.5",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "colors": [COLOR_1, COLOR_3],
        },
    },
    # Label with properties only
    {
        "version": "0.5",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "properties": [PROPERTY_1, PROPERTY_2],
        },
    },
    # Label with source only
    {
        "version": "0.5",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "source": SOURCE_1,
        },
    },
    # Label with all fields
    {
        "version": "0.5",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "colors": [COLOR_1, COLOR_2],
            "properties": [PROPERTY_1, PROPERTY_2],
            "source": SOURCE_1,
        },
    },
    # Label with empty source
    {
        "version": "0.5",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "source": {},
        },
    },
    # Edge case: RGBA with 0 and 255 values
    {
        "version": "0.5",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "colors": [
                {"label-value": 0, "rgba": [0, 0, 0, 0]},  # black, transparent
                {"label-value": 255, "rgba": [255, 255, 255, 255]},  # white, opaque
            ],
        },
    },
]

V05_VALID_LABELS_GROUPS = [
    # Simple labels group
    {
        "labels": ["cell_segmentation"],
    },
    # Multiple labels
    {
        "labels": ["cell_segmentation", "nucleus_segmentation", "boundary"],
    },
]


@pytest.mark.parametrize("obj", V05_VALID_LABEL_IMAGES)
def test_valid_v05_labels(obj: dict) -> None:
    validate_ome_object(obj, v05.LabelImage)


@pytest.mark.parametrize("obj", V05_VALID_LABELS_GROUPS)
def test_valid_v05_labels_groups(obj: dict) -> None:
    validate_ome_object(obj, v05.LabelsGroup)


# list of (invalid_obj, error_msg_substring)
V05_INVALID_LABELS: list[tuple[dict, str]] = [
    # Invalid RGBA values (outside 0-255 range)
    (
        {
            "version": "0.5",
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [256, 0, 0, 255]},  # 256 > 255
                ],
            },
        },
        "Input should be less than or equal to 255",
    ),
    (
        {
            "version": "0.5",
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [-1, 0, 0, 255]},  # -1 < 0
                ],
            },
        },
        "Input should be greater than or equal to 0",
    ),
    # Invalid RGBA array length (not exactly 4)
    (
        {
            "version": "0.5",
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [255, 0, 0]},  # only 3 values
                ],
            },
        },
        "at least 4 items",
    ),
    (
        {
            "version": "0.5",
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [255, 0, 0, 255, 128]},  # 5 values
                ],
            },
        },
        "at most 4 items",
    ),
    # Empty colors array (MinLen violation)
    (
        {
            "version": "0.5",
            "image-label": {
                "colors": [],  # empty
            },
        },
        "at least 1 item",
    ),
    # Empty properties array (MinLen violation)
    (
        {
            "version": "0.5",
            "image-label": {
                "properties": [],  # empty
            },
        },
        "at least 1 item",
    ),
    # Duplicate colors (UniqueList violation)
    (
        {
            "version": "0.5",
            "image-label": {
                "colors": [COLOR_1, COLOR_1],  # duplicate
            },
        },
        "List items are not unique",
    ),
    # Duplicate properties (UniqueList violation)
    (
        {
            "version": "0.5",
            "image-label": {
                "properties": [PROPERTY_1, PROPERTY_1],  # duplicate
            },
        },
        "List items are not unique",
    ),
    # Missing required fields in colors
    (
        {
            "version": "0.5",
            "image-label": {
                "colors": [
                    {"rgba": [255, 0, 0, 255]},  # missing label-value
                ],
            },
        },
        "Field required",
    ),
    # Missing required fields in properties
    (
        {
            "version": "0.5",
            "image-label": {
                "properties": [
                    {},  # missing label-value
                ],
            },
        },
        "Field required",
    ),
    # Wrong type for label-value in properties (should be int)
    (
        {
            "version": "0.5",
            "image-label": {
                "properties": [
                    {"label-value": 1.5},  # float not allowed for properties
                ],
            },
        },
        "Input should be a valid integer",
    ),
    # Wrong type for RGBA values (should be int)
    (
        {
            "version": "0.5",
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [255.5, 0, 0, 255]},  # float not int
                ],
            },
        },
        "Input should be a valid integer",
    ),
]


@pytest.mark.parametrize("obj, msg", V05_INVALID_LABELS)
def test_invalid_v05_labels(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        v05.LabelImage.model_validate(obj)


V05_INVALID_LABELS_GROUPS: list[tuple[dict, str]] = [
    # Empty labels array
    (
        {
            "labels": [],
        },
        "at least 1 item",
    ),
    # Missing labels field
    (
        {},
        "Field required",
    ),
]


@pytest.mark.parametrize("obj, msg", V05_INVALID_LABELS_GROUPS)
def test_invalid_v05_labels_groups(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        v05.LabelsGroup.model_validate(obj)
