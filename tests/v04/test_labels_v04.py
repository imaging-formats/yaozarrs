import pytest
from pydantic import ValidationError

from yaozarrs import v04

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

V04_VALID_LABELS = [
    # Simple label with just version
    {
        "version": "0.4",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "version": "0.4",
        },
    },
    # Label with colors
    {
        "version": "0.4",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "colors": [
                {"label-value": 1, "rgba": [255, 0, 0, 255]},
                {"label-value": 2, "rgba": [0, 255, 0, 255]},
            ]
        },
    },
    # Label with properties
    {
        "version": "0.4",
        "multiscales": MULTISCALES_2D,
        "image-label": {"properties": [{"label-value": 1}, {"label-value": 2}]},
    },
    # Label with source
    {
        "version": "0.4",
        "multiscales": MULTISCALES_2D,
        "image-label": {"source": {"image": "../image.zarr"}},
    },
    # Label with all fields
    {
        "version": "0.4",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "version": "0.4",
            "colors": [{"label-value": 1, "rgba": [255, 0, 0, 255]}],
            "properties": [{"label-value": 1}],
            "source": {"image": "../image.zarr"},
        },
    },
    # Label with colors without RGBA (valid since rgba is optional)
    {
        "version": "0.4",
        "multiscales": MULTISCALES_2D,
        "image-label": {
            "colors": [{"label-value": 1.5}, {"label-value": 2.0}],
        },
    },
]


V04_INVALID_LABELS = [
    # Missing image-label
    {},
    # Empty image-label is actually valid per schema
    # {"image-label": {}},
    # Invalid RGBA values (out of range)
    {
        "image-label": {
            "colors": [
                {
                    "label-value": 1,
                    "rgba": [256, 0, 0, 255],  # 256 is out of range
                }
            ]
        }
    },
    # Invalid RGBA array length
    {
        "image-label": {
            "colors": [
                {
                    "label-value": 1,
                    "rgba": [255, 0, 0],  # Should be 4 elements
                }
            ]
        }
    },
    # Empty colors array (should have minItems: 1)
    {"image-label": {"colors": []}},
    # Empty properties array (should have minItems: 1)
    {"image-label": {"properties": []}},
    # Missing required label-value in colors
    {
        "image-label": {
            "colors": [
                {"rgba": [255, 0, 0, 255]}  # Missing label-value
            ]
        }
    },
    # Missing required label-value in properties
    {
        "image-label": {
            "properties": [
                {}  # Missing label-value
            ]
        }
    },
]


@pytest.mark.parametrize("data", V04_VALID_LABELS)
def test_valid_v04_labels(data: dict) -> None:
    """Test that valid v04 label metadata can be parsed."""
    label = v04.Image.model_validate(data)
    assert label.image_label is not None


# list of (invalid_obj, error_msg_substring) - combining all invalid cases
V04_INVALID_LABELS_EXTENDED: list[tuple[dict, str]] = [
    # RGBA values out of range (256 > 255)
    (
        {
            "version": "0.4",
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [256, 0, 0, 255]},
                ],
            },
        },
        "Input should be less than or equal to 255",
    ),
    # RGBA values out of range (-1 < 0)
    (
        {
            "version": "0.4",
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [-1, 0, 0, 255]},
                ],
            },
        },
        "Input should be greater than or equal to 0",
    ),
    # Invalid RGBA array length (too short)
    (
        {
            "version": "0.4",
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [255, 0, 0]},
                ],
            },
        },
        "at least 4 items",
    ),
    # Invalid RGBA array length (too long)
    (
        {
            "version": "0.4",
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [255, 0, 0, 255, 128]},
                ],
            },
        },
        "at most 4 items",
    ),
    # Duplicate colors (UniqueList violation)
    (
        {
            "version": "0.4",
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "colors": [
                    {"label-value": 1, "rgba": [255, 0, 0, 255]},
                    {"label-value": 1, "rgba": [255, 0, 0, 255]},
                ],
            },
        },
        "List items are not unique",
    ),
    # Duplicate properties (UniqueList violation)
    (
        {
            "version": "0.4",
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "properties": [
                    {"label-value": 1},
                    {"label-value": 1},
                ],
            },
        },
        "List items are not unique",
    ),
]

# Convert V04_INVALID_LABELS to same format as v05, with better error patterns
V04_INVALID_LABELS_WITH_MESSAGES: list[tuple[dict, str]] = [
    # Missing multiscales (required for Image)
    ({}, "Field required"),
    # Invalid RGBA values (out of range)
    (
        {
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "colors": [
                    {
                        "label-value": 1,
                        "rgba": [256, 0, 0, 255],  # 256 is out of range
                    }
                ]
            },
        },
        "Input should be less than or equal to 255",
    ),
    # Invalid RGBA array length
    (
        {
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "colors": [
                    {
                        "label-value": 1,
                        "rgba": [255, 0, 0],  # Should be 4 elements
                    }
                ]
            },
        },
        "at least 4 items",
    ),
    # Empty colors array (should have minItems: 1)
    ({"multiscales": MULTISCALES_2D, "image-label": {"colors": []}}, "at least 1 item"),
    # Empty properties array (should have minItems: 1)
    (
        {"multiscales": MULTISCALES_2D, "image-label": {"properties": []}},
        "at least 1 item",
    ),
    # Missing required label-value in colors
    (
        {
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "colors": [
                    {"rgba": [255, 0, 0, 255]}  # Missing label-value
                ]
            },
        },
        "Field required",
    ),
    # Missing required label-value in properties
    (
        {
            "multiscales": MULTISCALES_2D,
            "image-label": {
                "properties": [
                    {}  # Missing label-value
                ]
            },
        },
        "Field required",
    ),
]

# Combine all invalid cases with specific error messages
V04_ALL_INVALID_LABELS: list[tuple[dict, str]] = (
    V04_INVALID_LABELS_WITH_MESSAGES + V04_INVALID_LABELS_EXTENDED
)


@pytest.mark.parametrize("obj, msg", V04_ALL_INVALID_LABELS)
def test_invalid_v04_labels(obj: dict, msg: str) -> None:
    """Test that invalid v04 label metadata raises validation errors."""
    with pytest.raises(ValidationError, match=msg):
        v04.Image.model_validate(obj)
