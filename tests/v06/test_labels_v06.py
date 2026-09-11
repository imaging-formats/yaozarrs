import pytest
from pydantic import ValidationError

from yaozarrs import v06, validate_ome_object

COLOR_1 = {"label-value": 1, "rgba": [255, 0, 0, 255]}
COLOR_2 = {"label-value": 2, "rgba": [0, 255, 0, 255]}
COLOR_3 = {"label-value": 3.5}

PROPERTY_1 = {"label-value": 1}
PROPERTY_2 = {"label-value": 2}
SOURCE_1 = {"image": "../../"}

X_AXIS = {"name": "x", "type": "space", "unit": "millimeter"}
Y_AXIS = {"name": "y", "type": "space", "unit": None}


def _scale_ds(path: str) -> dict:
    return {
        "path": path,
        "coordinateTransformations": [
            {
                "type": "scale",
                "scale": [1, 1],
                "input": {"path": path},
                "output": {"name": "cs"},
            }
        ],
    }


MULTISCALES_2D = [
    {
        "name": "image",
        "coordinateSystems": [{"name": "cs", "axes": [X_AXIS, Y_AXIS]}],
        "datasets": [_scale_ds("0"), _scale_ds("1")],
    }
]


def _label(image_label: dict) -> dict:
    return {
        "version": "0.6.dev4",
        "multiscales": MULTISCALES_2D,
        "image-label": image_label,
    }


V06_VALID_LABEL_IMAGES = [
    _label({}),  # image-label is required but may be empty
    _label({"colors": [COLOR_1, COLOR_2]}),
    _label({"colors": [COLOR_1, COLOR_3]}),
    _label({"properties": [PROPERTY_1, PROPERTY_2]}),
    _label({"source": SOURCE_1}),
    _label(
        {"colors": [COLOR_1, COLOR_2], "properties": [PROPERTY_1], "source": SOURCE_1}
    ),
    _label({"source": {}}),
    _label(
        {
            "colors": [
                {"label-value": 0, "rgba": [0, 0, 0, 0]},
                {"label-value": 255, "rgba": [255, 255, 255, 255]},
            ]
        }
    ),
]


@pytest.mark.parametrize("obj", V06_VALID_LABEL_IMAGES)
def test_valid_v06_labels(obj: dict) -> None:
    validate_ome_object(obj, v06.LabelImage)


def test_image_label_required_on_label_image() -> None:
    # a label image MUST carry image-label in v0.6
    with pytest.raises(ValidationError, match=r"image.label|Field required"):
        v06.LabelImage.model_validate(
            {"version": "0.6.dev4", "multiscales": MULTISCALES_2D}
        )


def test_image_label_has_no_version_field() -> None:
    # v0.6 moved the version out of image-label up to the ome level
    assert "version" not in v06.ImageLabel.model_fields


V06_VALID_LABELS_GROUPS = [
    {"labels": ["cell_segmentation"]},
    {"labels": ["cell_segmentation", "nucleus_segmentation", "boundary"]},
]


@pytest.mark.parametrize("obj", V06_VALID_LABELS_GROUPS)
def test_valid_v06_labels_groups(obj: dict) -> None:
    validate_ome_object(obj, v06.LabelsGroup)


V06_INVALID_LABELS: list[tuple[dict, str]] = [
    (
        _label({"colors": [{"label-value": 1, "rgba": [256, 0, 0, 255]}]}),
        "less than or equal to 255",
    ),
    (
        _label({"colors": [{"label-value": 1, "rgba": [-1, 0, 0, 255]}]}),
        "greater than or equal to 0",
    ),
    (_label({"colors": [{"label-value": 1, "rgba": [255, 0, 0]}]}), "at least 4 items"),
    (
        _label({"colors": [{"label-value": 1, "rgba": [1, 2, 3, 4, 5]}]}),
        "at most 4 items",
    ),
    (_label({"colors": []}), "at least 1 item"),
    (_label({"properties": []}), "at least 1 item"),
    (_label({"colors": [COLOR_1, COLOR_1]}), "List items are not unique"),
    (_label({"properties": [{"label-value": 1.5}]}), "Input should be a valid integer"),
    (
        _label({"colors": [{"label-value": 1, "rgba": [255.5, 0, 0, 255]}]}),
        "valid integer",
    ),
]


@pytest.mark.parametrize("obj, msg", V06_INVALID_LABELS)
def test_invalid_v06_labels(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        v06.LabelImage.model_validate(obj)


@pytest.mark.parametrize(
    "obj, msg", [({"labels": []}, "at least 1 item"), ({}, "Field required")]
)
def test_invalid_v06_labels_groups(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        v06.LabelsGroup.model_validate(obj)
