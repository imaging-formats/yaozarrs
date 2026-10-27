from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from yaozarrs import v06

MULTISCALE = {
    "coordinateSystems": [
        {
            "name": "physical",
            "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        }
    ],
    "datasets": [
        {
            "path": "s0",
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [1, 1],
                    "input": {"path": "s0"},
                    "output": {"name": "physical"},
                }
            ],
        }
    ],
}
COLOR_1 = {"label-value": 1, "rgba": [255, 0, 0, 255]}
COLOR_2 = {"label-value": 2}


def label_image(image_label: dict) -> dict:
    return {"version": "0.6", "multiscales": [MULTISCALE], "image-label": image_label}


VALID_LABEL_IMAGES = [
    label_image({}),
    label_image({"colors": [COLOR_1, COLOR_2]}),
    label_image(
        {
            "colors": [COLOR_1],
            "properties": [{"label-value": 1, "class": "cell", "area (pixels)": 1200}],
            "source": {"image": "../../"},
        }
    ),
]


@pytest.mark.parametrize("obj", VALID_LABEL_IMAGES)
def test_valid_label_images(obj: dict) -> None:
    img = v06.LabelImage.model_validate(obj)
    again = v06.LabelImage.model_validate_json(img.model_dump_json())
    assert again == img


INVALID_LABEL_IMAGES = [
    ({"version": "0.6", "multiscales": [MULTISCALE]}, "image-label"),
    ({"version": "0.6", "image-label": {}}, "multiscales"),
    (label_image({"colors": []}), "at least 1 item"),
    (label_image({"properties": []}), "at least 1 item"),
    (label_image({"colors": [COLOR_1, COLOR_1]}), "not unique"),
    (label_image({"colors": [{"rgba": [0, 0, 0, 0]}]}), "Field required"),
    (label_image({"colors": [{"label-value": 3.5}]}), "valid integer"),
    (label_image({"colors": [{"label-value": 1, "rgba": [0, 0, 0]}]}), "at least 4"),
    (label_image({"colors": [{"label-value": 1, "rgba": [0, 0, 0, 256]}]}), "255"),
    (label_image({"properties": [{"value": "foo"}]}), "Field required"),
    (label_image({"source": "../../"}), "valid dictionary"),
]


@pytest.mark.parametrize("obj, msg", INVALID_LABEL_IMAGES)
def test_invalid_label_images(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(msg)):
        v06.LabelImage.model_validate(obj)


def test_label_properties_keep_extra_keys() -> None:
    obj = label_image({"properties": [{"label-value": 1, "cell type": "neuron"}]})
    dumped = v06.LabelImage.model_validate(obj).model_dump(mode="json")
    assert dumped["image-label"]["properties"] == [
        {"label-value": 1, "cell type": "neuron"}
    ]


def test_image_label_has_no_version() -> None:
    # v0.6 removed `image-label.version`; a stray one (from 0.5) is ignored
    img = v06.LabelImage.model_validate(label_image({"version": "0.5"}))
    assert "version" not in img.model_dump(mode="json")["image-label"]
    assert "version" not in v06.ImageLabel.model_fields


def test_labels_group() -> None:
    group = v06.LabelsGroup.model_validate({"version": "0.6", "labels": ["cells"]})
    assert group.labels == ["cells"]
    with pytest.raises(ValidationError, match="at least 1 item"):
        v06.LabelsGroup.model_validate({"version": "0.6", "labels": []})
    with pytest.raises(ValidationError, match=re.escape("'0.6'")):
        v06.LabelsGroup.model_validate({"version": "0.5", "labels": ["cells"]})


# ------------------------------------------------------------------------------
# upstream conformance fixtures: these only exercise the `image-label` object
# ------------------------------------------------------------------------------

FIXTURES = (
    Path(__file__).parent.parent / "data" / "v06" / "spec" / "tests" / "attributes"
)
LABEL_FIXTURES = sorted(
    p
    for p in FIXTURES.rglob("*.json")
    if "image-label" in json.loads(p.read_text()).get("ome", {})
    # (strict/invalid fixtures are only invalid under the strict schemas, which
    # yaozarrs doesn't model)
    and "strict/invalid" not in str(p)
)


@pytest.mark.parametrize(
    "path", LABEL_FIXTURES, ids=lambda p: str(p.relative_to(FIXTURES))
)
def test_upstream_label_fixtures(path: Path) -> None:
    upstream_valid = "/valid/" in str(path)
    try:
        v06.ImageLabel.model_validate(
            json.loads(path.read_text())["ome"]["image-label"]
        )
        valid = True
    except ValidationError:
        valid = False
    assert valid == upstream_valid
