from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from yaozarrs import v06
from yaozarrs._validation_warning import ValidationWarning

Y = {"name": "y", "type": "space", "unit": "micrometer"}
X = {"name": "x", "type": "space", "unit": "micrometer"}
Z = {"name": "z", "type": "space", "unit": "micrometer"}
T = {"name": "t", "type": "time", "unit": "second"}
C = {"name": "c", "type": "channel"}
PHYSICAL = {"name": "physical", "axes": [Y, X]}


def scale(path: str, *s: float, out: str = "physical") -> dict:
    return {
        "type": "scale",
        "scale": list(s),
        "input": {"path": path},
        "output": {"name": out},
    }


def dataset(path: str, transform: dict) -> dict:
    return {"path": path, "coordinateTransformations": [transform]}


def image(
    datasets: list[dict] | None = None,
    systems: list[dict] | None = None,
    transforms: list[dict] | None = None,
    **kwargs: Any,
) -> dict:
    ms: dict = {
        "coordinateSystems": [PHYSICAL] if systems is None else systems,
        "datasets": [dataset("s0", scale("s0", 1, 1))]
        if datasets is None
        else datasets,
        **kwargs,
    }
    if transforms is not None:
        ms["coordinateTransformations"] = transforms
    return {"version": "0.6", "multiscales": [ms]}


SEQ = {
    "type": "sequence",
    "transformations": [
        {"type": "scale", "scale": [2, 2]},
        {"type": "translation", "translation": [0.5, 0.5]},
    ],
    "input": {"path": "s1"},
    "output": {"name": "physical"},
}
OTHER = {"name": "other", "axes": [Y, X]}
TO_OTHER = {
    "type": "translation",
    "translation": [10, 10],
    "input": {"name": "physical"},
    "output": {"name": "other"},
}

VALID_IMAGES = [
    image(),
    image([dataset("s0", scale("s0", 1, 1)), dataset("s1", SEQ)]),
    image([dataset("s0", {**scale("s0"), "type": "identity"})]),
    # 5D, with name/type/metadata
    image(
        [dataset("0", scale("0", 1, 1, 0.5, 0.5, 0.5))],
        [{"name": "physical", "axes": [T, C, Z, Y, X]}],
        name="img",
        type="gaussian",
        metadata={"method": "skimage.transform.pyramid_gaussian"},
    ),
    # flips are allowed (conflict D1); resolution order compares magnitudes
    image([dataset("s0", scale("s0", -1, 1)), dataset("s1", scale("s1", -2, 2))]),
    # additional transform to another coordinate system of this multiscale
    image(systems=[PHYSICAL, OTHER], transforms=[TO_OTHER]),
    # ... and in the other direction
    image(
        systems=[PHYSICAL, OTHER],
        transforms=[
            {**TO_OTHER, "input": {"name": "other"}, "output": {"name": "physical"}}
        ],
    ),
    # transform to a child label image (identity, scale, translation)
    image(
        transforms=[
            {
                "type": "identity",
                "input": {"name": "physical"},
                "output": {"name": "physical", "path": "labels/cells"},
            }
        ]
    ),
    # ... or a sequence of a scale and a translation (conflict P3, ome/ngff#639)
    image(
        transforms=[
            {
                **SEQ,
                "input": {"name": "physical"},
                "output": {"name": "physical", "path": "labels/cells"},
            }
        ]
    ),
    # ... in either order (the labels section gives none)
    image(
        transforms=[
            {
                **SEQ,
                "transformations": SEQ["transformations"][::-1],
                "input": {"name": "physical"},
                "output": {"name": "physical", "path": "labels/cells"},
            }
        ]
    ),
    # explicit null path means "same document" (conflict N1)
    image(
        systems=[PHYSICAL, OTHER],
        transforms=[{**TO_OTHER, "output": {"name": "other", "path": None}}],
    ),
    {**image(), "version": "0.6", "omero": {"channels": []}},
]


@pytest.mark.parametrize("obj", VALID_IMAGES)
def test_valid_images(obj: dict) -> None:
    img = v06.Image.model_validate(obj)
    again = v06.Image.model_validate_json(img.model_dump_json())
    assert again == img


INVALID_IMAGES = [
    ({"version": "0.6", "multiscales": []}, "at least 1 item"),
    ({"version": "0.5", "multiscales": [image()["multiscales"][0]]}, "'0.6'"),
    ({"version": "0.6", "multiscales": [{"datasets": []}]}, "coordinateSystems"),
    (image(systems=[]), "at least 1 item"),
    (
        {
            "version": "0.6",
            "multiscales": [{**image()["multiscales"][0], "datasets": []}],
        },
        "at least 1 item",
    ),
    # dataset transformations
    (image([{"path": "s0", "coordinateTransformations": []}]), "at least 1 item"),
    (
        image([{"path": "s0", "coordinateTransformations": [scale("s0", 1, 1)] * 2}]),
        "at most 1 item",
    ),
    (
        image(
            [
                dataset(
                    "s0", {**scale("s0"), "type": "translation", "translation": [1, 1]}
                )
            ]
        ),
        "does not match any of the expected tags",
    ),
    (
        image(
            [dataset("s1", {**SEQ, "transformations": SEQ["transformations"][::-1]})]
        ),
        "must be a scale followed by a translation",
    ),
    (
        image(
            [
                dataset(
                    "s0",
                    {"type": "scale", "scale": [1, 1], "output": {"name": "physical"}},
                )
            ]
        ),
        "must specify the dataset 'path'",
    ),
    (image([dataset("s0", scale("s1", 1, 1))]), "does not match the dataset path"),
    (
        image([dataset("s0", {**scale("s0", 1, 1), "output": {"path": "x"}})]),
        "must specify the 'name'",
    ),
    (image([dataset("s0", scale("s0", 1, 1, 1))]), "scale: has 3 values"),
    # intrinsic coordinate system
    (
        image(
            [
                dataset("s0", scale("s0", 1, 1)),
                dataset("s1", scale("s1", 2, 2, out="other")),
            ],
            [PHYSICAL, OTHER],
            [TO_OTHER],
        ),
        "same (intrinsic) coordinate system",
    ),
    (image([dataset("s0", scale("s0", 1, 1, out="nope"))]), "'nope' is not declared"),
    # multiscale axes rules
    (image(systems=[{"name": "physical", "axes": [Y, X, T]}]), "required order"),
    (
        image(systems=[{"name": "physical", "axes": [{**X, "name": "w"}, Z, Y, X]}]),
        "2 or 3 axes of type 'space'",
    ),
    # multiscale-level transformations
    (
        image(systems=[PHYSICAL, OTHER], transforms=[{**TO_OTHER, "input": None}]),
        "both an 'input' and an 'output'",
    ),
    (
        image(
            systems=[PHYSICAL, OTHER], transforms=[{**TO_OTHER, "input": {"path": "x"}}]
        ),
        "must specify a 'name'",
    ),
    (
        image(
            systems=[PHYSICAL, OTHER, {**OTHER, "name": "third"}],
            transforms=[
                TO_OTHER,
                {**TO_OTHER, "input": {"name": "other"}, "output": {"name": "third"}},
            ],
        ),
        "must reference the intrinsic coordinate system",
    ),
    (image(transforms=[TO_OTHER]), "'other' is not declared in this multiscale"),
    (
        image(
            transforms=[
                {
                    "type": "affine",
                    "affine": [[1, 0, 0], [0, 1, 0]],
                    "input": {"name": "physical"},
                    "output": {"name": "x", "path": "labels/x"},
                }
            ]
        ),
        "child label image must be identity, scale, translation",
    ),
    (
        image(
            systems=[PHYSICAL, OTHER],
            transforms=[{**TO_OTHER, "translation": [1, 1, 1]}],
        ),
        "translation: has 3 values",
    ),
    # connectivity
    (image(systems=[PHYSICAL, OTHER]), "['other'] are not linked to the intrinsic"),
    # resolution order
    (
        image([dataset("s0", scale("s0", 2, 2)), dataset("s1", scale("s1", 1, 1))]),
        "not ordered from highest to lowest",
    ),
]


@pytest.mark.parametrize("obj, msg", INVALID_IMAGES)
def test_invalid_images(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(msg)):
        v06.Image.model_validate(obj)


WARNING_IMAGES = [
    # conflict P1: SHOULD omit input.name / output.path on dataset transforms
    (
        image(
            [dataset("s0", {**scale("s0", 1, 1), "input": {"path": "s0", "name": "a"}})]
        ),
        "SHOULD be omitted",
    ),
    (
        image(
            [
                dataset(
                    "s0",
                    {**scale("s0", 1, 1), "output": {"name": "physical", "path": "."}},
                )
            ]
        ),
        "SHOULD be omitted",
    ),
    # conflict D21
    ({**image(), "version": "0.6rc0"}, "pre-release"),
    # conflict D6: array coordinate systems inside multiscales
    (
        image(
            [
                dataset(
                    "s0",
                    {
                        "type": "identity",
                        "input": {"path": "s0"},
                        "output": {"name": "array"},
                    },
                )
            ],
            [
                {
                    "name": "array",
                    "axes": [
                        {"name": "i", "type": "array"},
                        {"name": "j", "type": "array"},
                    ],
                }
            ],
        ),
        "'array' axes are only allowed",
    ),
]


@pytest.mark.parametrize("obj, msg", WARNING_IMAGES)
def test_image_warnings(obj: dict, msg: str) -> None:
    with pytest.warns(ValidationWarning, match=re.escape(msg)):
        v06.Image.model_validate(obj)


def test_multiscale_properties() -> None:
    img = v06.Image.model_validate(
        image(
            [dataset("s0", scale("s0", 1, 1)), dataset("s1", SEQ)],
            [PHYSICAL, OTHER],
            [TO_OTHER],
        )
    )
    ms = img.multiscales[0]
    assert ms.intrinsic_coordinate_system.name == "physical"
    assert [ax.name for ax in ms.axes] == ["y", "x"]
    assert ms.ndim == 2
    s0, s1 = ms.datasets
    assert s0.scale_transform.scale == [1, 1]  # type: ignore[union-attr]
    assert s0.translation_transform is None
    assert s1.scale_transform.scale == [2, 2]  # type: ignore[union-attr]
    assert s1.translation_transform.translation == [0.5, 0.5]  # type: ignore[union-attr]


# ------------------------------------------------------------------------------
# upstream conformance fixtures (tests/attributes/**), image documents only
# ------------------------------------------------------------------------------

FIXTURES = (
    Path(__file__).parent.parent / "data" / "v06" / "spec" / "tests" / "attributes"
)


def _image_fixtures() -> list[Path]:
    out = []
    for path in sorted(FIXTURES.rglob("*.json")):
        ome = json.loads(path.read_text()).get("ome", {})
        if "multiscales" in ome and "image-label" not in ome:
            out.append(path)
    return out


# fixtures where yaozarrs deliberately disagrees with upstream's verdict
# (each value is the reason, citing dev_notes/v06/conflicts.md)
EXPECTED_DEVIATIONS: dict[str, str] = {
    # upstream invalid, yaozarrs valid
    "spec/invalid/image/invalid_channels_window.json": (
        'N1: pydantic lax mode accepts the numeric string "100" for window.end '
        "(library-wide; shared Omero class)"
    ),
    # upstream valid, yaozarrs invalid (fixture bugs the schema can't catch)
    "spec/valid/image/mismatch_axes_units.json": (
        "fixture bug: 2-value scale for a 3-axis coordinate system violates "
        '"MUST equal the length of the scale array" (upstream_fixes.md #15)'
    ),
    "spec/valid/image/multiscales_transform_additional_transforms.json": (
        "fixture bug: the final byDimension step of a 3D sequence covers only output "
        "axes 0 and 1, violating \"Every axis index in the parent byDimension's output "
        "coordinate system MUST appear in exactly one child transformation's "
        'outputAxes array" (upstream_fixes.md #17)'
    ),
    "strict/valid/image/multiscales_example.json": (
        'fixture bug: dataset path "1" but input.path "s1" violates "input ... MUST '
        'specify a path field matching the path field of the dataset" '
        "(upstream_fixes.md #16; the published example is correct)"
    ),
}


@pytest.mark.parametrize(
    "path", _image_fixtures(), ids=lambda p: str(p.relative_to(FIXTURES))
)
def test_upstream_image_fixtures(path: Path) -> None:
    upstream_valid = "/valid/" in str(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidationWarning)
        try:
            v06.Image.model_validate(json.loads(path.read_text())["ome"])
            valid = True
        except ValidationError:
            valid = False
    if (key := str(path.relative_to(FIXTURES))) in EXPECTED_DEVIATIONS:
        assert valid != upstream_valid, f"deviation no longer needed: {key}"
    else:
        assert valid == upstream_valid
