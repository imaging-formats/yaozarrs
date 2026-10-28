from __future__ import annotations

import re
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
    # flips are allowed (scale values "SHOULD be non-zero"); resolution order
    # compares magnitudes
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
    # ... or "a sequence of a scale and a translation transformation" (ome/ngff#639)
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
    # "If the `path` field is null or omitted, this is to be interpreted as referring
    # to a named coordinate system in the same `zarr.json` file."
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
    # (as in v05)
    (image([dataset("s0", scale("s0", 1, 1))] * 2), "List items are not unique"),
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
    # "The `name` field under `input` SHOULD be omitted. ... The `path` field of
    # `output` SHOULD be omitted."
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
    # _version.schema also allows "0.6rc0"
    ({**image(), "version": "0.6rc0"}, "pre-release"),
    # array coordinate systems inside multiscales (allowed by axes.schema, not by
    # the prose)
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
