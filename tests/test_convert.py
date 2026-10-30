from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typing_extensions import assert_type

from yaozarrs import ConversionWarning, to_version, v04, v05, v06, validate_ome_object

DATA = Path(__file__).parent / "data"
VERSIONS: dict[str, Any] = {"v04": "0.4", "v05": "0.5", "v06": "0.6"}
PACKAGES: dict[str, Any] = {"0.4": v04, "0.5": v05, "0.6": v06}
OME_KEYS = {
    "multiscales",
    "plate",
    "well",
    "labels",
    "series",
    "scene",
    "bioformats2raw.layout",
}


def _load(path: Path) -> Any | None:
    """Return the OME metadata model in a `.zattrs` or `zarr.json` (if it has any)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    package = path.relative_to(DATA).parts[0]
    if package != "v04":
        data = (data.get("attributes") or {}).get("ome")
    if not (isinstance(data, dict) and OME_KEYS & data.keys()):
        return None
    cls: Any = (
        v04.OMEZarrGroupJSON
        if package == "v04"
        else PACKAGES[VERSIONS[package]].OMEMetadata
    )
    return validate_ome_object(data, cls)


FIXTURES = sorted(
    path
    for package in VERSIONS
    for path in (DATA / package).rglob("*")
    if path.name in (".zattrs", "zarr.json")
)
IDS = [str(p.relative_to(DATA)) for p in FIXTURES]


def _multiscale_05(**kwargs: Any) -> dict:
    return {
        "axes": [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 0.5, 0.5]},
                ],
            },
            {
                "path": "1",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                    {"type": "translation", "translation": [0.0, 0.25, 0.25]},
                ],
            },
        ],
        **kwargs,
    }


def _multiscale_06(transform: dict | None = None, **kwargs: Any) -> dict:
    transform = transform or {"type": "scale", "scale": [0.5, 0.5]}
    return {
        "coordinateSystems": [
            {
                "name": "intrinsic",
                "axes": [
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
            },
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {
                        **transform,
                        "input": {"path": "0"},
                        "output": {"name": "intrinsic"},
                    }
                ],
            }
        ],
        **kwargs,
    }


def _label_image_04(**image_label: Any) -> v04.LabelImage:
    ms = {**_multiscale_05(), "version": "0.4"}
    return v04.LabelImage.model_validate(
        {"multiscales": [ms], "image-label": {"version": "0.4", **image_label}}
    )


# ------------------------------------------------------------------------------
# All fixtures
# ------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("path", FIXTURES, ids=IDS)
@pytest.mark.parametrize("version", VERSIONS.values())
def test_fixture_converts_to_every_version(path: Path, version: Any) -> None:
    if (obj := _load(path)) is None:
        pytest.skip("no OME metadata")
    if isinstance(obj, v06.Scene) and version != "0.6":
        with pytest.raises(ValueError, match="no equivalent"):
            to_version(obj, version)
        return
    assert type(to_version(obj, version)) is getattr(
        PACKAGES[version], type(obj).__name__
    )


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("path", FIXTURES, ids=IDS)
def test_fixture_round_trip_is_exact(path: Path) -> None:
    if (obj := _load(path)) is None:
        pytest.skip("no OME metadata")
    source = VERSIONS[path.relative_to(DATA).parts[0]]
    if source == "0.6":
        pytest.skip("v0.6 can hold more than older versions")
    for via in VERSIONS.values():
        back = to_version(to_version(obj, via), source)
        assert back.model_dump() == obj.model_dump()


# ------------------------------------------------------------------------------
# Upgrades
# ------------------------------------------------------------------------------


def test_upgrade_moves_axes_and_transforms() -> None:
    ms = to_version(v05.Multiscale.model_validate(_multiscale_05()), "0.6")
    assert ms.coordinateSystems[0].name == "intrinsic"
    assert [ax.name for ax in ms.axes] == ["t", "y", "x"]
    assert ms.axes[0].unit == "second"

    first, second = ms.datasets
    assert first.transform.type == "scale"
    assert first.transform.input.path == "0"
    assert first.transform.output.name == "intrinsic"
    assert second.transform.type == "sequence"
    assert second.translation_transform.translation == [0.0, 0.25, 0.25]


def test_coordinate_system_name() -> None:
    src = v05.Multiscale.model_validate(_multiscale_05())
    ms = to_version(src, "0.6", coordinate_system="physical")
    assert ms.coordinateSystems[0].name == "physical"
    assert ms.datasets[0].output_name == "physical"


def test_coordinate_system_never_renames() -> None:
    src = v06.Multiscale.model_validate(_multiscale_06())
    assert to_version(src, "0.6", coordinate_system="x").datasets[0].output_name == (
        "intrinsic"
    )


def test_upgrade_folds_multiscale_transforms() -> None:
    src = v05.Multiscale.model_validate(
        _multiscale_05(
            coordinateTransformations=[
                {"type": "scale", "scale": [0.1, 2.0, 2.0]},
                {"type": "translation", "translation": [5.0, 1.0, 1.0]},
            ]
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ms = to_version(src, "0.6")

    assert ms.coordinateTransformations is None
    assert len(ms.coordinateSystems) == 1
    assert ms.axes[0].unit == "second"  # units apply to the folded values

    # a dataset that had no translation gains the multiscale's
    first, second = ms.datasets
    assert first.scale_transform.scale == pytest.approx([0.1, 1.0, 1.0])
    assert first.translation_transform.translation == pytest.approx([5.0, 1.0, 1.0])
    # a dataset's own translation is scaled too: gs * t + gt
    assert second.scale_transform.scale == pytest.approx([0.1, 2.0, 2.0])
    assert second.translation_transform.translation == pytest.approx([5.0, 1.5, 1.5])


def test_upgrade_folds_multiscale_scale_without_translation() -> None:
    ms = _multiscale_05(
        coordinateTransformations=[{"type": "scale", "scale": [0.1, 2.0, 2.0]}]
    )
    first, second = to_version(v05.Multiscale.model_validate(ms), "0.6").datasets
    assert first.transform.type == "scale"
    assert first.scale_transform.scale == pytest.approx([0.1, 1.0, 1.0])
    # a dataset's own translation is still scaled
    assert second.translation_transform.translation == pytest.approx([0.0, 0.5, 0.5])


def test_tool_version_in_metadata_survives() -> None:
    metadata = {"method": "skimage.transform.pyramid_gaussian", "version": "0.16.1"}
    src = v05.Image.model_validate(
        {"multiscales": [_multiscale_05(metadata=metadata, type="gaussian")]}
    )
    for version in ("0.4", "0.6"):
        ms = to_version(src, version).multiscales[0]
        assert ms.metadata == metadata
        assert ms.type == "gaussian"


def test_nested_versions_are_replaced() -> None:
    src = _label_image_04(colors=[{"label-value": 1, "rgba": [1, 2, 3, 4]}])
    assert to_version(src, "0.5").version == "0.5"
    assert to_version(src, "0.6").version == "0.6"
    assert to_version(to_version(src, "0.5"), "0.4").image_label.version == "0.4"


def test_large_label_ids_are_exact_in_v04() -> None:
    big = 2**53 + 1  # not representable as a float
    label = {"label-value": big}
    src = v05.LabelImage.model_validate(
        {
            "multiscales": [_multiscale_05()],
            "image-label": {"colors": [label], "properties": [label]},
        }
    )
    result = to_version(src, "0.4").model_dump()["image-label"]
    assert result["colors"] == [label]
    assert result["colors"][0]["label-value"] == result["properties"][0]["label-value"]
    assert to_version(to_version(src, "0.4"), "0.5").model_dump() == src.model_dump()


def test_custom_label_properties_survive() -> None:
    src = _label_image_04(properties=[{"label-value": 1, "class": "neuron"}])
    for version in ("0.5", "0.6"):
        result = to_version(src, version)
        assert result.model_dump()["image-label"]["properties"] == [
            {"label-value": 1, "class": "neuron"}
        ]


def test_null_label_property_survives() -> None:
    src = _label_image_04(properties=[{"label-value": 1, "class": None}])
    result = to_version(src, "0.5").model_dump(exclude_none=False)
    assert result["image-label"]["properties"] == [{"label-value": 1, "class": None}]


def test_omero_is_untouched() -> None:
    omero = {
        "version": "0.1",
        "channels": [
            {"color": "FF0000", "window": {"min": 0, "max": 1, "start": 0, "end": 1}}
        ],
    }
    src = v05.Image.model_validate({"multiscales": [_multiscale_05()], "omero": omero})
    assert to_version(src, "0.6").omero == src.omero
    assert to_version(src, "0.6").omero.version == "0.1"


def test_bf2raw_plate_marker_survives() -> None:
    src = v05.Plate.model_validate(
        {
            "bioformats2raw.layout": 3,
            "plate": {
                "columns": [{"name": "1"}],
                "rows": [{"name": "A"}],
                "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
            },
        }
    )
    assert to_version(src, "0.4").bioformats2raw_layout == 3
    assert to_version(src, "0.6").bioformats2raw_layout == 3


# ------------------------------------------------------------------------------
# Downgrades
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (
            {"type": "scale", "scale": [0.5, 0.5]},
            [{"type": "scale", "scale": [0.5, 0.5]}],
        ),
        ({"type": "identity"}, [{"type": "scale", "scale": [1.0, 1.0]}]),
        (
            {
                "type": "sequence",
                "transformations": [
                    {"type": "scale", "scale": [0.5, 0.5]},
                    {"type": "translation", "translation": [1.0, 2.0]},
                ],
            },
            [
                {"type": "scale", "scale": [0.5, 0.5]},
                {"type": "translation", "translation": [1.0, 2.0]},
            ],
        ),
    ],
)
@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_downgrade_transforms(transform: dict, expected: list, version: Any) -> None:
    src = v06.Multiscale.model_validate(_multiscale_06(transform))
    # NOTE: a clean downgrade must not warn (pytest is configured to raise on warnings)
    ms = to_version(src, version)
    assert ms.model_dump()["datasets"][0]["coordinateTransformations"] == expected
    assert [ax.name for ax in ms.axes] == ["y", "x"]


def test_downgrade_drops_with_a_single_warning() -> None:
    src = v06.Image.model_validate(
        {
            "multiscales": [
                _multiscale_06(
                    coordinateSystems=[
                        {
                            "name": "intrinsic",
                            "axes": [
                                {"name": "y", "type": "space", "longName": "Y axis"},
                                {"name": "x", "type": "space", "discrete": False},
                            ],
                        },
                        {
                            "name": "other",
                            "axes": [
                                {"name": "y", "type": "space"},
                                {"name": "x", "type": "space"},
                            ],
                        },
                    ],
                    coordinateTransformations=[
                        {
                            "type": "scale",
                            "scale": [2.0, 2.0],
                            "input": {"name": "intrinsic"},
                            "output": {"name": "other"},
                        }
                    ],
                )
            ]
        }
    )
    with pytest.warns(ConversionWarning) as record:
        ms = to_version(src, "0.4").multiscales[0]
    assert len(record) == 1
    message = str(record[0].message)
    assert record[0].filename == __file__
    for text in (
        "coordinate system 'other'",
        "multiscale coordinateTransformations",
        "axis 'y' longName",
        "axis 'x' discrete",
    ):
        assert text in message
    assert [ax.name for ax in ms.axes] == ["y", "x"]
    assert ms.datasets[0].coordinateTransformations[0].scale == [0.5, 0.5]


def test_conversion_warning_can_be_an_error() -> None:
    src = v06.Multiscale.model_validate(
        _multiscale_06(
            coordinateSystems=[
                {
                    "name": "intrinsic",
                    "axes": [
                        {"name": "y", "type": "space", "longName": "Y"},
                        {"name": "x", "type": "space"},
                    ],
                }
            ]
        )
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConversionWarning)
        with pytest.raises(ConversionWarning):
            to_version(src, "0.5")


def test_label_color_extras_warn_only_for_v04() -> None:
    colors = [{"label-value": 1, "rgba": [1, 2, 3, 4], "note": "x"}]
    src = v05.LabelImage.model_validate(
        {"multiscales": [_multiscale_05()], "image-label": {"colors": colors}}
    )
    upgraded = to_version(src, "0.6")  # no warning
    assert upgraded.model_dump()["image-label"]["colors"] == colors
    with pytest.warns(ConversionWarning, match="label color key 'note'"):
        to_version(src, "0.4")
    with pytest.warns(ConversionWarning, match="label color key 'note'"):
        to_version(upgraded, "0.4")


def test_scene_cannot_be_downgraded() -> None:
    scene = v06.Scene.model_validate(
        {
            "scene": {
                "coordinateSystems": [
                    {
                        "name": "world",
                        "axes": [
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ],
                    }
                ],
                "coordinateTransformations": [
                    {
                        "type": "identity",
                        "input": {"path": "a", "name": "intrinsic"},
                        "output": {"name": "world"},
                    }
                ],
            }
        }
    )
    with pytest.raises(ValueError, match="no equivalent of 'Scene'"):
        to_version(scene, "0.5")  # type: ignore
    assert to_version(scene, "0.6") == scene


# ------------------------------------------------------------------------------
# Things that are not representable
# ------------------------------------------------------------------------------


def test_fractional_label_value_raises() -> None:
    src = _label_image_04(colors=[{"label-value": 1.5, "rgba": [1, 2, 3, 4]}])
    with pytest.raises(ValidationError):
        to_version(src, "0.5")


def test_array_axes_raise_on_downgrade() -> None:
    # (v0.6 multiscales allow these with a warning, older versions require "space")
    axes = [{"name": n, "type": "array"} for n in "yx"]
    ms = _multiscale_06(coordinateSystems=[{"name": "intrinsic", "axes": axes}])
    with pytest.warns(Warning, match="only allowed by the JSON schema"):
        src = v06.Multiscale.model_validate(ms)
    with pytest.raises(ValidationError):
        to_version(src, "0.5")


def test_unsupported_input() -> None:
    with pytest.raises(TypeError, match="Cannot convert"):
        to_version({"version": "0.5"}, "0.6")  # type: ignore
    src = v05.Multiscale.model_validate(_multiscale_05())
    with pytest.raises(ValueError, match="Unknown version"):
        to_version(src, "0.7")  # type: ignore


# ------------------------------------------------------------------------------
# General behavior
# ------------------------------------------------------------------------------


def test_same_version_normalizes_prerelease_tag() -> None:
    doc = {"version": "0.6.dev4", "labels": ["a"]}
    with pytest.warns(Warning, match="pre-release"):
        src = v06.LabelsGroup.model_validate(doc)
    result = to_version(src, "0.6")
    assert result.version == "0.6"
    assert result is not src


def test_source_is_not_modified_or_aliased() -> None:
    metadata = {"nested": {"a": 1}}
    src = v05.Image.model_validate({"multiscales": [_multiscale_05(metadata=metadata)]})
    before = src.model_dump()
    result = to_version(src, "0.6")
    assert src.model_dump() == before
    assert result.multiscales[0].metadata == metadata
    assert result.multiscales[0].metadata is not src.multiscales[0].metadata


def test_uri_is_not_carried_over() -> None:
    src = v04.Image.model_validate(
        {"multiscales": [{**_multiscale_05(), "version": "0.4"}]}
    )
    src.uri = "/some/path/.zattrs"
    assert to_version(src, "0.4").uri is None


def test_subclass_converts_to_its_parent_kind() -> None:
    class MyImage(v05.Image):
        datasets: list | None = None  # clashes with a key of the spec's `Multiscale`

    src = MyImage.model_validate({"multiscales": [_multiscale_05()]})
    assert type(to_version(src, "0.6")) is v06.Image


@pytest.mark.parametrize("target", ["0.5", "0.6"])
def test_group_json_keeps_extra_attributes(target: Any) -> None:
    src = v05.OMEZarrGroupJSON.model_validate(
        {
            "attributes": {
                "other": {"keep": "me"},
                "ome": {"version": "0.5", "multiscales": [_multiscale_05()]},
            }
        }
    )
    result = to_version(src, target)
    assert type(result) is PACKAGES[target].OMEZarrGroupJSON
    assert result.attributes.ome.version == target
    assert result.attributes.model_extra == {"other": {"keep": "me"}}


def test_group_json_to_v04_returns_the_document() -> None:
    ome = {"version": "0.5", "multiscales": [_multiscale_05()]}
    plain = v05.OMEZarrGroupJSON.model_validate({"attributes": {"ome": ome}})
    assert isinstance(to_version(plain, "0.4"), v04.Image)

    extra = v05.OMEAttributes.model_validate({"other": 1, "ome": ome})
    with pytest.warns(ConversionWarning, match="attribute 'other'"):
        assert isinstance(to_version(extra, "0.4"), v04.Image)


def test_static_return_types() -> None:
    image = v04.Image.model_validate(
        {"multiscales": [{**_multiscale_05(), "version": "0.4"}]}
    )
    assert_type(to_version(image, "0.4"), v04.Image)
    assert_type(to_version(image, "0.5"), v05.Image)
    assert_type(to_version(image, "0.6"), v06.Image)
    label = _label_image_04()
    assert_type(to_version(label, "0.6"), v06.LabelImage)
    assert_type(to_version(image.multiscales[0], "0.6"), v06.Multiscale)
