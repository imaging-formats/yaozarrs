"""The version-agnostic `validate_ome_*` entry points dispatch on declared version."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

import yaozarrs
from yaozarrs import v04, v05, v06
from yaozarrs._validation_warning import ValidationWarning

if TYPE_CHECKING:
    from pathlib import Path

AXES = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
DATASETS = [
    {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}]}
]
V05_IMAGE = {"version": "0.5", "multiscales": [{"axes": AXES, "datasets": DATASETS}]}
V05_PLATE = {
    "version": "0.5",
    "plate": {
        "columns": [{"name": "1"}],
        "rows": [{"name": "A"}],
        "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
    },
}
V04_IMAGE = {"multiscales": [{"version": "0.4", "axes": AXES, "datasets": DATASETS}]}
V06_IMAGE = {
    "version": "0.6",
    "multiscales": [
        {
            "coordinateSystems": [{"name": "physical", "axes": AXES}],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [1, 1],
                            "input": {"path": "0"},
                            "output": {"name": "physical"},
                        }
                    ],
                }
            ],
        }
    ],
}


def _zarr_json(ome: dict) -> dict:
    return {"zarr_format": 3, "node_type": "group", "attributes": {"ome": ome}}


@pytest.mark.parametrize(
    "obj, cls",
    [
        (V04_IMAGE, v04.Image),
        (V05_IMAGE, v05.Image),
        ({"ome": V05_IMAGE}, v05.OMEAttributes),
        (_zarr_json(V05_IMAGE), v05.OMEZarrGroupJSON),
        # before pydantic 2.8, the plain union resolved this to v04.Plate (which keeps
        # its version inside `plate`, so accepts any top-level version)
        (V05_PLATE, v05.Plate),
        # no version declared: all versions are tried
        ({"bioformats2raw.layout": 3}, v04.Bf2Raw),
    ],
    ids=["v04", "v05", "v05-attributes", "v05-group", "v05-plate", "no-version"],
)
def test_dispatch(obj: dict, cls: type) -> None:
    assert isinstance(yaozarrs.validate_ome_object(obj), cls)
    assert isinstance(yaozarrs.validate_ome_json(json.dumps(obj)), cls)


@pytest.mark.parametrize(
    "obj, cls",
    [
        ({**V04_IMAGE, "image-label": {}}, v04.LabelImage),
        ({**V05_IMAGE, "image-label": {}}, v05.LabelImage),
        ({**V06_IMAGE, "image-label": {}}, v06.LabelImage),
    ],
    ids=["v04", "v05", "v06"],
)
def test_label_image_dumped_by_field_name(obj: dict, cls: type) -> None:
    dumped = yaozarrs.validate_ome_object(obj).model_dump(by_alias=False)
    assert "image_label" in dumped
    assert isinstance(yaozarrs.validate_ome_object(dumped), cls)


def test_validate_ome_uri(tmp_path: Any) -> None:
    pytest.importorskip("fsspec")
    (tmp_path / "zarr.json").write_text(json.dumps(_zarr_json(V05_PLATE)))
    obj = yaozarrs.validate_ome_uri(tmp_path)
    assert isinstance(obj, v05.OMEZarrGroupJSON)
    assert isinstance(obj.attributes.ome, v05.Plate)


def test_other_versions_dont_warn() -> None:
    # (v04.Image also accepts these axes, so a plain union warned twice)
    axes = [{**AXES[0], "unit": "lightyear"}, AXES[1]]
    img = {**V05_IMAGE, "multiscales": [{"axes": axes, "datasets": DATASETS}]}
    with pytest.warns(ValidationWarning, match="lightyear") as record:
        yaozarrs.validate_ome_object(img)
    assert len(record) == 1


def test_errors_only_for_declared_version() -> None:
    img = {**V05_IMAGE, "multiscales": [{"axes": AXES, "datasets": []}]}
    with pytest.raises(ValidationError) as exc_info:
        yaozarrs.validate_ome_object(img)
    assert exc_info.value.error_count() == 1
    assert exc_info.value.errors()[0]["loc"][0] == "v05-metadata"


@pytest.mark.parametrize(
    "obj, loc",
    [
        (
            {"multiscales": [{"version": "0.3", "axes": AXES, "datasets": DATASETS}]},
            ("multiscales", 0, "version"),
        ),
        ({"well": {"version": "foo", "images": [{"path": "0"}]}}, ("well", "version")),
        ({"version": "9.9", "well": {"images": [{"path": "0"}]}}, ("version",)),
        (
            {"ome": {"version": "9.9", "well": {"images": [{"path": "0"}]}}},
            ("ome", "version"),
        ),
        (
            _zarr_json({"version": "9.9", "series": ["0"]}),
            ("attributes", "ome", "version"),
        ),
    ],
)
def test_unknown_version_is_rejected(obj: dict, loc: tuple) -> None:
    # (rather than being tried against other versions, which may ignore it), with
    # a single error saying which versions are supported
    with pytest.raises(ValidationError) as from_obj:
        yaozarrs.validate_ome_object(obj)
    with pytest.raises(ValidationError) as from_json:
        yaozarrs.validate_ome_json(json.dumps(obj))
    for exc_info in (from_obj, from_json):
        (err,) = exc_info.value.errors()
        assert err["type"] == "unsupported_version"
        assert err["loc"] == loc
        assert "expected one of 0.4, 0.5, 0.6" in err["msg"]


def test_unknown_version_from_uri(tmp_path: Path) -> None:
    pytest.importorskip("fsspec")
    (tmp_path / "zarr.json").write_text(json.dumps(_zarr_json({"version": "0.3"})))
    with pytest.raises(ValidationError, match=r"Unsupported OME-Zarr version '0\.3'"):
        yaozarrs.validate_ome_uri(tmp_path)


PLATE = {k: v for k, v in V05_PLATE.items() if k != "version"}


@pytest.mark.parametrize(
    "obj, loc",
    [
        ({"ome": PLATE}, ("ome", "version")),
        (_zarr_json(PLATE), ("attributes", "ome", "version")),
        ({"ome": {k: v for k, v in V06_IMAGE.items() if k != "version"}}, None),
    ],
    ids=["attributes", "zarr.json", "v06-image"],
)
def test_missing_version_is_not_guessed(obj: dict, loc: tuple | None) -> None:
    # (a plate is the same in v0.5 and v0.6, so it used to pass as the first of them)
    with pytest.raises(ValidationError) as from_dict:
        yaozarrs.validate_ome_object(obj)
    with pytest.raises(ValidationError) as from_json:
        yaozarrs.validate_ome_json(json.dumps(obj))
    for exc_info in (from_dict, from_json):
        errors = exc_info.value.errors()
        assert len(errors) == 1  # (not one per version's models)
        assert errors[0]["type"] == "missing_version"
        assert errors[0]["loc"] == (loc or ("ome", "version"))


def test_documents_valid_without_version() -> None:
    # v0.4 metadata doesn't need a version, nor does a v0.5 labels group
    assert isinstance(yaozarrs.validate_ome_object(PLATE), v04.Plate)
    labels = yaozarrs.validate_ome_object({"ome": {"labels": ["cells"]}})
    assert isinstance(labels, v05.OMEAttributes)
    # and models of a given version fill it in
    assert v06.Plate.model_validate(PLATE).version == "0.6"
    assert yaozarrs.validate_ome_object(PLATE, v05.Plate).version == "0.5"


@pytest.mark.parametrize(
    "obj",
    [
        None,
        [],
        "0.5",
        {"version": ["0.5"]},
        {"multiscales": "nope"},
        {"multiscales": []},
        {"attributes": None},
        {"ome": "nope"},
        {"zarr_format": 3, "node_type": "group", "attributes": {"ome": []}},
    ],
)
def test_malformed(obj: Any) -> None:
    with pytest.raises(ValidationError):
        yaozarrs.validate_ome_object(obj)


@pytest.mark.parametrize("data", ["{not json", b"\xff"])
def test_malformed_json(data: str | bytes) -> None:
    # (still a ValidationError, not the JSONDecodeError of re-parsing it)
    with pytest.raises(ValidationError):
        yaozarrs.validate_ome_json(data)


COERCIBLE = [
    {"well": {"images": [{"acquisition": "0", "path": "0"}]}},
    {"version": "0.5", "well": {"images": [{"acquisition": "0", "path": "0"}]}},
    {
        "ome": {
            "version": "0.6",
            "well": {"images": [{"acquisition": "0", "path": "0"}]},
        }
    },
    {
        **V05_IMAGE,
        "omero": {
            "channels": [
                {
                    "color": "ff0000",
                    "window": {"start": 0, "min": 0, "end": "100", "max": 255},
                }
            ]
        },
    },
]


@pytest.mark.parametrize("obj", COERCIBLE)
def test_strict_rejects_what_is_coerced_by_default(obj: dict) -> None:
    yaozarrs.validate_ome_object(obj)
    yaozarrs.validate_ome_json(json.dumps(obj))
    with pytest.raises(ValidationError):
        yaozarrs.validate_ome_object(obj, strict=True)
    with pytest.raises(ValidationError):
        yaozarrs.validate_ome_json(json.dumps(obj), strict=True)


@pytest.mark.parametrize("obj", COERCIBLE)
def test_coerced_input_is_never_output(obj: dict) -> None:
    dumped = yaozarrs.validate_ome_object(obj).model_dump_json(exclude_unset=True)
    yaozarrs.validate_ome_json(dumped, strict=True)


@pytest.mark.parametrize("obj", [V04_IMAGE, V05_IMAGE, V05_PLATE, V06_IMAGE])
def test_strict_accepts_valid_documents(obj: dict) -> None:
    yaozarrs.validate_ome_object(obj, strict=True)
    yaozarrs.validate_ome_json(json.dumps(obj), strict=True)


@pytest.mark.parametrize("mod", [v04, v05, v06], ids=lambda m: m.__name__)
def test_bf2raw_layout_defaults_to_3(mod: Any) -> None:
    assert mod.Bf2Raw().model_dump(by_alias=True)["bioformats2raw.layout"] == 3
    # the default doesn't make a document without the key a bioformats2raw one
    doc = {} if mod is v04 else {"version": mod.Bf2Raw().version}
    with pytest.raises(ValidationError):
        yaozarrs.validate_ome_object(doc)


def test_validation_warning_is_a_user_warning() -> None:
    # (so the usual `UserWarning` filters catch it, like yaozarrs' other warnings)
    assert issubclass(ValidationWarning, UserWarning)
