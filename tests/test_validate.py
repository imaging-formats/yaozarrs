"""The version-agnostic `validate_ome_*` entry points dispatch on declared version."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

import yaozarrs
from yaozarrs import v04, v05
from yaozarrs._validation_warning import ValidationWarning

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
