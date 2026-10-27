"""v0.6 documents are recognized by the version-agnostic entry points."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import VERSION

import yaozarrs
from yaozarrs import v05, v06
from yaozarrs._tree import render_tree
from yaozarrs._validation_warning import ValidationWarning

if TYPE_CHECKING:
    from pathlib import Path

IMAGE_OME = {
    "version": "0.6",
    "multiscales": [
        {
            "coordinateSystems": [
                {
                    "name": "physical",
                    "axes": [
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
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
    ],
}
PYDANTIC_VERSION = tuple(int(x) for x in VERSION.split(".")[:2])
ZARR_JSON = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": IMAGE_OME}}


def test_validate_ome_json() -> None:
    doc = yaozarrs.validate_ome_json(json.dumps(ZARR_JSON))
    assert isinstance(doc, v06.OMEZarrGroupJSON)
    assert isinstance(doc.attributes.ome, v06.Image)


def test_validate_ome_object() -> None:
    assert isinstance(yaozarrs.validate_ome_object(IMAGE_OME), v06.Image)
    plate = {
        "version": "0.6",
        "plate": {
            "columns": [{"name": "1"}],
            "rows": [{"name": "A"}],
            "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
        },
    }
    # NOTE: before pydantic 2.8, the (smart-mode) union picked the first member that
    # validates, and v04.Plate (which keeps `version` inside `plate`) accepts plates
    # of any version, so this resolved to v04.Plate (for v0.5 plates as well).
    if PYDANTIC_VERSION >= (2, 8):
        assert isinstance(yaozarrs.validate_ome_object(plate), v06.Plate)
        # a v0.5 document still resolves to v05
        v05_plate = {**plate, "version": "0.5"}
        assert isinstance(yaozarrs.validate_ome_object(v05_plate), v05.Plate)


@pytest.fixture
def v06_store(tmp_path: Path) -> Path:
    path = tmp_path / "image.zarr"
    path.mkdir()
    (path / "zarr.json").write_text(json.dumps(ZARR_JSON))
    return path


def test_zarr_group_ome_metadata(v06_store: Path) -> None:
    pytest.importorskip("fsspec")
    group = yaozarrs.open_group(v06_store)
    assert group.ome_version() == "0.6"
    assert isinstance(group.ome_metadata(), v06.Image)


def test_zarr_group_prerelease_version(v06_store: Path) -> None:
    pytest.importorskip("fsspec")
    doc = {**ZARR_JSON, "attributes": {"ome": {**IMAGE_OME, "version": "0.6rc0"}}}
    (v06_store / "zarr.json").write_text(json.dumps(doc))
    with pytest.warns(ValidationWarning, match="pre-release"):
        meta = yaozarrs.open_group(v06_store).ome_metadata()
    assert isinstance(meta, v06.Image)


def test_tree_annotation(v06_store: Path) -> None:
    pytest.importorskip("fsspec")
    assert "<- v06.Image" in render_tree(yaozarrs.open_group(v06_store))


def test_storage_validation_not_implemented(v06_store: Path) -> None:
    pytest.importorskip("fsspec")
    with pytest.raises(NotImplementedError, match=r"0\.6"):
        yaozarrs.validate_zarr_store(v06_store)
