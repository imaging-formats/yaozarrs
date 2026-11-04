from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from yaozarrs import v04, v05, v06, validate_ome_object

AXES = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
DATASETS = [
    {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1]}]}
]
IMAGES = [
    v04.Image(multiscales=[{"version": "0.4", "axes": AXES, "datasets": DATASETS}]),
    v05.Image(multiscales=[{"axes": AXES, "datasets": DATASETS}]),
]


@pytest.mark.parametrize("img", IMAGES, ids=lambda x: type(x).__module__)
def test_round_trip_without_nulls(img: v04.Image | v05.Image) -> None:
    # unset optional fields default to None, and the NGFF schemas never allow null
    js = img.model_dump_json()
    assert "null" not in js
    assert type(img).model_validate_json(js) == img


def test_unset_fields_not_dumped() -> None:
    axis = v05.SpaceAxis(name="y")
    assert axis.model_dump_json() == '{"name":"y","type":"space"}'
    assert axis.model_dump() == {"name": "y", "type": "space"}
    assert axis.model_dump(exclude_none=False)["unit"] is None


DATA = Path(__file__).parent / "data"
OME_KEYS = {"multiscales", "plate", "well", "labels", "series", "scene"}
GROUP_DOCS: dict[str, Any] = {
    "v04": v04.OMEZarrGroupJSON,
    "v05": v05.OMEZarrGroupJSON,
    "v06": v06.OMEZarrGroupJSON,
}
FIXTURES = sorted(
    path
    for version in GROUP_DOCS
    for path in (DATA / version).rglob("*")
    if path.name in (".zattrs", "zarr.json")
)
MARK = "x-unknown-key"


def _ome(doc: dict) -> Any:
    return doc if "zarr_format" not in doc else doc.get("attributes", {}).get("ome")


def _mark(obj: Any, path: tuple = ()) -> set[tuple]:
    """Add MARK to every JSON object in `obj`; return where it went."""
    marked = set()
    if isinstance(obj, dict):
        # (scene.schema allows no other keys in the input/output of its transforms)
        if not (path[-1:] in (("input",), ("output",)) and "scene" in path):
            obj[MARK] = 1
            marked.add(path)
        for key, value in list(obj.items()):
            marked |= _mark(value, (*path, key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            marked |= _mark(value, (*path, i))
    return marked


def _marks(obj: Any, path: tuple = ()) -> set[tuple]:
    if isinstance(obj, dict):
        found = {path} if MARK in obj else set()
        return found.union(*(_marks(v, (*path, k)) for k, v in obj.items()))
    if isinstance(obj, list):
        return set().union(*(_marks(v, (*path, i)) for i, v in enumerate(obj)))
    return set()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: str(p.relative_to(DATA)))
def test_unknown_keys_round_trip(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    ome = _ome(doc)
    if not (isinstance(ome, dict) and OME_KEYS & ome.keys()):
        pytest.skip("no OME metadata")
    marked = _mark(doc)
    model = validate_ome_object(doc, GROUP_DOCS[path.relative_to(DATA).parts[0]])
    assert _marks(model.model_dump(mode="json")) == marked
