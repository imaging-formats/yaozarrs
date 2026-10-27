from __future__ import annotations

import pytest

from yaozarrs import v04, v05

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
