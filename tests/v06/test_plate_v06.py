from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from yaozarrs import v06
from yaozarrs._validation_warning import ValidationWarning

COLUMNS = [{"name": "1"}, {"name": "2"}]
ROWS = [{"name": "A"}, {"name": "B"}]
WELLS = [
    {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
    {"path": "B/2", "rowIndex": 1, "columnIndex": 1},
]


def plate(**kwargs: object) -> dict:
    return {
        "version": "0.6",
        "plate": {"columns": COLUMNS, "rows": ROWS, "wells": WELLS, **kwargs},
    }


def well(*images: dict) -> dict:
    return {"version": "0.6", "well": {"images": list(images)}}


VALID_PLATES = [
    plate(),
    plate(
        name="plate",
        field_count=2,
        acquisitions=[{"id": 0, "name": "a"}, {"id": 1, "maximumfieldcount": 2}],
    ),
    {**plate(), "bioformats2raw.layout": 3},
]


@pytest.mark.parametrize("obj", VALID_PLATES)
def test_valid_plates(obj: dict) -> None:
    p = v06.Plate.model_validate(obj)
    assert v06.Plate.model_validate_json(p.model_dump_json()) == p


INVALID_PLATES = [
    ({**plate(), "version": "0.5"}, "'0.6'"),
    (plate(wells=[{"path": "A/1", "rowIndex": 1, "columnIndex": 0}]), "does not match"),
    (plate(acquisitions=[{"id": 0}, {"id": 0}]), "Acquisition ids must be unique"),
    (plate(columns=[{"name": "1-1"}]), "should match pattern"),
    ({**plate(), "bioformats2raw.layout": 2}, "Input should be 3"),
]


@pytest.mark.parametrize("obj, msg", INVALID_PLATES)
def test_invalid_plates(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(msg)):
        v06.Plate.model_validate(obj)


# v0.6 relaxed field-of-view paths: `a-z`, `A-Z`, `0-9`, `-`, `_`, `.` (no warning)
VALID_WELLS = [
    well({"path": "0"}),
    well({"path": "0", "acquisition": 1}, {"path": "1", "acquisition": 2}),
    well({"path": "fov_1"}, {"path": "fov-2"}, {"path": "fov.3"}),
]


@pytest.mark.parametrize("obj", VALID_WELLS)
def test_valid_wells(obj: dict) -> None:
    w = v06.Well.model_validate(obj)
    assert v06.Well.model_validate_json(w.model_dump_json()) == w


INVALID_WELLS = [
    (well(), "at least 1 item"),
    (well({"path": ""}), "should match pattern"),
    (well({"path": "a/b"}), "should match pattern"),
    (well({"path": "fov 1"}), "should match pattern"),
    (well({"path": ".."}), "not a valid Zarr node name"),
    (well({"path": "__0"}), "not a valid Zarr node name"),
    (well({"path": "0"}, {"path": "0", "acquisition": 1}), "must be unique"),
]


@pytest.mark.parametrize("obj, msg", INVALID_WELLS)
def test_invalid_wells(obj: dict, msg: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(msg)):
        v06.Well.model_validate(obj)


def test_prerelease_version_warns() -> None:
    with pytest.warns(ValidationWarning, match="pre-release"):
        v06.Well.model_validate({**well({"path": "0"}), "version": "0.6rc0"})
