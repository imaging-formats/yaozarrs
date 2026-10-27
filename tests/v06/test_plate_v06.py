from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_bf2raw_and_series() -> None:
    b = v06.Bf2Raw.model_validate({"version": "0.6", "bioformats2raw.layout": 3})
    assert b.model_dump(mode="json") == {"version": "0.6", "bioformats2raw.layout": 3}
    s = v06.Series.model_validate({"version": "0.6", "series": ["0", "1"]})
    assert s.series == ["0", "1"]
    with pytest.raises(ValidationError, match="at least 1 item"):
        v06.Series.model_validate({"version": "0.6", "series": []})


def test_prerelease_version_warns() -> None:
    with pytest.warns(ValidationWarning, match="pre-release"):
        v06.Well.model_validate({**well({"path": "0"}), "version": "0.6rc0"})


# ------------------------------------------------------------------------------
# upstream conformance fixtures (plate and well documents)
# ------------------------------------------------------------------------------

FIXTURES = (
    Path(__file__).parent.parent / "data" / "v06" / "spec" / "tests" / "attributes"
)


def _fixtures(key: str) -> list[Path]:
    return sorted(
        p
        for p in FIXTURES.rglob("*.json")
        if key in json.loads(p.read_text()).get("ome", {})
        # (strict/invalid fixtures are only invalid under the strict schemas, which
        # yaozarrs doesn't model)
        and "strict/invalid" not in str(p)
    )


# fixtures where yaozarrs deliberately disagrees with upstream's verdict
_SWAPPED = (
    "fixture bug: rows and columns are swapped (rows ['1'], columns ['A'], well path "
    "'A/1'), but \"The path MUST consist of a name in the rows array, a file separator "
    '(/), and a name from the columns array, in that order" (upstream_fixes.md #20)'
)
EXPECTED_DEVIATIONS: dict[str, str] = {
    "spec/valid/plate/minimal_acquisitions.json": _SWAPPED,
    "spec/valid/plate/minimal_no_acquisitions.json": _SWAPPED,
    "spec/valid/plate/non_alphanumeric_row.json": _SWAPPED,
    "strict/valid/plate/strict_acquisitions.json": _SWAPPED,
    "strict/valid/plate/strict_no_acquisitions.json": _SWAPPED,
}


@pytest.mark.parametrize(
    "path, model",
    [(p, v06.Plate) for p in _fixtures("plate")]
    + [(p, v06.Well) for p in _fixtures("well")],
    ids=lambda x: str(x.relative_to(FIXTURES)) if isinstance(x, Path) else "",
)
def test_upstream_plate_well_fixtures(path: Path, model: type) -> None:
    upstream_valid = "/valid/" in str(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidationWarning)
        try:
            model.model_validate(json.loads(path.read_text())["ome"])
            valid = True
        except ValidationError:
            valid = False
    if (key := str(path.relative_to(FIXTURES))) in EXPECTED_DEVIATIONS:
        assert valid != upstream_valid, f"deviation no longer needed: {key}"
    else:
        assert valid == upstream_valid
