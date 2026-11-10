"""`yaozarrs.bioformats2raw_series`: the image paths of a bioformats2raw collection."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import pytest

import yaozarrs
from yaozarrs import DimSpec, v05

if TYPE_CHECKING:
    from pathlib import Path

BF2RAW = {"version": "0.5", "bioformats2raw.layout": 3}


def _group(ome: dict[str, Any] | None = None) -> bytes:
    doc: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}
    if ome is not None:
        doc["attributes"] = {"ome": ome}
    return json.dumps(doc).encode()


def _v05_store(
    series: list[str] | None = None, numbered: int = 0, ome: Any = None
) -> dict[str, bytes]:
    store = {"zarr.json": _group(BF2RAW)}
    if series is not None:
        ome = {"version": "0.5", "series": series}
    if ome is not None:
        store["OME/zarr.json"] = _group(ome)
    for i in range(numbered):
        store[f"{i}/zarr.json"] = _group()
    return store


def test_series_list() -> None:
    store = _v05_store(series=["b", "a/nested"], numbered=3)
    assert yaozarrs.bioformats2raw_series(store) == ["b", "a/nested"]


def _v04_store(numbered: int = 0) -> dict[str, bytes]:
    zgroup = json.dumps({"zarr_format": 2}).encode()
    store = {".zgroup": zgroup, ".zattrs": b'{"bioformats2raw.layout": 3}'}
    for i in range(numbered):
        store[f"{i}/.zgroup"] = zgroup
    return store


# (batches of 16, 32, 64...: 16 and 48 are the edges of the first two)
@pytest.mark.parametrize("n", [1, 16, 17, 48, 49])
@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_numbered_children(n: int, version: str) -> None:
    # (no `series` list: the numbered children, up to the first missing number)
    if version == "0.4":
        store = _v04_store(numbered=n)
        store[f"{n + 1}/.zgroup"] = store[".zgroup"]  # after a gap: not a series
    else:
        store = _v05_store(numbered=n)
        store[f"{n + 1}/zarr.json"] = _group()  # after a gap: not a series
    assert yaozarrs.bioformats2raw_series(store) == [str(i) for i in range(n)]


class _CountingStore(dict):
    """A store that records the size of each batched fetch."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.batches: list[int] = []

    def getitems(self, keys: list[str], on_error: str = "omit") -> dict[str, Any]:
        self.batches.append(len(keys))
        return {k: self[k] if k in self else KeyError(k) for k in keys}


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_numbered_children_batches_double(version: str) -> None:
    store = _CountingStore(
        _v04_store(100) if version == "0.4" else _v05_store(None, 100)
    )
    assert len(yaozarrs.bioformats2raw_series(store)) == 100
    # (after the `OME` lookup: v2 children have 3 metadata keys each, v3 have 1)
    per_child = 3 if version == "0.4" else 1
    assert store.batches == [per_child * b for b in (1, 16, 32, 64)]


@pytest.mark.parametrize("ome", [None, {"version": "0.5"}])
def test_ome_group_without_series(ome: dict | None) -> None:
    # (`series` is optional in the OME group: without it, use the numbered children)
    store = _v05_store(numbered=2)
    store["OME/zarr.json"] = _group(ome)
    assert yaozarrs.bioformats2raw_series(store) == ["0", "1"]


def _plate_store(bf2raw: bool = True) -> dict[str, bytes]:
    plate: dict[str, Any] = {
        "version": "0.5",
        "plate": {
            "columns": [{"name": "1"}, {"name": "2"}],
            "rows": [{"name": "A"}],
            "wells": [
                {"path": "A/2", "rowIndex": 0, "columnIndex": 1},
                {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
            ],
        },
    }
    if bf2raw:
        plate["bioformats2raw.layout"] = 3
    store = {"zarr.json": _group(plate)}
    for well, fovs in [("A/1", ["0"]), ("A/2", ["0", "1"])]:
        images = [{"path": f} for f in fovs]
        store[f"{well}/zarr.json"] = _group(
            {"version": "0.5", "well": {"images": images}}
        )
        for fov in fovs:
            store[f"{well}/{fov}/zarr.json"] = _group()
    store["0/zarr.json"] = _group()  # (ignored: a plate's images are in its wells)
    return store


def test_plate() -> None:
    store = _plate_store()
    assert yaozarrs.bioformats2raw_series(store) == ["A/2/0", "A/2/1", "A/1/0"]


def test_plate_with_series() -> None:
    # (a series list of the plate's images gives their order)
    store = _plate_store()
    store["OME/zarr.json"] = _group(
        {"version": "0.5", "series": ["A/1/0", "A/2/0", "A/2/1"]}
    )
    assert yaozarrs.bioformats2raw_series(store) == ["A/1/0", "A/2/0", "A/2/1"]


@pytest.mark.parametrize(
    "series", [["A/1/0", "A/2/0"], ["A/1/0", "A/2/0", "A/2/1", "B/1/0"], "invalid"]
)
def test_plate_with_other_series(series: Any) -> None:
    # (the plate takes precedence over a series list that doesn't match it)
    store = _plate_store()
    store["OME/zarr.json"] = _group({"version": "0.5", "series": series})
    assert yaozarrs.bioformats2raw_series(store) == ["A/2/0", "A/2/1", "A/1/0"]


def test_plate_missing_well_raises() -> None:
    store = _plate_store()
    del store["A/1/zarr.json"]
    with pytest.raises(ValueError, match="'A/1' has no well metadata"):
        yaozarrs.bioformats2raw_series(store)


def test_plate_not_a_collection() -> None:
    with pytest.raises(ValueError, match="not a bioformats2raw collection"):
        yaozarrs.bioformats2raw_series(_plate_store(bf2raw=False))


def test_invalid_ome_metadata_raises() -> None:
    store = _v05_store(ome={"version": "0.5", "series": "not-a-list"}, numbered=1)
    with pytest.raises(ValueError):
        yaozarrs.bioformats2raw_series(store)


def test_not_a_collection() -> None:
    store = {"zarr.json": _group({"version": "0.5", "series": ["0"]})}
    with pytest.raises(ValueError, match="not a bioformats2raw collection"):
        yaozarrs.bioformats2raw_series(store)


def test_v04() -> None:
    store = {
        ".zgroup": json.dumps({"zarr_format": 2}).encode(),
        ".zattrs": json.dumps({"bioformats2raw.layout": 3}).encode(),
        "OME/.zgroup": json.dumps({"zarr_format": 2}).encode(),
        "OME/.zattrs": json.dumps({"series": ["x", "y"]}).encode(),
    }
    assert yaozarrs.bioformats2raw_series(store) == ["x", "y"]


def test_v04_plate() -> None:
    plate = {
        "version": "0.4",
        "columns": [{"name": "1"}],
        "rows": [{"name": "A"}],
        "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
    }
    well = {"version": "0.4", "images": [{"path": "0"}]}
    store = {
        ".zgroup": json.dumps({"zarr_format": 2}).encode(),
        ".zattrs": json.dumps({"bioformats2raw.layout": 3, "plate": plate}).encode(),
        "OME/.zgroup": json.dumps({"zarr_format": 2}).encode(),
        "OME/.zattrs": b"{}",  # (no `series`: use the plate)
        "A/1/.zgroup": json.dumps({"zarr_format": 2}).encode(),
        "A/1/.zattrs": json.dumps({"well": well}).encode(),
    }
    assert yaozarrs.bioformats2raw_series(store) == ["A/1/0"]


def test_written_collection(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    w05 = pytest.importorskip("yaozarrs.write.v05")
    dims = [DimSpec(name="y"), DimSpec(name="x")]
    image = v05.Image(multiscales=[v05.Multiscale.from_dims(dims)])
    dest = tmp_path / "c.zarr"
    builder = w05.Bf2RawBuilder(dest)
    for name in ["0", "1"]:
        builder.write_image(name, image, np.zeros((4, 4), np.uint8))
    assert yaozarrs.bioformats2raw_series(dest) == ["0", "1"]

    shutil.rmtree(dest / "OME")  # (numbered series: still a valid collection)
    assert yaozarrs.bioformats2raw_series(dest) == ["0", "1"]
    yaozarrs.validate_zarr_store(dest)
