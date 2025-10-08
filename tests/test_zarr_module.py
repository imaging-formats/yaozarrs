from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable
from unittest.mock import patch

import pytest

try:
    import fsspec
except ImportError:
    pytest.skip("fsspec not installed", allow_module_level=True)

from pydantic import BaseModel

from yaozarrs._zarr import (
    ZarrArray,
    ZarrGroup,
    ZarrMetadata,
    ZarrNode,
    _CachedMapper,
    open_group,
)

if TYPE_CHECKING:
    from pathlib import Path


def _setup_memory_v3_store(store_name: str) -> None:
    mapper = fsspec.get_mapper(store_name)
    mapper["zarr.json"] = json.dumps({"zarr_format": 3, "node_type": "group"}).encode()
    mapper["child/zarr.json"] = json.dumps(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [1, 2, 3],
            "data_type": "<i2",
        }
    ).encode()
    mapper["group_child/zarr.json"] = json.dumps(
        {"zarr_format": 3, "node_type": "group"}
    ).encode()
    mapper["unknown/zarr.json"] = json.dumps(
        {
            "zarr_format": 3,
            "node_type": "mystery",
        }
    ).encode()


# ------------------------


@pytest.fixture
def memory_mapper(tmp_path: Path) -> fsspec.FSMap:
    mapper = fsspec.get_mapper(f"memory://{tmp_path.name}.zarr")
    return mapper


@pytest.fixture
def v2_store(tmp_path: Path) -> Path:
    """Create a local v2 zarr store with test data."""
    root = tmp_path / "store.zarr"

    root.mkdir(exist_ok=True)

    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(json.dumps({"name": "local"}))

    array_path = root / "array"
    array_path.mkdir()
    array_meta = {"zarr_format": 2, "chunks": [2, 2], "dtype": "<i4", "shape": [2, 2]}
    (array_path / ".zarray").write_text(json.dumps(array_meta))
    (array_path / ".zattrs").write_text(json.dumps({"kind": "array"}))

    group_child_path = root / "group_child"
    group_child_path.mkdir()
    (group_child_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (group_child_path / ".zattrs").write_text(json.dumps({"kind": "group"}))
    return root


@pytest.fixture
def v3_memory_store(tmp_path: Path) -> str:
    """Create a v3 zarr store in memory with test data."""
    store_name = f"memory://{tmp_path.name}_v3.zarr"
    _setup_memory_v3_store(store_name)
    return store_name


def test_cached_mapper_caches_values_and_exceptions(
    memory_mapper: fsspec.FSMap,
) -> None:
    memory_mapper["value"] = b"first"
    cached = _CachedMapper(memory_mapper)

    assert cached.get("value") == b"first"
    memory_mapper["value"] = b"second"
    assert cached.get("value") == b"first"

    missing = cached.getitems(["missing"])
    assert "missing" in missing
    with pytest.raises(KeyError):
        cached.get("missing")


def test_cached_mapper_contains_len_iter_and_getitem(
    memory_mapper: fsspec.FSMap,
) -> None:
    memory_mapper["exists"] = b"data"
    cached = _CachedMapper(memory_mapper)

    assert "exists" in cached
    assert 123 not in cached
    assert "missing" not in cached
    assert list(cached) == list(memory_mapper)
    assert len(cached) == len(memory_mapper)
    assert cached["exists"] == b"data"
    with pytest.raises(KeyError):
        _ = cached["missing"]

    assert cached.fs is memory_mapper.fs


@pytest.mark.parametrize(
    "setup_key,get_raises,on_error_arg,expected",
    [
        ("key", False, None, {"key": b"value"}),
        (None, True, "raise", FileNotFoundError),
        (None, True, None, {}),
    ],
    ids=["fallback", "on_error_raise", "on_error_omit"],
)
def test_cached_mapper_getitems_error_handling(
    monkeypatch: pytest.MonkeyPatch,
    memory_mapper: fsspec.FSMap,
    setup_key: str | None,
    get_raises: bool,
    on_error_arg: str | None,
    expected: dict | type[Exception],
) -> None:
    if setup_key:
        memory_mapper[setup_key] = b"value"

    cached = _CachedMapper(memory_mapper)

    def boom(keys, on_error=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(cached._fsmap, "getitems", boom)

    if get_raises:

        def bad_get(key, default=None):
            raise FileNotFoundError("nope")

        monkeypatch.setattr(cached._fsmap, "get", bad_get)

    key = setup_key if setup_key else "missing"
    kwargs = {"on_error": on_error_arg} if on_error_arg else {}

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            cached.getitems([key], **kwargs)
    else:
        result = cached.getitems([key], **kwargs)
        assert result == expected


def test_zarrnode_wraps_plain_fsmap(tmp_path: Path) -> None:
    store = f"memory://{tmp_path.name}_wrap.zarr"
    mapper = fsspec.get_mapper(store)
    mapper["zarr.json"] = json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {}}
    ).encode()

    node = ZarrNode(mapper)
    assert isinstance(node._store, _CachedMapper)


def test_zarrnode_loads_v2_array_metadata(tmp_path: Path) -> None:
    store = f"memory://{tmp_path.name}_v2root.zarr"
    mapper = fsspec.get_mapper(store)
    mapper[".zarray"] = json.dumps(
        {
            "zarr_format": 2,
            "dtype": "<u2",
            "shape": [3],
            "chunks": [3],
            "compressor": None,
            "fill_value": 0,
            "order": "C",
            "filters": None,
        }
    ).encode()
    mapper[".zattrs"] = json.dumps({"key": "value"}).encode()

    node = ZarrNode(_CachedMapper(mapper))
    assert node.zarr_format == 2
    assert node.attrs["key"] == "value"
    array = ZarrArray(node._store, node.path, node._metadata)
    assert array.ndim == 1
    assert isinstance(node.store, Mapping)


def test_zarrgroup_v3_behaviour(v3_memory_store: str) -> None:
    group = open_group(v3_memory_store)
    assert isinstance(group, ZarrGroup)
    assert isinstance(group.metadata, ZarrMetadata)
    assert "ZarrGroup" in repr(group)
    assert group.zarr_format == 3

    with patch.object(group._store, "getitems") as m:
        group.prefetch_children(["child"])
    m.assert_called_once_with(["child/zarr.json"])

    assert "child" in group
    child = group["child"]
    assert isinstance(child, ZarrArray)
    assert child.path == "child"
    assert child.ndim == 3
    assert child.dtype == "<i2"

    subgroup = group["group_child"]
    assert isinstance(subgroup, ZarrGroup)

    with pytest.raises(ValueError):
        _ = group["unknown"]

    assert group.get("missing", "sentinel") == "sentinel"


def test_zarrgroup_v2_behaviour(v2_store: Path) -> None:
    group = open_group(v2_store)
    assert isinstance(group, ZarrGroup)
    assert group.attrs == {"name": "local"}

    with patch.object(group._store, "getitems") as m:
        group.prefetch_children(["array", "group_child", "missing"])
    m.assert_called_once_with(
        [
            "array/.zgroup",
            "array/.zarray",
            "group_child/.zgroup",
            "group_child/.zarray",
            "missing/.zgroup",
            "missing/.zarray",
        ]
    )

    assert "array" in group
    arr = group["array"]
    assert isinstance(arr, ZarrArray)
    assert arr.path == "array"
    assert arr.ndim == 2
    assert arr.dtype == "<i4"
    assert group.get("missing") is None
    with pytest.raises(KeyError):
        _ = group["missing"]

    subgroup = group["group_child"]
    assert isinstance(subgroup, ZarrGroup)
    assert subgroup.attrs["kind"] == "group"


def test_zarrgroup_prefetch_handles_exceptions(v2_store: Path) -> None:
    group = open_group(v2_store)
    with patch.object(group._store, "getitems", RuntimeError("boom")):
        group.prefetch_children(["array"])


def test_zarrgroup_ome_model_cached(write_demo_ome: Callable) -> None:
    pytest.importorskip("zarr")
    path = write_demo_ome("image", version="0.5")
    group = open_group(path)
    first = group.ome_metadata()
    second = group.ome_metadata()
    assert first is second


def test_zarrgroup_to_zarr_python(v2_store: Path) -> None:
    zarr = pytest.importorskip("zarr")
    group = open_group(v2_store)
    result = group.to_zarr_python()
    assert isinstance(result, zarr.Group)


@pytest.mark.parametrize("type", ["image", "plate", "labels"])
@pytest.mark.parametrize("version", ["0.5", "0.4"])
def test_zarrgroup_from_zarr_python(
    write_demo_ome: Callable, version: str, type: str
) -> None:
    zarr = pytest.importorskip("zarr")
    path = write_demo_ome(type, version=version)
    zarr_group = zarr.open_group(path)
    group = open_group(zarr_group)
    assert isinstance(group, ZarrGroup)
    assert isinstance(group.metadata, ZarrMetadata)
    assert isinstance(group.metadata.ome_metadata(), BaseModel)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_zarrarray_to_tensorstore(write_demo_ome: Callable, version: str) -> None:
    ts = pytest.importorskip("tensorstore")
    path = write_demo_ome("image", version=version)
    group = open_group(path)
    assert path.name in repr(group)
    array = group["0"]
    assert isinstance(array, ZarrArray)
    assert f"{path.name}/0" in repr(array)
    result = array.to_tensorstore()
    assert isinstance(result, ts.TensorStore)


def test_open_group_raises_when_root_not_group() -> None:
    store = "memory://not_group.zarr"
    mapper = fsspec.get_mapper(store)
    mapper["zarr.json"] = json.dumps({"zarr_format": 3, "node_type": "array"}).encode()
    with pytest.raises(ValueError, match="Expected root node to be 'group'"):
        open_group(store)
