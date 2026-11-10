from __future__ import annotations

import copy
import importlib.util
import json
import pickle
import shutil
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

try:
    import fsspec
except ImportError:
    pytest.skip("fsspec not installed", allow_module_level=True)

from pydantic import BaseModel

import yaozarrs
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

    with pytest.raises(KeyError):
        _missing = cached["missing"]


def test_cached_mapper_does_not_cache_default(memory_mapper: fsspec.FSMap) -> None:
    cached = _CachedMapper(memory_mapper)
    assert cached.get("missing", b"fallback") == b"fallback"
    assert cached.get("missing") is None
    assert "missing" not in cached


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

    monkeypatch.setattr(cached._mapper, "getitems", boom)

    if get_raises:

        def bad_get(key, default=None):
            raise FileNotFoundError("nope")

        monkeypatch.setattr(cached._mapper, "get", bad_get)

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
            "array/.zattrs",
            "group_child/.zgroup",
            "group_child/.zarray",
            "group_child/.zattrs",
            "missing/.zgroup",
            "missing/.zarray",
            "missing/.zattrs",
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


@pytest.mark.parametrize("roundtrip", ["pickle", "deepcopy"])
def test_zarr_nodes_roundtrip(v2_store: Path, roundtrip: str) -> None:
    def _rt(obj: object) -> object:
        if roundtrip == "pickle":
            return pickle.loads(pickle.dumps(obj))
        return copy.deepcopy(obj)

    group = open_group(v2_store)
    group.ome_metadata()  # populate the `_ome_metadata` slot
    group2 = _rt(group)
    assert isinstance(group2, ZarrGroup)
    assert group2.path == group.path
    assert group2.store_path == group.store_path
    assert group2.metadata == group.metadata
    assert group2.ome_metadata() == group.ome_metadata()
    # the shared metadata cache is not carried over
    assert group._store._cache
    assert not group2._store._cache
    assert isinstance(group2["array"], ZarrArray)

    arr = group["array"]
    arr2 = _rt(arr)
    assert isinstance(arr2, ZarrArray)
    assert arr2.path == arr.path
    assert arr2.metadata == arr.metadata


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


@pytest.mark.parametrize("converter", ["to_zarr_python", "to_tensorstore"])
def test_path_with_spaces(
    write_demo_ome: Callable, converter: str, tmp_path: Path
) -> None:
    """to_zarr_python/to_tensorstore should work when the path has spaces."""
    if converter == "to_zarr_python":
        pytest.importorskip("zarr")
    else:
        pytest.importorskip("tensorstore")

    src = write_demo_ome("image", version="0.4")
    dest = tmp_path / "path with spaces" / "store.zarr"
    shutil.copytree(src, dest)

    group = open_group(dest)
    array = group["s0"]
    result = getattr(array, converter)()
    assert result is not None


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
    array = group["s0"]
    assert isinstance(array, ZarrArray)
    assert f"{path.name}/s0" in repr(array)
    result = array.to_tensorstore()
    assert isinstance(result, ts.TensorStore)


def test_open_group_raises_when_root_not_group() -> None:
    store = "memory://not_group.zarr"
    mapper = fsspec.get_mapper(store)
    mapper["zarr.json"] = json.dumps({"zarr_format": 3, "node_type": "array"}).encode()
    with pytest.raises(ValueError, match="Expected root node to be 'group'"):
        open_group(store)


@pytest.mark.parametrize("type", ["image", "plate", "labels"])
@pytest.mark.parametrize("version", ["0.5", "0.4"])
def test_zarrgroup_tree(write_demo_ome: Callable, version: str, type: str) -> None:
    path = write_demo_ome(type, version=version)
    group = open_group(path)

    # Basic tree output
    tree_str = group.tree()
    assert path.name in tree_str
    assert any(icon in tree_str for icon in ("📦", "🖼️", "🏷️", "🅾️", "📁"))

    # Test depth limiting
    tree_depth1 = group.tree(depth=1)
    tree_depth3 = group.tree(depth=3)
    # Deeper trees should generally be longer or equal
    assert len(tree_depth1) <= len(tree_depth3)

    # Test max_per_level
    tree_limited = group.tree(max_per_level=1)
    if "⋯" in tree_limited:
        # Truncation indicator should appear if there are multiple children
        pass


def test_zarrgroup_tree_plain_text(write_demo_ome: Callable) -> None:
    """Test plain text fallback when rich is not available."""
    from yaozarrs._tree import _build_tree, _render_plain

    path = write_demo_ome("image", version="0.5")
    group = open_group(path)

    # Build tree data and render to plain text
    tree_data = _build_tree(group, depth=2, max_per_level=None)
    parts = _render_plain(tree_data)
    tree_str = "\n".join(parts)
    assert path.name in tree_str
    # Plain text uses box drawing characters
    assert "├──" in tree_str or "└──" in tree_str


def _assert_layout_matches_zarr(y_arr: ZarrArray, z_arr: Any) -> None:
    for attr in ("shape", "chunks", "shards"):
        try:
            expected = getattr(z_arr, attr)
        except NotImplementedError:
            with pytest.raises(NotImplementedError):
                getattr(y_arr, attr)
        else:
            assert getattr(y_arr, attr) == expected, attr


@pytest.mark.parametrize("ndim", range(1, 6))
@pytest.mark.parametrize("layout", ["v2", "v3", "v3-sharded"])
def test_zarrarray_chunk_layout(tmp_path: Path, layout: str, ndim: int) -> None:
    zarr = pytest.importorskip("zarr")
    shape = (10, 13, 7, 6, 5)[:ndim]
    chunks = (2, 3, 4, 2, 5)[:ndim]
    shards = tuple(2 * c for c in chunks) if layout == "v3-sharded" else None
    fmt = 2 if layout == "v2" else 3
    root = zarr.open_group(tmp_path / "a.zarr", mode="w", zarr_format=fmt)
    z_arr = root.create_array(
        "a", shape=shape, chunks=chunks, shards=shards, dtype="u2"
    )

    y_arr = yaozarrs.open_group(tmp_path / "a.zarr")["a"]
    assert isinstance(y_arr, yaozarrs.ZarrArray)
    assert (y_arr.shape, y_arr.chunks, y_arr.shards) == (shape, chunks, shards)
    _assert_layout_matches_zarr(y_arr, z_arr)

    if importlib.util.find_spec("tensorstore"):
        ts_layout = y_arr.to_tensorstore().chunk_layout
        assert ts_layout.read_chunk.shape == y_arr.chunks
        assert ts_layout.write_chunk.shape == (y_arr.shards or y_arr.chunks)


@pytest.mark.filterwarnings("ignore:Combining a `sharding_indexed` codec")
@pytest.mark.parametrize(
    "case", ["nested", "transpose", "rectilinear", "rectilinear-sharded"]
)
def test_zarrarray_chunk_layout_edge_cases(tmp_path: Path, case: str) -> None:
    """Unusual v3 layouts report the same chunks/shards as zarr-python (or raise)."""
    zarr = pytest.importorskip("zarr")
    codecs = zarr.codecs
    kwargs: dict = {"shape": (16, 16), "dtype": "u1", "chunks": (8, 8)}
    if case == "nested":  # outermost shard, and the chunks directly inside it
        inner = codecs.ShardingCodec(chunk_shape=(2, 2))
        kwargs["serializer"] = codecs.ShardingCodec(chunk_shape=(4, 4), codecs=[inner])
    elif case == "transpose":  # zarr-python only treats [sharding_indexed] as sharded
        kwargs["filters"] = [codecs.TransposeCodec(order=(1, 0))]
        kwargs["serializer"] = codecs.ShardingCodec(chunk_shape=(4, 2))
    elif case == "rectilinear":
        kwargs["chunks"] = [[6, 10], [16]]
    else:
        kwargs.update(chunks=(2, 4), shards=[[6, 10], [16]])
    rectilinear = case.startswith("rectilinear")
    if rectilinear and zarr.config.get("array.rectilinear_chunks", None) is None:
        pytest.skip("zarr-python version lacks rectilinear chunk grids")

    with zarr.config.set({"array.rectilinear_chunks": rectilinear}):
        root = zarr.open_group(tmp_path / "a.zarr", mode="w")
        z_arr = root.create_array("a", compressors=None, **kwargs)
        y_arr = yaozarrs.open_group(tmp_path / "a.zarr")["a"]
        assert isinstance(y_arr, yaozarrs.ZarrArray)
        _assert_layout_matches_zarr(y_arr, z_arr)
