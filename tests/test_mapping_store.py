"""Tests for validating stores that are plain `Mapping[str, bytes]` objects."""

from __future__ import annotations

import json
import posixpath
import sys
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import pytest

import yaozarrs
from yaozarrs import DimSpec, open_group, v06, validate_zarr_store
from yaozarrs._errors import StorageValidationError, StorageValidationWarning

if TYPE_CHECKING:
    from collections.abc import Callable


class LazyStore(Mapping[str, bytes]):
    """A mapping that records every key it's asked for, with no `getitems`."""

    def __init__(self, data: Mapping[str, bytes]) -> None:
        self.store = data  # (not to be confused with a zarr-python Group's `store`)
        self.requested: list[str] = []

    def __getitem__(self, key: str) -> bytes:
        self.requested.append(key)
        return self.store[key]

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def test_mapping_store_without_fsspec(
    write_demo_ome: Callable, metadata_dict: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LazyStore(metadata_dict(write_demo_ome("image", version="0.5")))
    monkeypatch.setitem(sys.modules, "fsspec", None)

    group = validate_zarr_store(store)
    # only metadata documents are read, never chunks
    assert {key.rpartition("/")[-1] for key in store.requested} == {"zarr.json"}

    assert repr(group["s0"]) == "<ZarrArray <LazyStore>/s0>"
    with pytest.raises(TypeError, match="not backed by a filesystem"):
        group.store_path  # noqa: B018
    with pytest.raises(ImportError, match="fsspec package is required"):
        open_group("some/path.zarr")


def test_series_parent_in_mapping(
    write_demo_ome: Callable, metadata_dict: Callable
) -> None:
    path = write_demo_ome("bioformats2raw", write_ome_group=True)
    # the parent of a nested OME group is in the store...
    validate_zarr_store(open_group(metadata_dict(path))["OME"])
    # ...but the parent of the root is not
    with pytest.warns(UserWarning, match="Cannot validate series paths"):
        validate_zarr_store(metadata_dict(path / "OME"))


@pytest.mark.parametrize("source", ["../../", "/abs/image.zarr", "https://a.b/c.zarr"])
def test_label_source_outside_of_mapping(
    source: str, write_demo_ome: Callable, metadata_dict: Callable
) -> None:
    path = write_demo_ome("image-with-labels", version="0.5")
    meta_file = path / "labels" / "annotations" / "zarr.json"
    meta = json.loads(meta_file.read_text())
    meta["attributes"]["ome"]["image-label"]["source"] = {"image": source}
    meta_file.write_text(json.dumps(meta))

    # (relative to labels/annotations, "../../" is the image: outside of `labels`)
    with pytest.warns(UserWarning, match="it is outside of the store"):
        validate_zarr_store(metadata_dict(path / "labels"))


def _group(attrs: dict) -> bytes:
    doc = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": attrs}}
    return json.dumps(doc).encode()


def _array(dtype: str) -> bytes:
    doc = {
        **{"zarr_format": 3, "node_type": "array", "shape": [4, 4]},
        **{"data_type": dtype, "fill_value": 0, "codecs": [{"name": "bytes"}]},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4, 4]}},
        "chunk_key_encoding": {"name": "default"},
        "dimension_names": ["y", "x"],
    }
    return json.dumps(doc).encode()


@pytest.mark.parametrize("version", ["0.5", "0.6"])
def test_label_dtype_checked_without_image_label(version: str) -> None:
    # a path listed in `labels` is a label image, with or without `image-label`
    ms = getattr(yaozarrs, f"v0{version[-1]}").Multiscale.from_dims(
        [DimSpec(name="y", size=4), DimSpec(name="x", size=4)]
    )
    ome = {"version": version, "multiscales": [ms.model_dump(mode="json")]}
    store = {
        "zarr.json": _group(ome),
        "0/zarr.json": _array("uint16"),
        "labels/zarr.json": _group({"version": version, "labels": ["cells"]}),
        "labels/cells/zarr.json": _group(ome),  # (no image-label)
        "labels/cells/0/zarr.json": _array("float32"),
    }
    with pytest.warns(UserWarning, match="SHOULD contain 'image-label'"):
        with pytest.raises(StorageValidationError, match="label_non_integer_dtype"):
            validate_zarr_store(store)


class HttpLikeStore(dict):
    """Fetches keys in batches, and resolves `.` in them, as http stores do."""

    round_trips = 0  # (each key fetched alone, or each batch of keys)

    def get(self, key: Any, default: Any = None) -> Any:
        self.round_trips += 1
        return dict.get(self, posixpath.normpath(key), default)

    def getitems(self, keys: list[str], on_error: str = "omit") -> dict[str, Any]:
        self.round_trips += 1
        if self.round_trips > 50:  # (a runaway: say every key is missing, to stop it)
            return {key: KeyError(key) for key in keys}
        return {k: dict.get(self, posixpath.normpath(k), KeyError(k)) for k in keys}


def _image(version: str = "0.6") -> dict:
    ms = v06.Multiscale.from_dims([DimSpec(name="y"), DimSpec(name="x")])
    return {"version": version, "multiscales": [ms.model_dump(mode="json")]}


def test_prefetch_stops_at_self_reference() -> None:
    labels = {"version": "0.6", "labels": ["."]}  # (lists itself)
    store = HttpLikeStore(
        {"zarr.json": _group(_image()), "labels/zarr.json": _group(labels)}
    )
    with pytest.raises(StorageValidationError, match="label_image_invalid"):
        validate_zarr_store(store)
    open_group(store).tree()
    assert store.round_trips < 10


def test_prefetch_leaves_errors_to_validation() -> None:
    # (an image where a well should be, with a malformed document below it)
    well = {"path": "A/1", "rowIndex": 0, "columnIndex": 0}
    plate = {"rows": [{"name": "A"}], "columns": [{"name": "1"}], "wells": [well]}
    store = HttpLikeStore(
        {
            "zarr.json": _group({"version": "0.6", "plate": plate}),
            "A/1/zarr.json": _group(_image()),
            "A/1/labels/zarr.json": b"{bad",
        }
    )
    with pytest.raises(StorageValidationError, match="well_invalid"):
        validate_zarr_store(store)


def test_prefetched_documents_validated_once() -> None:
    # ("0.6.dev4" warns once per document: the image, its labels group, its label)
    image = _image("0.6.dev4")
    store = HttpLikeStore(
        {
            "zarr.json": _group(image),
            "0/zarr.json": _array("uint8"),
            "labels/zarr.json": _group({"version": "0.6.dev4", "labels": ["cells"]}),
            "labels/cells/zarr.json": _group({**image, "image-label": {}}),
            "labels/cells/0/zarr.json": _array("uint8"),
        }
    )
    with pytest.warns(StorageValidationWarning) as record:
        validate_zarr_store(store)
    assert len(record.pop(StorageValidationWarning).message.warnings()) == 3


def _scene(n: int) -> dict[str, bytes]:
    """A v0.6 scene of `n` 2D images, `t0`, `t1`, ..."""
    world = {"name": "world", "axes": [{"name": ax, "type": "space"} for ax in "yx"]}
    refs = [
        {"type": "identity", "input": {"path": f"t{i}", "name": "intrinsic"}}
        | {"output": {"name": "world"}}
        for i in range(n)
    ]
    scene = {"coordinateSystems": [world], "coordinateTransformations": refs}
    store = {"zarr.json": _group({"version": "0.6", "scene": scene})}
    for i in range(n):
        store |= {
            f"t{i}/zarr.json": _group(_image()),
            f"t{i}/0/zarr.json": _array("u2"),
        }
    return store


# (on stores that batch: one fetch per level of the hierarchy, not one per group)
@pytest.mark.parametrize("action", ["validate", "tree"])
@pytest.mark.parametrize("kind", ["plate", "bioformats2raw", "scene"])
def test_round_trips(
    action: str, kind: str, request: Any, write_demo_ome: Callable, metadata_dict: Any
) -> None:
    if kind == "plate":  # (with labels)
        data = metadata_dict(request.getfixturevalue("complex_ome_zarr"))
    elif kind == "bioformats2raw":
        data = metadata_dict(write_demo_ome(kind, num_series=40))
    else:
        data = _scene(40)
    store = HttpLikeStore(data)
    if action == "validate":
        validate_zarr_store(store)
    else:
        open_group(store).tree()
    assert store.round_trips < 10


def test_v2_documents_fetched_together(
    write_demo_ome: Callable, metadata_dict: Callable
) -> None:
    store = HttpLikeStore(metadata_dict(write_demo_ome("image", version="0.4")))
    image = open_group(store)
    datasets = image.ome_metadata().multiscales[0].datasets
    arrays = [image[ds.path] for ds in datasets]
    assert store.round_trips == 2 + len(arrays)  # (zarr.json, then one per node)


def test_tree_follows_series_list(
    write_demo_ome: Callable, metadata_dict: Callable
) -> None:
    # (in a store that can't be listed: more than 100 images, not numbered)
    image = metadata_dict(write_demo_ome("bioformats2raw"))
    names = [f"s{i}" for i in range(120)]
    store = {"zarr.json": image["zarr.json"]}
    store |= {
        f"{n}/{k[2:]}": v for n in names for k, v in image.items() if k[:2] == "0/"
    }
    store["OME/zarr.json"] = _group({"version": "0.5", "series": names})
    assert open_group(store).tree(depth=1).count("🖼️") == 120


def test_tree_shows_group_with_invalid_metadata() -> None:
    store = _scene(2) | {"t1/zarr.json": _group({"version": "0.6", "multiscales": []})}
    assert "📁 t1" in open_group(store).tree()
