"""Tests for validating stores that are plain `Mapping[str, bytes]` objects."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import pytest

import yaozarrs
from yaozarrs import DimSpec, open_group, validate_zarr_store
from yaozarrs._storage import StorageValidationError

if TYPE_CHECKING:
    from typing import Callable


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


@pytest.mark.parametrize("version", ["0.5", "0.6"])
def test_label_dtype_checked_without_image_label(version: str) -> None:
    # a path listed in `labels` is a label image, with or without `image-label`
    ms = getattr(yaozarrs, f"v0{version[-1]}").Multiscale.from_dims(
        [DimSpec(name="y", size=4), DimSpec(name="x", size=4)]
    )
    ome = {"version": version, "multiscales": [ms.model_dump(mode="json")]}

    def group(attrs: dict) -> bytes:
        doc = {"zarr_format": 3, "node_type": "group", "attributes": {"ome": attrs}}
        return json.dumps(doc).encode()

    def array(dtype: str) -> bytes:
        doc = {
            **{"zarr_format": 3, "node_type": "array", "shape": [4, 4]},
            **{"data_type": dtype, "fill_value": 0, "codecs": [{"name": "bytes"}]},
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4, 4]}},
            "chunk_key_encoding": {"name": "default"},
            "dimension_names": ["y", "x"],
        }
        return json.dumps(doc).encode()

    store = {
        "zarr.json": group(ome),
        "0/zarr.json": array("uint16"),
        "labels/zarr.json": group({"version": version, "labels": ["cells"]}),
        "labels/cells/zarr.json": group(ome),  # (no image-label)
        "labels/cells/0/zarr.json": array("float32"),
    }
    with pytest.warns(UserWarning, match="SHOULD contain 'image-label'"):
        with pytest.raises(StorageValidationError, match="label_non_integer_dtype"):
            validate_zarr_store(store)
