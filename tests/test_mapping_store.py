"""Tests for validating stores that are plain `Mapping[str, bytes]` objects."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import pytest

from yaozarrs import open_group, validate_zarr_store

if TYPE_CHECKING:
    from typing import Callable


class LazyStore(Mapping[str, bytes]):
    """A mapping that records every key it's asked for, with no `getitems`."""

    def __init__(self, data: Mapping[str, bytes]) -> None:
        self._data = data
        self.requested: list[str] = []

    def __getitem__(self, key: str) -> bytes:
        self.requested.append(key)
        return self._data[key]

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

    assert group.store_path == "<LazyStore>"
    assert group["s0"].store_path == "<LazyStore>/s0"
    with pytest.raises(TypeError, match="not backed by a filesystem"):
        group.to_zarr_python()
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
