"""Tests for validating stores that are plain `Mapping[str, bytes]` objects."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import pytest

from yaozarrs import open_group, validate_zarr_store
from yaozarrs._common._storage import _source_store_key
from yaozarrs._storage import StorageValidationError
from yaozarrs._zarr import ZarrArray, ZarrGroup

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Callable

IMAGE = {
    "version": "0.5",
    "multiscales": [
        {
            "axes": [
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0]}
                    ],
                }
            ],
        }
    ],
}
ARRAY = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [64, 64],
    "data_type": "uint8",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64, 64]}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 0,
    "codecs": [{"name": "bytes"}],
    "dimension_names": ["y", "x"],
}


def _group(ome: dict | None = None) -> dict:
    attrs = {"ome": ome} if ome is not None else {}
    return {"zarr_format": 3, "node_type": "group", "attributes": attrs}


def _encode(docs: dict[str, dict]) -> dict[str, bytes]:
    return {k: json.dumps(v).encode() for k, v in docs.items()}


def _image_store(prefix: str = "") -> dict[str, bytes]:
    return _encode({f"{prefix}zarr.json": _group(IMAGE), f"{prefix}0/zarr.json": ARRAY})


class LazyStore(Mapping[str, bytes]):
    """A mapping that fetches on `__getitem__`, with no `getitems`, `iter` or `len`."""

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


def test_validate_dict_store_without_fsspec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fsspec", None)
    group = validate_zarr_store(_image_store())
    assert isinstance(group, ZarrGroup)
    assert isinstance(group["0"], ZarrArray)
    # a ZarrGroup opened from a mapping can be passed back in, also without fsspec
    assert validate_zarr_store(group) is group

    bad = _image_store()
    del bad["0/zarr.json"]
    with pytest.raises(StorageValidationError, match="dataset_path_not_found"):
        validate_zarr_store(bad)


def test_open_group_uri_still_requires_fsspec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(ImportError, match="fsspec package is required"):
        open_group("some/path.zarr")


def test_lazy_mapping_only_reads_metadata() -> None:
    store = LazyStore(_image_store())
    validate_zarr_store(store)
    assert store.requested
    # every key read is a metadata document, never a chunk
    names = {key.rpartition("/")[-1] for key in store.requested}
    assert names <= {"zarr.json", ".zgroup", ".zattrs", ".zarray"}


def test_open_group_on_mapping_errors() -> None:
    with pytest.raises(ValueError, match="Expected root node to be 'group'"):
        open_group(_encode({"zarr.json": ARRAY}))
    with pytest.raises(FileNotFoundError):
        open_group({})


def test_mapping_store_has_no_uri() -> None:
    group = open_group(_image_store())
    assert group.store_path == "<dict>"
    assert group["0"].store_path == "<dict>/0"
    assert "<dict>" in repr(group)
    with pytest.raises(TypeError, match="not backed by a filesystem"):
        group.to_zarr_python()
    with pytest.raises(TypeError, match="not backed by a filesystem"):
        group["0"].to_tensorstore()  # type: ignore[union-attr]


def test_mapping_store_tree() -> None:
    # no directory listing for a mapping: children come from the OME metadata
    tree = open_group(_image_store()).tree()
    assert "0" in str(tree)


def test_fsmap_is_accepted_as_a_mapping(write_demo_ome: Callable) -> None:
    fsspec = pytest.importorskip("fsspec")
    path = write_demo_ome("image", version="0.5")
    group = open_group(fsspec.get_mapper(str(path)))
    assert group.store_path == path.as_uri()
    validate_zarr_store(group)


def test_v2_metadata_in_mapping() -> None:
    image = {**IMAGE, "version": "0.4"}
    store = _encode(
        {
            ".zgroup": {"zarr_format": 2},
            ".zattrs": image,
            "0/.zarray": {
                "zarr_format": 2,
                "shape": [64, 64],
                "chunks": [64, 64],
                "dtype": "|u1",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
            },
        }
    )
    validate_zarr_store(store)


# ---------------- series parents -----------------


def _bf2raw_store() -> dict[str, bytes]:
    return {
        **_image_store("0/"),
        **_encode(
            {
                "zarr.json": _group({"version": "0.5", "bioformats2raw.layout": 3}),
                "OME/zarr.json": _group({"version": "0.5", "series": ["0"]}),
            }
        ),
    }


def test_series_parent_inside_mapping() -> None:
    store = _bf2raw_store()
    validate_zarr_store(store)
    # the OME group isn't the root, so its parent is reachable
    validate_zarr_store(open_group(store)["OME"])

    del store["0/zarr.json"]
    with pytest.raises(StorageValidationError, match="series_path_not_found"):
        validate_zarr_store(open_group(store)["OME"])


def test_series_parent_outside_mapping_warns() -> None:
    # the OME group is the root of the mapping, so the series paths can't be reached
    store = _encode({"zarr.json": _group({"version": "0.5", "series": ["0"]})})
    with pytest.warns(UserWarning, match="Cannot validate series paths"):
        validate_zarr_store(store)


# ---------------- label sources -----------------


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_label_source_in_mapping(
    version: str, write_demo_ome: Callable, metadata_dict: Callable
) -> None:
    path = write_demo_ome("image-with-labels", version=version)
    _set_label_source(path, version, "../../")

    validate_zarr_store(metadata_dict(path))

    labels = metadata_dict(path / "labels")
    with pytest.warns(UserWarning, match="it is outside of the store"):
        validate_zarr_store(labels)


@pytest.mark.parametrize("source", ["/abs/image.zarr", "https://example.com/a.zarr"])
def test_label_source_not_relative_warns(
    source: str, write_demo_ome: Callable, metadata_dict: Callable
) -> None:
    path = write_demo_ome("image-with-labels", version="0.5")
    _set_label_source(path, "0.5", source)
    with pytest.warns(UserWarning, match="it is outside of the store"):
        validate_zarr_store(metadata_dict(path))


def test_label_source_in_mapping_not_an_image(
    write_demo_ome: Callable, metadata_dict: Callable
) -> None:
    path = write_demo_ome("image-with-labels", version="0.5")
    _set_label_source(path, "0.5", "../")  # the labels group: not an image
    with pytest.raises(StorageValidationError, match="label_image_source_invalid"):
        validate_zarr_store(metadata_dict(path))


def _set_label_source(root: Path, version: str, source: str) -> None:
    label = root / "labels" / "annotations"
    meta_file = label / (".zattrs" if version == "0.4" else "zarr.json")
    data: dict[str, Any] = json.loads(meta_file.read_text())
    ome = data if version == "0.4" else data["attributes"]["ome"]
    ome["image-label"]["source"] = {"image": source}
    meta_file.write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("group_path", "src", "expected"),
    [
        ("labels/annotations", "../../", ""),
        ("labels/annotations", "../../other", "other"),
        ("labels/annotations", "../../../", None),
        ("", "../image", None),
        ("", "..", None),
        ("a", "./b", "a/b"),
        ("a", "..", ""),
        ("a", "/abs/b", None),
        ("a", "s3://bucket/b", None),
    ],
)
def test_source_store_key(group_path: str, src: str, expected: str | None) -> None:
    group = ZarrGroup(_encode({"zarr.json": _group()}), group_path, _root_meta())
    assert _source_store_key(group, src) == expected


def _root_meta() -> Any:
    return open_group(_encode({"zarr.json": _group()})).metadata
