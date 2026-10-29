from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from yaozarrs import validate_zarr_store
from yaozarrs._storage import (
    StorageErrorType,
    StorageValidationError,
    StorageValidationWarning,
)

pytest.importorskip("fsspec")

if TYPE_CHECKING:
    from collections.abc import Callable

DATA = Path(__file__).parent.parent / "data" / "v06"
STORES = ["bf2raw", "image_2d", "image_tczyx", "image_transforms", "image_with_labels"]
STORES += ["image_zyx_oriented", "plate", "scene", "scene_registration"]
SPACE = [{"name": n, "type": "space"} for n in "zyx"]


def _write_array(
    path: Path, ndim: int, dtype: str, shape: list[int] | None = None
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape or [8] * ndim,
        "data_type": dtype,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8] * ndim}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    (path / "zarr.json").write_text(json.dumps(meta))


def make_store(tmp_path: Path, name: str) -> Path:
    """Copy a tests/data/v06 store, adding array metadata for every dataset."""
    root = shutil.copytree(DATA / f"{name}.ome.zarr", tmp_path / f"{name}.ome.zarr")
    for zarr_json in list(root.rglob("zarr.json")):
        ome = json.loads(zarr_json.read_text())["attributes"].get("ome", {})
        dtype = "uint8" if "image-label" in ome else "uint16"
        for ms in ome.get("multiscales", ()):
            ndim = len(ms["coordinateSystems"][0]["axes"])
            for ds in ms["datasets"]:
                _write_array(zarr_json.parent / ds["path"], ndim, dtype)
    return root


def edit(subpath: str, fn: Callable[[dict], Any]) -> Callable[[Path], None]:
    """Return a mutator that edits the zarr.json at `subpath` in place."""

    def _edit(root: Path) -> None:
        zarr_json = root / subpath / "zarr.json"
        data = json.loads(zarr_json.read_text())
        fn(data)
        zarr_json.write_text(json.dumps(data))

    return _edit


def _ms(data: dict) -> dict:
    return data["attributes"]["ome"]["multiscales"][0]


def _label_ref(data: dict) -> dict:
    return _ms(data)["coordinateTransformations"][0]["output"]


def _scene(data: dict) -> dict:
    return data["attributes"]["ome"]["scene"]


def _drop_first_axis(subpath: str, dtype: str = "uint16") -> Callable[[Path], None]:
    """Return a mutator that makes the image at `subpath` one dimension smaller."""

    def _drop(root: Path) -> None:
        def _edit(data: dict) -> None:
            ms = _ms(data)
            ms["coordinateSystems"][0]["axes"].pop(0)
            ndim = len(ms["coordinateSystems"][0]["axes"])
            for ds in ms["datasets"]:
                t = ds["coordinateTransformations"][0]
                for step in t.get("transformations", [t]):  # (scale, or a sequence)
                    step.get("scale", step.get("translation")).pop(0)
                _write_array(root / subpath / ds["path"], ndim, dtype)

        edit(subpath, _edit)(root)

    return _drop


def _write_group(path: Path, ome: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    attrs = {"ome": {"version": "0.6", **ome}} if ome else {}
    meta = {"zarr_format": 3, "node_type": "group", "attributes": attrs}
    (path / "zarr.json").write_text(json.dumps(meta))


def _write_field(
    path: Path, *, length: int = 2, axis: dict | None = None, ndim: int = 2
) -> None:
    """Write a displacement field: a multiscale image with a displacement axis."""
    vector = axis or {"name": "d", "type": "displacement", "discrete": True}
    axes = [vector, *SPACE[-ndim:]]
    scale = {"type": "scale", "scale": [1.0] * len(axes)}
    ms = {
        "coordinateSystems": [{"name": "physical", "axes": axes}],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {**scale, "input": {"path": "s0"}, "output": {"name": "physical"}}
                ],
            }
        ],
    }
    _write_group(path, {"multiscales": [ms]})
    _write_array(path / "s0", len(axes), "float32", [length] + [8] * ndim)


DFIELD_PATH = "coordinateTransformations/dfield"
DFIELD = {
    "type": "displacements",
    "path": DFIELD_PATH,
    "input": {"path": "tile_0", "name": "intrinsic"},
    "output": {"path": "tile_1", "name": "intrinsic"},
}


def _with_field(path: str = DFIELD_PATH, **field_kwargs: Any) -> Callable[[Path], None]:
    """Mutator: add a displacements transformation (field at `path`) to `scene`."""

    def _add(root: Path) -> None:
        edit(
            "",
            lambda d: _scene(d)["coordinateTransformations"].append(
                {**DFIELD, "path": path}
            ),
        )(root)
        if field_kwargs.get("as_array"):
            _write_array(root / DFIELD_PATH, 3, "float32")
        elif not field_kwargs.get("missing"):
            _write_field(root / path.lstrip("/"), **field_kwargs)

    return _add


def _with_matrix(shape: list[int] | None, path: str = "coordinateTransformations/aff"):
    """Mutator: store the first affine of `image_transforms` in a Zarr array."""

    def _move(root: Path) -> None:
        def _edit(data: dict) -> None:
            t = _ms(data)["coordinateTransformations"][0]
            del t["affine"]
            t["path"] = path

        edit("", _edit)(root)
        if shape is not None:
            _write_array(root / path.lstrip("/"), 2, "float64", shape)

    return _move


def _with_rotation(shape: list[int]) -> Callable[[Path], None]:
    """Mutator: store the rotation in `scene_registration`'s sequence in an array."""

    def _move(root: Path) -> None:
        def _edit(data: dict) -> None:
            step = _scene(data)["coordinateTransformations"][0]["transformations"][0]
            del step["rotation"]
            step["path"] = "coordinateTransformations/rot"

        edit("", _edit)(root)
        _write_array(root / "coordinateTransformations" / "rot", 2, "float64", shape)

    return _move


def _with_bijection(root: Path) -> None:
    # registration fields both ways, as in the spec's scene example
    def _edit(data: dict) -> None:
        fields = {
            name: {"type": "displacements", "path": f"coordinateTransformations/{name}"}
            for name in ("fwd", "inv")
        }
        bijection = {
            **{k: DFIELD[k] for k in ("input", "output")},
            "type": "bijection",
            "forward": fields["fwd"],
            "inverse": fields["inv"],
        }
        _scene(data)["coordinateTransformations"].append(bijection)

    edit("", _edit)(root)
    for name in ("fwd", "inv"):
        _write_field(root / "coordinateTransformations" / name)


def _then_group(mutate: Callable[[Path], None], path: str) -> Callable[[Path], None]:
    """Run `mutate`, then put a plain group (no OME metadata) at `path`."""

    def _run(root: Path) -> None:
        mutate(root)
        shutil.rmtree(root / path, ignore_errors=True)
        _write_group(root / path)

    return _run


def _add_orphan_cs(subpath: str = "") -> Callable[[Path], None]:
    """Mutator: add a coordinate system that no transformation links to."""

    def _edit(data: dict) -> None:
        orphan = {"name": "orphan", "axes": SPACE[-2:]}
        if subpath == "scene":
            _scene(data)["coordinateSystems"].append(orphan)
        else:
            _ms(data)["coordinateSystems"].append(orphan)

    return edit("" if subpath == "scene" else subpath, _edit)


def _rename_labels_group(root: Path) -> None:
    # a labels group not called "labels", found via the transformation's `path`
    (root / "labels").rename(root / "segmentations")
    edit("", lambda d: _label_ref(d).update(path="segmentations/nuclei"))(root)


def _copy_label_outside_labels_group(root: Path) -> None:
    # an image with the referenced coordinate system, but not in a labels group
    shutil.copytree(root / "labels" / "nuclei", root / "other" / "nuclei")
    (root / "other" / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    edit("", lambda d: _label_ref(d).update(path="other/nuclei"))(root)


@pytest.mark.parametrize("name", STORES)
def test_valid_stores(tmp_path: Path, name: str) -> None:
    validate_zarr_store(make_store(tmp_path, name))


def test_labels_group_found_via_transformation(tmp_path: Path) -> None:
    root = make_store(tmp_path, "image_with_labels")
    _rename_labels_group(root)
    validate_zarr_store(root)
    # the whole labels group is validated, not just the referenced label image
    edit("segmentations/cells/s0", lambda d: d.update(data_type="float32"))(root)
    with pytest.raises(StorageValidationError, match="label_non_integer_dtype"):
        validate_zarr_store(root)


@pytest.mark.parametrize(
    "name, mutate",
    [
        ("scene", _with_field()),
        ("image_transforms", _with_matrix([4, 5])),
        ("scene_registration", _with_rotation([3, 3])),
        ("scene", _with_bijection),
    ],
    ids=["displacements-field", "affine-array", "rotation-array", "bijection"],
)
def test_valid_parameter_arrays(
    tmp_path: Path, name: str, mutate: Callable[[Path], None]
) -> None:
    root = make_store(tmp_path, name)
    mutate(root)
    validate_zarr_store(root)


WARNINGS = [
    # outside of a scene, an unlinked coordinate system only warns
    ("image_2d", _add_orphan_cs(), StorageErrorType.graph_not_connected),
    # a leading "/" isn't defined by the spec
    (
        "image_transforms",
        _with_matrix(None, path="/missing"),
        StorageErrorType.transform_parameters_not_found,
    ),
    # ... SHOULD be stored in a Zarr group ... called "coordinateTransformations"
    (
        "scene",
        _with_field(path="fields/dfield"),
        StorageErrorType.transform_parameters_location,
    ),
]


@pytest.mark.parametrize(
    "name, mutate, warn_type", WARNINGS, ids=[str(c[2]) for c in WARNINGS]
)
def test_warnings(
    tmp_path: Path, name: str, mutate: Callable[[Path], None], warn_type: Any
) -> None:
    root = make_store(tmp_path, name)
    mutate(root)
    with pytest.warns(StorageValidationWarning, match=str(warn_type)):
        validate_zarr_store(root)


def test_leading_slash_found_at_store_root(tmp_path: Path) -> None:
    root = make_store(tmp_path, "image_transforms")
    _with_matrix([4, 5], path="/coordinateTransformations/aff")(root)
    validate_zarr_store(root)


def test_dimension_names_not_checked(tmp_path: Path) -> None:
    # (0.6 has no rule about them; 0.5.2 required them to match the axes names)
    root = make_store(tmp_path, "image_2d")
    edit("scale0/image", lambda d: d.update(dimension_names=["a", "b"]))(root)
    validate_zarr_store(root)


INVALID = [
    (
        "image_2d",
        lambda root: shutil.rmtree(root / "scale1"),
        StorageErrorType.dataset_path_not_found,
    ),
    (
        "image_2d",
        edit("scale1/image", lambda d: d.update(shape=[8, 8, 8])),
        StorageErrorType.dataset_dimension_mismatch,
    ),
    # Every Zarr array referred to by a `path` MUST have the same ... datatype
    (
        "image_2d",
        edit("scale1/image", lambda d: d.update(data_type="float32")),
        StorageErrorType.dataset_dtype_mismatch,
    ),
    (
        "image_with_labels",
        edit("labels/cells/s0", lambda d: d.update(data_type="float32")),
        StorageErrorType.label_non_integer_dtype,
    ),
    # transformations to a coordinate system in a child labels group
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(path="labels/missing")),
        StorageErrorType.coordinate_system_path_not_found,
    ),
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(path="s0")),
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(path="labels")),
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "image_with_labels",
        _copy_label_outside_labels_group,
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(path="labels/nuclei/s0")),
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "image_with_labels",
        edit("", lambda d: _label_ref(d).update(name="missing")),
        StorageErrorType.coordinate_system_not_found,
    ),
    (
        "image_with_labels",
        _drop_first_axis("labels/nuclei", "uint8"),
        StorageErrorType.transform_dimension_mismatch,
    ),
    # scenes: references into image subgroups
    (
        "scene",
        edit(
            "",
            lambda d: _scene(d)["coordinateTransformations"][1]["input"].update(
                path="tile_9"
            ),
        ),
        StorageErrorType.coordinate_system_path_not_found,
    ),
    (
        "scene",
        edit(
            "",
            lambda d: _scene(d)["coordinateTransformations"][1]["input"].update(
                path="tile_1/scale0/image"
            ),
        ),
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "scene",
        edit(
            "",
            lambda d: _scene(d)["coordinateTransformations"][1]["input"].update(
                path="../tile_1"
            ),
        ),
        StorageErrorType.coordinate_system_path_invalid,
    ),
    (
        "scene",
        edit(
            "",
            lambda d: _scene(d)["coordinateTransformations"][1]["input"].update(
                name="missing"
            ),
        ),
        StorageErrorType.coordinate_system_not_found,
    ),
    (
        "scene_registration",
        _drop_first_axis("highres"),
        StorageErrorType.transform_dimension_mismatch,
    ),
    # The graph MUST be connected
    ("scene", _add_orphan_cs("scene"), StorageErrorType.graph_not_connected),
    ("scene", _add_orphan_cs("tile_0"), StorageErrorType.graph_not_connected),
    # parameters stored in Zarr arrays
    (
        "image_transforms",
        _with_matrix(None),
        StorageErrorType.transform_parameters_not_found,
    ),
    (
        "image_transforms",
        _with_matrix([4, 4]),
        StorageErrorType.transform_parameters_invalid,
    ),
    (
        "image_transforms",
        _with_matrix([4, 5, 1]),
        StorageErrorType.transform_parameters_invalid,
    ),
    (
        "scene_registration",
        _with_rotation([3, 2]),
        StorageErrorType.transform_parameters_invalid,
    ),
    (
        "image_transforms",
        _then_group(_with_matrix(None), "coordinateTransformations/aff"),
        StorageErrorType.transform_parameters_invalid,
    ),
    # displacement fields are multiscale images
    (
        "scene",
        _with_field(missing=True),
        StorageErrorType.transform_parameters_not_found,
    ),
    (
        "scene",
        _with_field(as_array=True),
        StorageErrorType.transform_parameters_invalid,
    ),
    (
        "scene",
        _then_group(_with_field(), "coordinateTransformations/dfield"),
        StorageErrorType.transform_parameters_invalid,
    ),
    ("scene", _with_field(length=3), StorageErrorType.transform_parameters_invalid),
    ("scene", _with_field(ndim=3), StorageErrorType.transform_parameters_invalid),
    (
        "scene",
        _with_field(axis={"name": "d", "type": "displacement", "discrete": False}),
        StorageErrorType.transform_parameters_invalid,
    ),
    (
        "scene",
        _with_field(axis={"name": "d", "type": "channel"}),
        StorageErrorType.transform_parameters_invalid,
    ),
    (
        "plate",
        lambda root: shutil.rmtree(root / "B" / "2"),
        StorageErrorType.well_path_not_found,
    ),
    (
        "plate",
        lambda root: shutil.rmtree(root / "A" / "1" / "1"),
        StorageErrorType.field_path_not_found,
    ),
    (
        "bf2raw",
        lambda root: shutil.rmtree(root / "1"),
        StorageErrorType.series_path_not_found,
    ),
]


@pytest.mark.parametrize(
    "name, mutate, err_type", INVALID, ids=[str(c[2]) for c in INVALID]
)
def test_invalid_stores(
    tmp_path: Path, name: str, mutate: Callable[[Path], None], err_type: Any
) -> None:
    root = make_store(tmp_path, name)
    mutate(root)
    with pytest.raises(StorageValidationError, match=str(err_type)) as exc:
        validate_zarr_store(root)
    assert {e["type"] for e in exc.value.errors()} == {str(err_type)}
