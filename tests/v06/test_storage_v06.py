"""Storage-level validation tests specific to OME-Zarr v0.6 (RFC-5).

These build minimal zarr v3 hierarchies by hand (metadata documents only; no
chunk data is needed for storage validation).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from yaozarrs import validate_zarr_store
from yaozarrs._storage import StorageValidationError, StorageValidationWarning

if TYPE_CHECKING:
    from pathlib import Path

# validate_zarr_store needs fsspec (the `io` extra); the stores here are
# hand-written zarr.json files, so zarr itself is not required.
pytest.importorskip("fsspec")

# ------------------------------------------------------------------
# store-building helpers
# ------------------------------------------------------------------


def _write_group(path: Path, ome: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"zarr_format": 3, "node_type": "group", "attributes": {}}
    if ome is not None:
        doc["attributes"]["ome"] = {"version": "0.6.dev4", **ome}
    path.joinpath("zarr.json").write_text(json.dumps(doc))


def _write_array(path: Path, shape: tuple[int, ...], dtype: str = "uint8") -> None:
    path.mkdir(parents=True, exist_ok=True)
    doc = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(shape),
        "data_type": dtype,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(shape)},
        },
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    path.joinpath("zarr.json").write_text(json.dumps(doc))


def _multiscale_meta(
    cs_name: str = "physical",
    axes: list[dict] | None = None,
    dataset_paths: tuple[str, ...] = ("s0",),
    coordinateTransformations: list[dict] | None = None,
    extra_coordinate_systems: list[dict] | None = None,
) -> dict:
    if axes is None:
        axes = [
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
    coordinate_systems = [{"name": cs_name, "axes": axes}]
    if extra_coordinate_systems:
        coordinate_systems.extend(extra_coordinate_systems)
    ms: dict[str, Any] = {
        "coordinateSystems": coordinate_systems,
        "datasets": [
            {
                "path": p,
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [float(2**i)] * len(axes),
                        "input": {"path": p},
                        "output": {"name": cs_name},
                    }
                ],
            }
            for i, p in enumerate(dataset_paths)
        ],
    }
    if coordinateTransformations is not None:
        ms["coordinateTransformations"] = coordinateTransformations
    return {"multiscales": [ms]}


def _write_image(
    path: Path,
    shape: tuple[int, ...] = (8, 8),
    dtype: str = "uint8",
    image_label: dict | None = None,
    **meta_kwargs: Any,
) -> None:
    ome = _multiscale_meta(**meta_kwargs)
    if image_label is not None:
        ome["image-label"] = image_label
    _write_group(path, ome=ome)
    meta = json.loads(path.joinpath("zarr.json").read_text())
    for ds in meta["attributes"]["ome"]["multiscales"][0]["datasets"]:
        _write_array(path / ds["path"], shape, dtype)


def _update_ome(path: Path, mutator: Any) -> None:
    """Load a group's zarr.json, apply `mutator(ome_dict)`, write it back."""
    zj = path / "zarr.json"
    doc = json.loads(zj.read_text())
    mutator(doc["attributes"]["ome"])
    zj.write_text(json.dumps(doc))


# ------------------------------------------------------------------
# scene validation
# ------------------------------------------------------------------


def _build_scene(tmp_path: Path) -> Path:
    """A valid scene: two images linked by an inline affine transform."""
    root = tmp_path / "scene.zarr"
    _write_group(
        root,
        ome={
            "scene": {
                "coordinateTransformations": [
                    {
                        "type": "affine",
                        "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        "input": {"path": "imageA", "name": "physical"},
                        "output": {"path": "imageB", "name": "physical"},
                    }
                ]
            }
        },
    )
    _write_image(root / "imageA")
    _write_image(root / "imageB")
    return root


def test_scene_valid(tmp_path: Path) -> None:
    root = _build_scene(tmp_path)
    validate_zarr_store(root)


def test_scene_missing_image(tmp_path: Path) -> None:
    import shutil

    root = _build_scene(tmp_path)
    shutil.rmtree(root / "imageB")
    with pytest.raises(StorageValidationError, match="transform_target_not_found"):
        validate_zarr_store(root)


def test_scene_endpoint_image_recursively_validated(tmp_path: Path) -> None:
    # a scene's endpoint images must themselves be fully valid (dataset arrays
    # present, etc), not just have parseable metadata -- `visit_scene` now
    # recurses into each referenced image via `visit_image`.
    import shutil

    root = _build_scene(tmp_path)
    shutil.rmtree(root / "imageB" / "s0")
    with pytest.raises(StorageValidationError, match="dataset_path_not_found"):
        validate_zarr_store(root)


def test_scene_missing_coordinate_system(tmp_path: Path) -> None:
    root = _build_scene(tmp_path)

    def _rename_cs(ome: dict) -> None:
        cs = ome["multiscales"][0]["coordinateSystems"][0]
        cs["name"] = "something_else"
        for ds in ome["multiscales"][0]["datasets"]:
            ds["coordinateTransformations"][0]["output"]["name"] = "something_else"

    _update_ome(root / "imageB", _rename_cs)
    with pytest.raises(StorageValidationError, match="transform_target_invalid"):
        validate_zarr_store(root)


def test_scene_scene_level_cs_not_declared(tmp_path: Path) -> None:
    root = _build_scene(tmp_path)

    def _use_undeclared(ome: dict) -> None:
        t = ome["scene"]["coordinateTransformations"][0]
        t["output"] = {"name": "world"}  # no path, not declared in the scene

    _update_ome(root, _use_undeclared)
    # SceneDef now catches this document-locally (no storage access needed), so
    # it's a pydantic ValidationError raised while parsing metadata, not a
    # StorageValidationError from the storage-layer graph/endpoint checks.
    with pytest.raises(ValidationError, match="is not declared in this scene"):
        validate_zarr_store(root)


def test_scene_disconnected_graph(tmp_path: Path) -> None:
    root = _build_scene(tmp_path)

    def _add_unconnected_cs(ome: dict) -> None:
        ome["scene"]["coordinateSystems"] = [
            {
                "name": "lonely",
                "axes": [
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
            }
        ]

    _update_ome(root, _add_unconnected_cs)
    with pytest.raises(StorageValidationError, match="transform_graph_disconnected"):
        validate_zarr_store(root)


def test_scene_affine_by_path(tmp_path: Path) -> None:
    root = _build_scene(tmp_path)

    def _use_path(ome: dict) -> None:
        t = ome["scene"]["coordinateTransformations"][0]
        del t["affine"]
        t["path"] = "coordinateTransformations/M"

    _update_ome(root, _use_path)

    # missing array -> error
    with pytest.raises(StorageValidationError, match="transform_path_not_found"):
        validate_zarr_store(root)

    # wrong shape (rotation-like 2x2, expected 2x3) -> error
    _write_group(root / "coordinateTransformations")
    _write_array(root / "coordinateTransformations" / "M", (2, 2), "float64")
    with pytest.raises(StorageValidationError, match="transform_array_invalid"):
        validate_zarr_store(root)

    # correct (M)x(N+1) shape -> valid
    _write_array(root / "coordinateTransformations" / "M", (2, 3), "float64")
    validate_zarr_store(root)


# ------------------------------------------------------------------
# multiscales > coordinateTransformations validation (image groups)
# ------------------------------------------------------------------

_WORLD_CS = {
    "name": "world",
    "axes": [
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ],
}


def test_image_affine_by_path(tmp_path: Path) -> None:
    root = tmp_path / "img.zarr"
    _write_image(
        root,
        extra_coordinate_systems=[_WORLD_CS],
        coordinateTransformations=[
            {
                "type": "affine",
                "path": "coordinateTransformations/M",
                "input": {"name": "physical"},
                "output": {"name": "world"},
            }
        ],
    )
    with pytest.raises(StorageValidationError, match="transform_path_not_found"):
        validate_zarr_store(root)

    _write_group(root / "coordinateTransformations")
    _write_array(root / "coordinateTransformations" / "M", (3, 3), "float64")
    with pytest.raises(StorageValidationError, match="transform_array_invalid"):
        validate_zarr_store(root)

    _write_array(root / "coordinateTransformations" / "M", (2, 3), "float64")
    validate_zarr_store(root)


def test_image_labels_link_target(tmp_path: Path) -> None:
    root = tmp_path / "img.zarr"
    _write_image(
        root,
        coordinateTransformations=[
            {
                "type": "identity",
                "input": {"name": "physical"},
                "output": {"name": "physical", "path": "labels/lbl"},
            }
        ],
    )
    # target group doesn't exist
    with pytest.raises(StorageValidationError, match="transform_target_not_found"):
        validate_zarr_store(root)

    # target exists but declares a differently-named coordinate system
    lbl_meta = {"colors": [{"label-value": 1}]}
    _write_group(root / "labels", ome={"labels": ["lbl"]})
    _write_image(root / "labels" / "lbl", cs_name="other", image_label=lbl_meta)
    with pytest.raises(StorageValidationError, match="transform_target_invalid"):
        validate_zarr_store(root)

    # matching coordinate system -> valid
    _write_image(root / "labels" / "lbl", cs_name="physical", image_label=lbl_meta)
    validate_zarr_store(root)


def test_image_displacement_field(tmp_path: Path) -> None:
    root = tmp_path / "img.zarr"
    _write_image(
        root,
        extra_coordinate_systems=[_WORLD_CS],
        coordinateTransformations=[
            {
                "type": "displacements",
                "path": "coordinateTransformations/disp",
                "input": {"name": "physical"},
                "output": {"name": "world"},
            }
        ],
    )
    with pytest.raises(StorageValidationError, match="transform_target_not_found"):
        validate_zarr_store(root)

    # valid displacement field: N+1 = 3 dims, displacement axis of length N=2
    # (v0.6rc0 requires "discrete": true on the displacement/coordinate axis)
    disp_axes = [
        {"name": "c", "type": "displacement", "discrete": True},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    _write_group(root / "coordinateTransformations")
    _write_image(
        root / "coordinateTransformations" / "disp",
        axes=disp_axes,
        shape=(2, 8, 8),
        dtype="float64",
    )
    validate_zarr_store(root)

    # wrong length along the displacement axis (3 != N=2)
    _write_image(
        root / "coordinateTransformations" / "disp",
        axes=disp_axes,
        shape=(3, 8, 8),
        dtype="float64",
    )
    with pytest.raises(StorageValidationError, match="vector_field_invalid"):
        validate_zarr_store(root)

    # no axis of type "displacement" (also trips the N+1 dimension check)
    _write_image(root / "coordinateTransformations" / "disp", shape=(8, 8))
    with pytest.raises(StorageValidationError, match="vector_field_invalid"):
        validate_zarr_store(root)


def test_image_dataset_dtype_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "img.zarr"
    _write_image(root, dataset_paths=("s0", "s1"))
    _write_array(root / "s1", (8, 8), "uint16")  # s0 stays uint8
    with pytest.raises(StorageValidationError, match="dataset_dtype_mismatch"):
        validate_zarr_store(root)


def test_image_dimension_names_mismatch_warns(tmp_path: Path) -> None:
    # v0.6 dropped the dimension_names requirement -> warning, not error
    root = tmp_path / "img.zarr"
    _write_image(root)
    zj = root / "s0" / "zarr.json"
    doc = json.loads(zj.read_text())
    doc["attributes"] = {"dimension_names": ["a", "b"]}
    zj.write_text(json.dumps(doc))
    with pytest.warns(StorageValidationWarning, match="dimension_names_mismatch"):
        validate_zarr_store(root)


def test_child_version_mismatch(tmp_path: Path) -> None:
    # spec (index.md): the OME-Zarr version MUST be consistent within a
    # hierarchy. A labels child declaring a different version than the root
    # must be reported, not silently force-parsed and accepted.
    root = tmp_path / "img.zarr"
    _write_image(root)
    _write_group(root / "labels", ome={"labels": ["cells"]})
    _write_image(root / "labels" / "cells", image_label={"colors": None})
    _update_ome(
        root / "labels" / "cells", lambda ome: ome.__setitem__("version", "0.6rc0")
    )
    with pytest.raises(StorageValidationError, match="version_mismatch"):
        validate_zarr_store(root)


def test_plain_group_where_label_expected_reports_error(tmp_path: Path) -> None:
    # a plain zarr group with no "ome" key (e.g. a stray directory) must be
    # reported as a normal validation error, not crash with a raw KeyError.
    root = tmp_path / "img.zarr"
    _write_image(root)
    _write_group(root / "labels", ome={"labels": ["cells"]})
    _write_group(root / "labels" / "cells")  # no "ome" metadata at all
    with pytest.raises(StorageValidationError, match="label_image_invalid"):
        validate_zarr_store(root)


# ------------------------------------------------------------------
# labels / plate cross-checks
# ------------------------------------------------------------------


def test_labels_intermediate_group_metadata(tmp_path: Path) -> None:
    root = tmp_path / "img.zarr"
    _write_image(root)
    _write_group(root / "labels", ome={"labels": ["original/0"]})
    _write_group(root / "labels" / "original")  # no metadata: fine
    label_meta = _multiscale_meta()
    label_meta["image-label"] = {"colors": [{"label-value": 1}]}
    _write_group(root / "labels" / "original" / "0", ome=label_meta)
    _write_array(root / "labels" / "original" / "0" / "s0", (8, 8), "uint8")
    validate_zarr_store(root)

    # give the intermediate group OME metadata -> error
    _write_group(root / "labels" / "original", ome={"labels": ["0"]})
    with pytest.raises(StorageValidationError, match="labels_intermediate_metadata"):
        validate_zarr_store(root)


def _build_plate(tmp_path: Path, images: list[dict], acquisitions: list[dict]) -> Path:
    root = tmp_path / "plate.zarr"
    _write_group(
        root,
        ome={
            "plate": {
                "acquisitions": acquisitions,
                "rows": [{"name": "A"}],
                "columns": [{"name": "1"}],
                "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
            }
        },
    )
    _write_group(root / "A")
    _write_group(root / "A" / "1", ome={"well": {"images": images}})
    for img in images:
        _write_image(root / "A" / "1" / img["path"])
    return root


def test_well_acquisition_unknown_id(tmp_path: Path) -> None:
    root = _build_plate(
        tmp_path,
        images=[{"path": "0", "acquisition": 5}],
        acquisitions=[{"id": 0}],
    )
    with pytest.raises(StorageValidationError, match="well_acquisition_invalid"):
        validate_zarr_store(root)


def test_well_acquisition_missing_when_multiple(tmp_path: Path) -> None:
    root = _build_plate(
        tmp_path,
        images=[{"path": "0"}],
        acquisitions=[{"id": 0}, {"id": 1}],
    )
    with pytest.raises(StorageValidationError, match="well_acquisition_invalid"):
        validate_zarr_store(root)


def test_well_acquisition_valid(tmp_path: Path) -> None:
    root = _build_plate(
        tmp_path,
        images=[{"path": "0", "acquisition": 1}],
        acquisitions=[{"id": 0}, {"id": 1}],
    )
    validate_zarr_store(root)
