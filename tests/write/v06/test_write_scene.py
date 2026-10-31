"""Tests for writing OME-Zarr v0.6 scenes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

import yaozarrs
from yaozarrs import DimSpec, v05, v06
from yaozarrs.write.v06 import LabelsBuilder, SceneBuilder, write_scene

if TYPE_CHECKING:
    from pathlib import Path

np = pytest.importorskip("numpy")
pytest.importorskip("zarr")

DATA = np.zeros((16, 16), dtype=np.uint16)
SPEC = ((16, 16), "uint16")
WORLD = {
    "name": "world",
    "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
}


def _image(coordinate_system: str = "intrinsic") -> v06.Image:
    dims = [DimSpec(name="y"), DimSpec(name="x")]
    ms = v06.Multiscale.from_dims(dims, coordinate_system=coordinate_system)
    return v06.Image(multiscales=[ms])


def _to_world(path: str, name: str = "intrinsic", dy: float = 0) -> dict[str, Any]:
    return {
        "type": "translation",
        "translation": [dy, 0],
        "input": {"path": path, "name": name},
        "output": {"name": "world"},
    }


def _scene(*transforms: dict[str, Any], systems: list | None = None) -> v06.Scene:
    defs = {"coordinateTransformations": list(transforms)}
    defs["coordinateSystems"] = [WORLD] if systems is None else systems
    return v06.Scene.model_validate({"scene": defs})


def _ome(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text())["attributes"]["ome"]


def test_write_scene(tmp_path: Path) -> None:
    scene = _scene(_to_world("tiles/0"), _to_world("tiles/1", dy=16))
    images = {"tiles/0": (_image(), DATA), "tiles/1": (_image(), DATA)}
    dest = write_scene(tmp_path / "s.zarr", scene, images, extra_attributes={"k": 1})
    yaozarrs.validate_zarr_store(dest)

    root = json.loads((dest / "zarr.json").read_text())["attributes"]
    assert root["k"] == 1
    assert root["ome"]["version"] == "0.6"
    assert len(root["ome"]["scene"]["coordinateTransformations"]) == 2
    # (the intermediate group is a plain Zarr group)
    assert json.loads((dest / "tiles" / "zarr.json").read_text()) == {
        "zarr_format": 3,
        "node_type": "group",
    }
    assert "multiscales" in _ome(dest / "tiles" / "1")


def test_scene_builder_immediate(tmp_path: Path) -> None:
    # (image to image, no scene coordinate systems at all)
    scene = _scene(
        {
            "type": "identity",
            "input": {"path": "b", "name": "intrinsic"},
            "output": {"path": "a", "name": "intrinsic"},
        },
        systems=[],
    )
    builder = SceneBuilder(tmp_path / "s.zarr", scene)
    builder.write_image("a", _image(), DATA).write_image("b", _image(), DATA)
    assert repr(builder) == "<SceneBuilder: 2 images>"
    yaozarrs.validate_zarr_store(builder.root_path)


def test_scene_builder_prepare(tmp_path: Path) -> None:
    scene = _scene(_to_world("a"), _to_world("b"))
    builder = SceneBuilder(tmp_path / "s.zarr", scene)
    builder.write_image("a", _image(), DATA)
    builder.add_image("b", _image(), SPEC)
    # (prepare checks the scene against both the written and the added images)
    dest, arrays = builder.prepare()
    assert set(arrays) == {"b/0"}
    arrays["b/0"][:] = DATA
    yaozarrs.validate_zarr_store(dest)


def test_scene_builder_per_image_options(tmp_path: Path) -> None:
    scene = _scene(_to_world("a"), _to_world("b"), _to_world("c"))
    builder = SceneBuilder(tmp_path / "s.zarr", scene, chunks=(8, 8))
    builder.write_image("a", _image(), DATA, chunks=(4, 16), compression="none")
    builder.write_image("b", _image(), DATA)
    builder.add_image("c", _image(), SPEC, chunks=None)
    dest, _ = builder.prepare()

    def _array_meta(path: str) -> dict[str, Any]:
        return json.loads((dest / path / "0" / "zarr.json").read_text())

    chunk_shapes = {
        p: _array_meta(p)["chunk_grid"]["configuration"]["chunk_shape"] for p in "abc"
    }
    assert chunk_shapes == {"a": [4, 16], "b": [8, 8], "c": [16, 16]}
    assert "blosc" not in json.dumps(_array_meta("a")["codecs"])
    assert "blosc" in json.dumps(_array_meta("b")["codecs"])


def test_scene_wrong_chunks_ndim(tmp_path: Path) -> None:
    scene = _scene(_to_world("a"))
    dest = tmp_path / "s.zarr"
    with pytest.raises(ValueError, match="Image 'a': chunks="):
        write_scene(dest, scene, {"a": (_image(), DATA)}, chunks=(8, 8, 8))
    with pytest.raises(ValueError, match="Image 'a': shards="):
        write_scene(dest, scene, {"a": (_image(), DATA)}, shards=(8,))
    with pytest.raises(ValueError, match="Image 'a': chunks="):
        SceneBuilder(dest, scene).add_image("a", _image(), SPEC, chunks=(8, 8, 8))
    assert not dest.exists()
    # (per-image settings replace the builder's)
    builder = SceneBuilder(dest, scene, chunks=(8, 8, 8))
    builder.write_image("a", _image(), DATA, chunks=(8, 8))
    assert (dest / "a" / "0" / "zarr.json").exists()


def test_scene_with_label_image(tmp_path: Path) -> None:
    # a reference to a multiscale image *inside* one of the images (here one of its
    # labels) isn't checked when writing: the label image is written separately
    labels_cs = {**WORLD, "name": "labels"}
    scene = _scene(
        _to_world("a"),
        {
            "type": "identity",
            "input": {"path": "a/labels/cells", "name": "intrinsic"},
            "output": {"name": "labels"},
        },
        systems=[WORLD, labels_cs],
    )
    # (the image links its intrinsic system to the label image's, which connects the
    # "labels" scene system to the rest)
    dims = [DimSpec(name="y"), DimSpec(name="x")]
    ms = v06.Multiscale.from_dims(dims).model_dump(exclude_unset=True, by_alias=True)
    ms["coordinateTransformations"] = [
        {
            "type": "identity",
            "input": {"name": "intrinsic"},
            "output": {"name": "intrinsic", "path": "labels/cells"},
        }
    ]
    image = v06.Image(multiscales=[v06.Multiscale.model_validate(ms)])
    label = v06.LabelImage(
        multiscales=[v06.Multiscale.from_dims(dims)], image_label=v06.ImageLabel()
    )
    dest = write_scene(tmp_path / "s.zarr", scene, {"a": (image, DATA)})
    LabelsBuilder(dest / "a" / "labels").write_label(
        "cells", label, DATA.astype("uint8")
    )
    yaozarrs.validate_zarr_store(dest)


@pytest.mark.parametrize(
    ("scene", "images", "match"),
    [
        (
            _scene(_to_world("a"), _to_world("missing")),
            {"a": _image()},
            "refers to 'missing', which is not one of the scene's images",
        ),
        (
            _scene(_to_world("a", name="physical")),
            {"a": _image()},
            "image 'a' has no coordinate system 'physical'",
        ),
        (
            # "other" is not linked to anything
            _scene(_to_world("a"), systems=[WORLD, {**WORLD, "name": "other"}]),
            {"a": _image()},
            "not all linked",
        ),
        (
            _scene(_to_world("a"), _to_world("a/b")),
            {"a": _image(), "a/b": _image()},
            "can't be nested",
        ),
        (
            # a 2D image into a 3D scene coordinate system
            _scene(
                {**_to_world("a"), "translation": [0, 0, 0]},
                systems=[{**WORLD, "axes": [*WORLD["axes"], {"name": "z"}]}],
            ),
            {"a": _image()},
            r"translation: has 3 values \(input has 2, output has 3 axes\)",
        ),
        (
            _scene(
                {
                    "type": "translation",
                    "translation": [0, 0, 0],
                    "input": {"path": "a", "name": "intrinsic"},
                    "output": {"path": "b", "name": "intrinsic"},
                },
                systems=[],
            ),
            {"a": _image(), "b": _image()},
            "translation: has 3 values",
        ),
        (_scene(_to_world("__a")), {"__a": _image()}, "Invalid image path"),
        (_scene(_to_world("zarr.json")), {"zarr.json": _image()}, "Invalid image path"),
        (_scene(_to_world("a/../b")), {"a/../b": _image()}, "Invalid image path"),
    ],
    ids=[
        "missing-image",
        "missing-cs",
        "disconnected",
        "nested",
        "ndim-scene",
        "ndim-images",
        "dunder",
        "zarr-json",
        "dots",
    ],
)
def test_write_scene_errors(
    tmp_path: Path, scene: v06.Scene, images: dict[str, v06.Image], match: str
) -> None:
    dest = tmp_path / "s.zarr"
    with pytest.raises(ValueError, match=match):
        write_scene(dest, scene, {p: (img, DATA) for p, img in images.items()})
    assert not dest.exists()
    # ... and the same from the prepare-only workflow
    builder = SceneBuilder(dest, scene)
    with pytest.raises(ValueError, match=match):
        for path, image in images.items():
            builder.add_image(path, image, SPEC)
        builder.prepare()
    assert not dest.exists()


def test_scene_builder_checks_each_image(tmp_path: Path) -> None:
    builder = SceneBuilder(tmp_path / "s.zarr", _scene(_to_world("a")))
    with pytest.raises(ValueError, match="image 'a' has no coordinate system"):
        builder.write_image("a", _image("physical"), DATA)
    assert not builder.root_path.exists()
    builder.write_image("a", _image(), DATA)
    with pytest.raises(ValueError, match="already added"):
        builder.write_image("a", _image(), DATA)
    with pytest.raises(ValueError, match="No images added"):
        builder.prepare()


def test_scene_wrong_version(tmp_path: Path) -> None:
    v05_image: Any = v05.Image.model_validate(
        {
            "multiscales": [
                {
                    "axes": [
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1, 1]}
                            ],
                        }
                    ],
                }
            ]
        }
    )
    scene = _scene(_to_world("a"))
    with pytest.raises(TypeError, match=r"expected a v06\.Image, got v05\.Image"):
        write_scene(tmp_path / "s.zarr", scene, {"a": (v05_image, DATA)})
    with pytest.raises(TypeError, match=r"expected a v06\.Scene, got v06\.Image"):
        not_a_scene: Any = _image()
        SceneBuilder(tmp_path / "s.zarr", not_a_scene)
    assert not (tmp_path / "s.zarr").exists()


def test_write_scene_checks_before_writing(tmp_path: Path) -> None:
    scene = _scene(_to_world("a"), _to_world("b"))
    dest = tmp_path / "s.zarr"
    # (the second image's data doesn't match its axes)
    images = {"a": (_image(), DATA), "b": (_image(), np.zeros((2, 16, 16)))}
    with pytest.raises(ValueError, match="'0' has 3 dimensions, but the image has 2"):
        write_scene(dest, scene, images)
    bogus: Any = "?"
    with pytest.raises(ValueError, match="Unknown writer"):
        good = {"a": (_image(), DATA), "b": (_image(), DATA)}
        write_scene(dest, scene, good, writer=bogus)
    assert not dest.exists()


def test_write_scene_without_images(tmp_path: Path) -> None:
    # (a scene of its own coordinate systems only is still a scene group)
    other = {**WORLD, "name": "other"}
    scene = _scene(
        {"type": "identity", "input": {"name": "world"}, "output": {"name": "other"}},
        systems=[WORLD, other],
    )
    dest = write_scene(tmp_path / "s.zarr", scene, {})
    assert "scene" in _ome(dest)
    yaozarrs.validate_zarr_store(dest)


def test_scene_builder_retry_after_error(tmp_path: Path) -> None:
    builder = SceneBuilder(tmp_path / "s.zarr", _scene(_to_world("a")))
    with pytest.raises(ValueError, match="has 3 dimensions"):
        builder.write_image("a", _image(), np.zeros((2, 16, 16)))
    builder.write_image("a", _image(), DATA)
    yaozarrs.validate_zarr_store(builder.root_path)
