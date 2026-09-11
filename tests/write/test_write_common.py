"""Behavior shared by the v0.5 and v0.6 writers (via `yaozarrs.write._core`)."""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING, Any

import pytest

import yaozarrs
from yaozarrs import DimSpec, v05, v06
from yaozarrs.write import v05 as w05
from yaozarrs.write import v06 as w06

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

np = pytest.importorskip("numpy")
# (python 3.10 can only install zarr 2.x; OME-Zarr v0.5+ writing needs zarr v3)
zarr = pytest.importorskip("zarr", minversion="3")


def _img05(n_levels: int = 2) -> v05.Image:
    axes = [v05.SpaceAxis(name="y"), v05.SpaceAxis(name="x")]
    datasets = [
        v05.Dataset(
            path=str(i),
            coordinateTransformations=[v05.ScaleTransformation(scale=[2.0**i] * 2)],
        )
        for i in range(n_levels)
    ]
    return v05.Image(multiscales=[v05.Multiscale(axes=axes, datasets=datasets)])


def _img06(n_levels: int = 2) -> v06.Image:
    dims = [DimSpec(name="y"), DimSpec(name="x")]
    return v06.Image(multiscales=[v06.Multiscale.from_dims(dims, n_levels=n_levels)])


# (writer module, model module, image factory, the *other* version's factory)
VERSIONS = {
    "v05": (w05, v05, _img05, _img06),
    "v06": (w06, v06, _img06, _img05),
}


@pytest.fixture(params=list(VERSIONS))
def ver(request: pytest.FixtureRequest) -> tuple[ModuleType, ModuleType, Any, Any]:
    return VERSIONS[request.param]


def _label(mod: ModuleType, img: Any) -> Any:
    return mod.LabelImage(**img.model_dump(), image_label={})


def _pyr(dtype: str = "uint16") -> list[Any]:
    return [np.zeros((64, 64), dtype), np.zeros((32, 32), dtype)]


def test_wrong_version_models_rejected(ver: tuple, tmp_path: Path) -> None:
    # passing e.g. a v0.5 Image to the v0.6 writer used to silently write v0.5
    # metadata into the store
    W, _mod, _mk, other = ver
    with pytest.raises(TypeError, match=r"expected a yaozarrs\.v0"):
        W.write_image(tmp_path / "a.zarr", other(), _pyr(), writer="zarr")
    with pytest.raises(TypeError, match=r"expected a yaozarrs\.v0"):
        W.prepare_image(tmp_path / "b.zarr", other(1), ((8, 8), "uint8"))
    with pytest.raises(TypeError, match="Series '0'"):
        W.Bf2RawBuilder(tmp_path / "c.zarr").add_series("0", other(1), ((8, 8), "u1"))
    with pytest.raises(TypeError, match="Well 'A/1', field '0'"):
        W.PlateBuilder(tmp_path / "d.zarr").add_well(
            row="A", col="1", images={"0": (other(1), ((8, 8), "uint8"))}
        )
    assert not any(tmp_path.iterdir())


def test_bad_label_fails_before_writing_image(ver: tuple, tmp_path: Path) -> None:
    W, mod, mk, _ = ver
    img = mk()
    dest = tmp_path / "img.zarr"
    with pytest.raises(ValueError, match="Label 'bad'"):
        W.write_image(
            dest,
            img,
            _pyr(),
            labels={"bad": (_label(mod, img), [np.zeros((64, 64), "uint8")])},
            writer="zarr",
        )
    assert not dest.exists()


def test_failed_label_write_not_registered(ver: tuple, tmp_path: Path) -> None:
    W, mod, mk, _ = ver
    img = mk()
    builder = W.LabelsBuilder(tmp_path / "img.zarr" / "labels", writer="zarr")
    builder.write_label("ok", _label(mod, img), _pyr("uint8"))
    with pytest.raises(ValueError, match="Number of data arrays"):
        builder.write_label("bad", _label(mod, img), [np.zeros((64, 64), "uint8")])
    meta = json.loads((builder.root_path / "zarr.json").read_text())
    assert meta["attributes"]["ome"]["labels"] == ["ok"]


def test_failed_series_write_not_registered(ver: tuple, tmp_path: Path) -> None:
    W, _mod, mk, _ = ver
    builder = W.Bf2RawBuilder(tmp_path / "b2r.zarr", writer="zarr")
    builder.write_image("0", mk(), _pyr())
    with pytest.raises(ValueError, match="Series '1'"):
        builder.write_image("1", mk(), [np.zeros((64, 64), "uint16")])
    meta = json.loads((builder.root_path / "OME" / "zarr.json").read_text())
    assert meta["attributes"]["ome"]["series"] == ["0"]
    yaozarrs.validate_zarr_store(builder.root_path)


def test_shards_clamped_per_level(ver: tuple, tmp_path: Path) -> None:
    # a fixed shard shape that fits level 0 but not level 1 used to produce an
    # invalid store partway through the pyramid
    W, _mod, mk, _ = ver
    data = [np.ones((200, 200), "uint8"), np.ones((100, 100), "uint8")]
    dest = W.write_image(
        tmp_path / "img.zarr",
        mk(),
        data,
        writer="zarr",
        chunks=(128, 128),
        shards=(256, 256),
    )
    yaozarrs.validate_zarr_store(dest)


def test_dask_write_misaligned_chunks(ver: tuple, tmp_path: Path) -> None:
    # dask chunks that don't line up with the storage chunks used to race and
    # silently corrupt data (da.store(..., lock=False))
    da = pytest.importorskip("dask.array")
    W, _mod, mk, _ = ver
    ref = np.arange(64 * 64, dtype="uint16").reshape(64, 64)
    data = [da.from_array(ref, chunks=(10, 10)), da.from_array(ref[::2, ::2], (7, 7))]
    writers = ["zarr"]
    if importlib.util.find_spec("tensorstore"):
        writers.append("tensorstore")
    for writer in writers:
        dest = W.write_image(tmp_path / f"{writer}.zarr", mk(), data, writer=writer)
        grp = zarr.open_group(dest, mode="r")
        np.testing.assert_array_equal(grp["0"][:], ref)
        np.testing.assert_array_equal(grp["1"][:], ref[::2, ::2])
