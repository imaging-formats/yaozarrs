"""Benchmarks for writing OME-Zarr groups (metadata only)."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

import pytest

from yaozarrs import DimSpec, v05
from yaozarrs.write.v05 import PlateBuilder

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_codspeed import BenchmarkFixture

np = pytest.importorskip("numpy")


def _numpy_writer(
    path: Path, shape: tuple[int, ...], dtype: Any, *_: Any, **__: Any
) -> Any:
    # (in-memory arrays, so that only our own overhead is measured)
    return np.empty(shape, dtype)


def _plate(n_rows: int, n_cols: int) -> v05.Plate:
    rows = [chr(ord("A") + i) for i in range(n_rows)]
    cols = [str(i + 1) for i in range(n_cols)]
    wells = [
        v05.PlateWell(path=f"{r}/{c}", rowIndex=ri, columnIndex=ci)
        for ri, r in enumerate(rows)
        for ci, c in enumerate(cols)
    ]
    return v05.Plate(
        plate=v05.PlateDef(
            rows=[v05.Row(name=r) for r in rows],
            columns=[v05.Column(name=c) for c in cols],
            wells=wells,
        )
    )


def test_write_plate_96(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    """Write a 96-well plate, one field of view per well."""
    plate = _plate(8, 12)
    dims = [DimSpec(name="y", scale=0.2), DimSpec(name="x", scale=0.2)]
    image = v05.Image(multiscales=[v05.Multiscale.from_dims(dims)])
    data = np.zeros((4, 4), "uint8")

    def write() -> None:
        dest = tempfile.mkdtemp(dir=tmp_path)
        builder = PlateBuilder(dest, plate=plate, writer=_numpy_writer, overwrite=True)
        for well in plate.plate.wells:
            row, col = well.path.split("/")
            builder.write_well(row, col, {"0": (image, data)})

    benchmark(write)
