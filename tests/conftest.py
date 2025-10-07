"""Test script to verify the OME-ZARR writer functions work correctly."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Callable, Literal

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def write_demo_ome(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Path]:
    try:
        from yaozarrs._demo_data import (
            write_ome_bf2raw,
            write_ome_image,
            write_ome_labels,
            write_ome_plate,
        )
    except ImportError:
        pytest.skip("ome-zarr not installed", allow_module_level=True)

    def _write_demo(
        type: Literal[
            "image", "labels", "plate", "image-with-labels", "bioformats2raw"
        ],
        **kwargs: Any,
    ) -> Path:
        ome_version = kwargs.setdefault("version", "0.5")
        if ome_version == "0.5" and version("zarr").startswith("2"):
            pytest.skip("zarr v2 does not support OME-Zarr v0.5")

        path = tmp_path_factory.mktemp(f"demo_{type}.zarr")
        if type in ("image", "image-with-labels", "bioformats2raw"):
            if type == "bioformats2raw":
                write_ome_bf2raw(path, **kwargs)
            else:
                write_ome_image(path, **kwargs)
            if type == "image-with-labels":
                write_ome_labels(path, **kwargs)
        elif type == "labels":
            # XXX: write_labels is special (in ome-zarr) ...
            # it creates a group "labels"
            write_ome_labels(path, **kwargs)
            path = path / "labels"
        elif type == "plate":
            write_ome_plate(path, **kwargs)
        else:
            raise ValueError(f"Unknown type: {type}")
        return path

    return _write_demo
