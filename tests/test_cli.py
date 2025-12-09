"""Tests for the command-line interface."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Callable

import pytest

try:
    import zarr  # noqa: F401
except ImportError:
    pytest.skip("zarr not installed", allow_module_level=True)

from yaozarrs._cli import main

if TYPE_CHECKING:
    from pathlib import Path


def test_cli_no_args() -> None:
    """Test CLI with no arguments shows help."""
    result = main([])
    assert result == 0


def test_cli_validate_valid_store(write_demo_ome: Callable) -> None:
    """Test CLI validate command with valid zarr store."""
    path = write_demo_ome("image", version="0.5")
    result = main(["validate", str(path)])
    assert result == 0


def test_cli_validate_invalid_store(write_demo_ome: Callable) -> None:
    """Test CLI validate command with invalid zarr store."""
    path = write_demo_ome("image", version="0.5")
    # Remove a required dataset to make it invalid
    shutil.rmtree(path / "0")

    result = main(["validate", str(path)])
    assert result == 1


def test_cli_validate_nonexistent_store(tmp_path: Path) -> None:
    """Test CLI validate command with nonexistent zarr store."""
    nonexistent = tmp_path / "nonexistent"
    result = main(["validate", str(nonexistent)])
    assert result == 2


def test_cli_validate_labels(write_demo_ome: Callable) -> None:
    """Test CLI validate command with labels zarr store."""
    path = write_demo_ome("labels", version="0.5")
    result = main(["validate", str(path)])
    assert result == 0


def test_cli_validate_plate(write_demo_ome: Callable) -> None:
    """Test CLI validate command with plate zarr store."""
    path = write_demo_ome("plate", version="0.5")
    result = main(["validate", str(path)])
    assert result == 0


def test_cli_validate_bioformats2raw(write_demo_ome: Callable) -> None:
    """Test CLI validate command with bioformats2raw zarr store."""
    path = write_demo_ome("bioformats2raw", version="0.5")
    result = main(["validate", str(path)])
    assert result == 0


def test_cli_validate_image_with_labels(write_demo_ome: Callable) -> None:
    """Test CLI validate command with image-with-labels zarr store."""
    path = write_demo_ome("image-with-labels", version="0.5")
    result = main(["validate", str(path)])
    assert result == 0
