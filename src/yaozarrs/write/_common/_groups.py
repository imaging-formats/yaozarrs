"""Zarr v3 group creation (`zarr.json`, with OME metadata under `attributes.ome`)."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel


def _atomic_write(path: Path, data: str) -> None:
    """Replace the file at `path` with `data`, atomically.

    (`data` is written to a temporary file next to it, which is then renamed over
    `path`: readers see either the old file or the new one, never a partial one.)
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_bytes(data.encode())
        _replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):  # (don't hide the error being raised)
            tmp.unlink(missing_ok=True)
        raise


def _replace(src: Path, dst: Path, attempts: int = 8) -> None:
    """`os.replace`, retried on Windows while another program has `dst` open."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            if sys.platform != "win32":
                raise
            if attempt == attempts - 1:
                raise PermissionError(
                    f"Couldn't replace {dst}: is it open in another program? "
                    "(Try again once it's closed.)"
                ) from e
            time.sleep(0.01 * 2**attempt)


def _write_group_json(
    dest_path: Path, attributes: dict[str, Any] | None = None, indent: int | None = None
) -> None:
    """Write the `zarr.json` of a group at `dest_path` (with `attributes`, if any)."""
    zarr_json: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}
    if attributes is not None:
        zarr_json["attributes"] = attributes
    _atomic_write(dest_path / "zarr.json", json.dumps(zarr_json, indent=indent))


def _create_zarr3_group(
    dest_path: Path,
    ome_model: BaseModel | None = None,
    overwrite: bool = False,
    extra_attributes: dict[str, Any] | None = None,
) -> None:
    """Create a zarr group directory with optional OME metadata in zarr.json."""
    if dest_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Zarr group already exists at {dest_path}. "
                "Use overwrite=True to replace it."
            )
        # Be cautious: don't delete anything that doesn't look like a zarr group.
        if not (dest_path / "zarr.json").exists():  # pragma: no cover
            raise FileExistsError(
                f"Destination {dest_path} exists, but is not a Zarr group. "
                "Refusing to overwrite.  Please delete manually."
            )
        shutil.rmtree(dest_path, ignore_errors=True)

    dest_path.mkdir(parents=True, exist_ok=True)
    attrs: dict[str, Any] = {}
    if ome_model is not None:
        attrs["ome"] = ome_model.model_dump(mode="json", exclude_none=True)
    attrs.update(extra_attributes or {})
    _write_group_json(dest_path, attrs or None, indent=2)


def _update_zarr3_group(dest_path: Path, ome_model: BaseModel) -> None:
    """Update the ome metadata in an existing zarr group."""
    zarr_json_path = dest_path / "zarr.json"
    if not zarr_json_path.exists():
        raise FileNotFoundError(f"Zarr group metadata not found at {zarr_json_path}")

    with open(zarr_json_path) as f:
        zarr_json = json.load(f)

    # Preserve existing extra attributes (non-ome keys)
    existing_attrs = zarr_json.get("attributes", {})
    attrs: dict[str, Any] = {k: v for k, v in existing_attrs.items() if k != "ome"}
    attrs["ome"] = ome_model.model_dump(mode="json", exclude_none=True)
    zarr_json["attributes"] = attrs
    _atomic_write(zarr_json_path, json.dumps(zarr_json, indent=2))


def _rewrite_zarr3_group(dest_path: Path, ome_model: BaseModel) -> None:
    """(Re)write the group at `dest_path`, keeping any non-ome attributes it has.

    Unlike `_update_zarr3_group`, the group's zarr.json needn't exist yet, and is
    written anew (with "ome" first in its attributes).
    """
    extra: dict[str, Any] = {}
    if (zarr_json := dest_path / "zarr.json").exists():
        existing = json.loads(zarr_json.read_text())
        extra = {k: v for k, v in existing.get("attributes", {}).items() if k != "ome"}
    ome = ome_model.model_dump(mode="json", exclude_none=True)
    _write_group_json(dest_path, {"ome": ome, **extra}, indent=2)


def _ensure_parent_groups(root: Path, path: str) -> None:
    """Create plain groups for each intermediate node of `path` below `root`.

    (Zarr v3 has no implicit groups, so e.g. `root/a/` needs a `zarr.json` of its own
    for a reader to resolve `root/a/b`.)  Existing groups are left alone.
    """
    parent = root
    for part in path.split("/")[:-1]:
        parent = parent / part
        parent.mkdir(parents=True, exist_ok=True)
        if not (parent / "zarr.json").exists():
            _write_group_json(parent)
