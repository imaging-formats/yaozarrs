"""Zarr metadata downloader for remote stores."""

from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import fsspec  # type: ignore

META_SUFFIXES = (
    "zarr.json",  # Zarr v3
    ".zarray",  # Zarr v2
    ".zgroup",  # Zarr v2
    ".zattrs",  # Zarr v2
    ".zmetadata",  # Zarr v2 consolidated
    ".json",  # any other JSON metadata alongside NGFF/Zarr
)


def _is_meta(path: str) -> bool:
    return any(path.endswith(suf) for suf in META_SUFFIXES)


def _remote_root_name(remote_path: str) -> str:
    """Return the last path component, preserving a trailing '.zarr' folder name."""
    path = PurePosixPath(remote_path.rstrip("/"))
    return path.name or path.parent.name


def _list_all_paths(fs: Any, root: str) -> list[str]:
    """Recursively list paths under root. Prefer fs.find. Fallback to .zmetadata."""
    # Try listing at root and with trailing slash
    for candidate in (root, root.rstrip("/") + "/"):
        try:
            paths = fs.find(candidate, detail=False)
            if paths:
                return paths
        except Exception:
            continue

    # Fallback: use consolidated metadata if available (Zarr v2)
    for base in (root, root.rstrip("/") + "/"):
        zmeta_path = f"{base.rstrip('/')}/.zmetadata"
        try:
            if fs.exists(zmeta_path):
                with fs.open(zmeta_path, "rb") as f:
                    zmd = json.load(f)
                keys = list(zmd.get("metadata", {}).keys())
                return [zmeta_path] + [f"{base.rstrip('/')}/{k}" for k in keys]
        except Exception:
            pass

    # Final fallback: parse zarr.json and discover nested datasets
    for base in (root, root.rstrip("/") + "/"):
        # Use string joining for URL construction, not PurePosixPath
        top_zarr = base.rstrip("/") + "/zarr.json"
        try:
            with fs.open(top_zarr, "rb") as f:
                content = f.read()

            try:
                zarr_data = json.loads(content)
                paths = [top_zarr]

                # Discover OME multiscale datasets
                attrs = zarr_data.get("attributes", {})
                ome = attrs.get("ome", {})
                if zarr_data.get("node_type") == "group" and "multiscales" in ome:
                    for multiscale in ome["multiscales"]:
                        for dataset in multiscale.get("datasets", []):
                            dataset_path = dataset.get("path")
                            if dataset_path:
                                # Use string joining for URL construction
                                dataset_zarr = (
                                    f"{base.rstrip('/')}/{dataset_path}/zarr.json"
                                )
                                try:
                                    with fs.open(dataset_zarr, "rb") as df:
                                        df.read(16)  # Test if exists
                                    paths.append(dataset_zarr)
                                except Exception:
                                    pass

                return paths

            except json.JSONDecodeError:
                return [top_zarr]

        except Exception:
            continue

    raise RuntimeError(
        "Cannot list remote store. The endpoint may not expose directory listings. "
        "Provide consolidated .zmetadata (Zarr v2) or ensure listing is enabled."
    )


def metadata_mirror(remote_url: str, local_parent: str, verbose: bool = True) -> str:
    """Mirror only Zarr/NGFF metadata files from a remote store into a local directory.

    Parameters
    ----------
    remote_url : str
        URL pointing to the root of a Zarr store. Anonymous access is used unless
        the URL embeds credentials or your fsspec config provides them.
    local_parent : str
        Local directory under which the mirrored store directory will be created.
    verbose : bool
        Whether to print progress messages. Default is True.

    Returns
    -------
    str
        Path to the local mirrored store directory.
    """
    fs, root = fsspec.core.url_to_fs(remote_url)

    root_path = PurePosixPath(root.rstrip("/"))
    root_name = root_path.name or root_path.parent.name
    local_root = Path(local_parent) / root_name
    local_root.mkdir(parents=True, exist_ok=True)

    all_paths = _list_all_paths(fs, root)
    meta_paths = [p for p in all_paths if _is_meta(p)]

    if not meta_paths:
        return str(local_root)

    for rpath in meta_paths:
        rel = PurePosixPath(rpath).relative_to(PurePosixPath(root))
        lpath = local_root / rel
        lpath.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"⬇ {rpath}\n  {lpath}\n")
        with fs.open(rpath, "rb") as rf, lpath.open("wb") as lf:
            lf.write(rf.read())

    # Remove empty directories
    for dirpath in sorted(
        local_root.rglob("*"), key=lambda p: len(p.parts), reverse=True
    ):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            try:
                dirpath.rmdir()
            except OSError:
                pass

    return str(local_root)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python zz.py <remote_zarr_url>")
        sys.exit(1)
    remote = sys.argv[1]
    dest = "./data"
    result = metadata_mirror(remote, dest)
    print(f"Mirrored metadata to: {result}")
