#!/usr/bin/env python3
"""Download only zarr metadata files (no data chunks) from a remote zarr store

parallelized version.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

# Thread-safe printing
print_lock = threading.Lock()


def safe_print(msg: str) -> None:
    """Thread-safe print function."""
    with print_lock:
        print(msg)


def download_file(url: str, local_path: Path) -> bool:
    """Download a file from URL to local path.

    Returns True if successful, False if 404 or other error.
    """
    try:
        req = Request(url)
        with urlopen(req) as response:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(response.read())
        safe_print(f"Downloaded: {url} -> {local_path}")
        return True
    except HTTPError as e:
        if e.code == 404:
            return False
        safe_print(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        safe_print(f"Error downloading {url}: {e}")
        return False


def download_metadata_files(base_url: str, local_base: Path, path: str = "") -> bool:
    """Download metadata files for a single path."""
    current_url = urljoin(base_url.rstrip("/") + "/", path)
    current_local = local_base / path if path else local_base

    found_something = False

    # Try to download .zgroup
    zgroup_url = urljoin(current_url.rstrip("/") + "/", ".zgroup")
    zgroup_local = current_local / ".zgroup"

    if download_file(zgroup_url, zgroup_local):
        found_something = True
        safe_print(f"Found group: {path or '(root)'}")

        # Download .zattrs if it exists
        zattrs_url = urljoin(current_url.rstrip("/") + "/", ".zattrs")
        zattrs_local = current_local / ".zattrs"
        download_file(zattrs_url, zattrs_local)

        # Download .zmetadata if it exists
        zmetadata_url = urljoin(current_url.rstrip("/") + "/", ".zmetadata")
        zmetadata_local = current_local / ".zmetadata"
        download_file(zmetadata_url, zmetadata_local)

    # Try to download .zarray
    zarray_url = urljoin(current_url.rstrip("/") + "/", ".zarray")
    zarray_local = current_local / ".zarray"

    if download_file(zarray_url, zarray_local):
        found_something = True
        safe_print(f"Found array: {path or '(root)'}")

        # Download .zattrs if it exists
        zattrs_url = urljoin(current_url.rstrip("/") + "/", ".zattrs")
        zattrs_local = current_local / ".zattrs"
        download_file(zattrs_url, zattrs_local)

    return found_something


def discover_all_paths(base_url: str, local_base: Path) -> set[str]:
    """Discover all paths that need to be downloaded."""
    paths_to_check = {""}  # Start with root
    discovered_paths = set()

    # First, download root metadata to discover structure
    if download_metadata_files(base_url, local_base, ""):
        discovered_paths.add("")

        # Check for .zattrs to discover structure
        zattrs_local = local_base / ".zattrs"
        if zattrs_local.exists():
            try:
                with open(zattrs_local) as f:
                    attrs = json.load(f)

                # Handle plates - discover all well paths
                if "plate" in attrs:
                    wells = attrs["plate"].get("wells", [])
                    safe_print(f"Found plate with {len(wells)} wells")

                    for well in wells:
                        well_path = well.get("path", "")
                        if well_path:
                            paths_to_check.add(well_path)

                            # Also check for common image paths within wells
                            for i in range(10):  # 0-9 images per well
                                paths_to_check.add(f"{well_path}/{i}")
                                # Common subpaths within images
                                for subpath in ["labels", "labels/0"]:
                                    paths_to_check.add(f"{well_path}/{i}/{subpath}")

                # Handle multiscales
                if "multiscales" in attrs:
                    for ms in attrs["multiscales"]:
                        for dataset in ms.get("datasets", []):
                            child_path = dataset.get("path", "")
                            if child_path:
                                paths_to_check.add(child_path)

            except Exception as e:
                safe_print(f"Warning: Could not parse root .zattrs: {e}")

    # Add common numeric patterns for any path
    additional_paths = set()
    for path in list(paths_to_check):
        if path:  # Don't add to root
            for i in range(10):
                additional_paths.add(f"{path}/{i}")
    paths_to_check.update(additional_paths)

    return paths_to_check


def parallel_download(
    base_url: str, local_base: Path, max_workers: int = 10
) -> set[str]:
    """Download zarr metadata in parallel."""

    # First discover all possible paths
    safe_print("Discovering paths to check...")
    paths_to_check = discover_all_paths(base_url, local_base)
    safe_print(f"Found {len(paths_to_check)} paths to check")

    # Download metadata for all paths in parallel
    discovered_paths = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_path = {
            executor.submit(download_metadata_files, base_url, local_base, path): path
            for path in paths_to_check
        }

        # Collect results
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                if future.result():
                    discovered_paths.add(path)
            except Exception as e:
                safe_print(f"Error processing path {path}: {e}")

    return discovered_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download zarr metadata files (no data chunks) "
        "from a remote zarr store (parallel version)"
    )
    parser.add_argument("url", help="URL to the zarr store")
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory (default: inferred from URL)",
        type=Path,
    )
    parser.add_argument(
        "-w",
        "--workers",
        help="Number of parallel workers (default: 10)",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Extract zarr name from URL
        parsed = urlparse(args.url)
        zarr_name = Path(parsed.path).name
        if not zarr_name.endswith(".zarr"):
            zarr_name += ".zarr"
        output_dir = Path(zarr_name)

    print(f"Downloading zarr metadata from: {args.url}")
    print(f"Output directory: {output_dir}")
    print(f"Using {args.workers} parallel workers")

    # Ensure the URL ends properly
    base_url = args.url.rstrip("/")
    if not base_url.endswith(".zarr"):
        base_url += ".zarr"

    # Discover and download all metadata in parallel
    discovered = parallel_download(base_url, output_dir, args.workers)

    if discovered:
        print(f"\nSuccessfully discovered {len(discovered)} groups/arrays:")
        for path in sorted(discovered):
            print(f"  {path or '(root)'}")
    else:
        print("No zarr groups or arrays found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
