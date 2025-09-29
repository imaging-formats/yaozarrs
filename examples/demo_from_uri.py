#!/usr/bin/env python3
"""Demonstration script for the ZarrGroupModel.from_uri functionality."""

import builtins
import sys

from yaozarrs import from_uri

try:
    from rich import print  # ty: ignore
except ImportError:
    print = builtins.print  # type: ignore # noqa


def demo_zarr_uri(uri: str) -> None:
    """Demonstrate loading from a zarr URI."""
    print(f"🔬 Loading Zarr Group from: {uri}")
    print("=" * 80)

    try:
        # Load the zarr group
        zarr_group = from_uri(uri)

        print(zarr_group.model_dump(exclude_unset=True, exclude_none=True))
        print("✅ Successfully loaded!")

    except Exception as e:
        builtins.print(f"❌ Failed to load URI: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo_from_uri.py <uri>")
        print()
        print("Examples:")
        print("  python demo_from_uri.py /path/to/data.zarr")
        print("  python demo_from_uri.py https://example.com/data.zarr")
        print("  python demo_from_uri.py https://example.com/data.zarr/zarr.json")
        sys.exit(1)

    uri = sys.argv[1]
    demo_zarr_uri(uri)
