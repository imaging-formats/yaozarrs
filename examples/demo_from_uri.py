# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yaozarrs[io]",
# ]
#
# [tool.uv.sources]
# yaozarrs = { path = "../", editable = true }
# ///
"""Demonstration script for the ZarrGroupModel.from_uri functionality."""

import builtins
import sys

from yaozarrs import validate_ome_uri

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
        zarr_group = validate_ome_uri(uri)

        print(zarr_group.model_dump(exclude_unset=True, exclude_none=True))
        print("✅ Successfully loaded!")

    except Exception as e:
        builtins.print(f"❌ Failed to load URI: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run examples/demo_from_uri.py <uri>")
        print()
        print("Examples:")
        print("  uv run examples/demo_from_uri.py /path/to/data.zarr")
        print(
            "  uv run examples/demo_from_uri.py https://uk1s3.embassy.ebi.ac.uk/idr/share/ome2024-ngff-challenge/idr0010/76-45.zarr"
        )
        sys.exit(1)

    demo_zarr_uri(sys.argv[1])
