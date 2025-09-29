"""Command-line interface for yaozarrs."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import yaozarrs
from yaozarrs import from_uri, v04


def _get_zarr_info(uri: str) -> dict[str, Any]:
    """Get basic information about a zarr store."""
    info = {
        "uri": uri,
        "type": "unknown",
        "version": "unknown",
        "accessible": False,
        "is_local": not uri.startswith(("http://", "https://")),
    }

    try:
        # Try to load metadata
        metadata = from_uri(uri)
        info["accessible"] = True
        info["type"] = type(metadata).__name__

        # Get version
        if hasattr(metadata, "version"):
            info["version"] = metadata.version
        elif hasattr(metadata, "attributes") and hasattr(metadata.attributes, "ome"):
            ome_data = metadata.attributes.ome
            if hasattr(ome_data, "version"):
                info["version"] = ome_data.version
        elif hasattr(metadata, "multiscales"):
            # For v0.4, check if any multiscale has a version
            for ms in metadata.multiscales:
                if hasattr(ms, "version"):
                    info["version"] = ms.version
                    break
            else:
                # If no version found in multiscales, assume v0.4 for direct multiscales
                info["version"] = "0.4"

        # Get OME metadata type
        if hasattr(metadata, "attributes") and hasattr(metadata.attributes, "ome"):
            ome_data = metadata.attributes.ome
            if hasattr(ome_data, "multiscales"):
                if hasattr(ome_data, "image_label"):
                    info["ome_type"] = "label_image"
                else:
                    info["ome_type"] = "image"
            elif hasattr(ome_data, "plate"):
                info["ome_type"] = "plate"
            elif hasattr(ome_data, "well"):
                info["ome_type"] = "well"
            elif hasattr(ome_data, "labels"):
                info["ome_type"] = "labels_group"
            elif hasattr(ome_data, "series"):
                info["ome_type"] = "series"
            elif hasattr(ome_data, "bioformats2raw"):
                info["ome_type"] = "bioformats2raw"
        elif hasattr(metadata, "multiscales"):
            if hasattr(metadata, "image_label"):
                info["ome_type"] = "label_image"
            else:
                info["ome_type"] = "image"
        elif hasattr(metadata, "plate"):
            info["ome_type"] = "plate"
        elif hasattr(metadata, "well"):
            info["ome_type"] = "well"
        elif hasattr(metadata, "labels"):
            info["ome_type"] = "labels_group"
        elif hasattr(metadata, "series"):
            info["ome_type"] = "series"
        elif hasattr(metadata, "bioformats2raw"):
            info["ome_type"] = "bioformats2raw"

    except Exception as e:
        info["error"] = str(e)

    return info


def _print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def _print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{title}")
    print("-" * len(title))


def _print_result(success: bool, message: str, indent: int = 0) -> None:
    """Print a validation result with appropriate formatting."""
    prefix = "  " * indent
    if success:
        print(f"{prefix}✓ {message}")
    else:
        print(f"{prefix}✗ {message}")


def _print_warning(message: str, indent: int = 0) -> None:
    """Print a warning message."""
    prefix = "  " * indent
    print(f"{prefix}⚠ {message}")


def cmd_info(args: argparse.Namespace) -> int:
    """Handle the info subcommand."""
    _print_header("OME-ZARR Store Information")

    print(f"URI: {args.uri}")

    # Get basic info
    info = _get_zarr_info(args.uri)

    _print_section("Basic Information")
    print(f"Store Type: {info.get('type', 'unknown')}")
    print(f"OME-ZARR Version: {info.get('version', 'unknown')}")
    print(f"OME Type: {info.get('ome_type', 'unknown')}")
    print(f"Location: {'Local' if info['is_local'] else 'Remote'}")
    print(f"Accessible: {'Yes' if info['accessible'] else 'No'}")

    if not info["accessible"]:
        print(f"\nError: {info.get('error', 'Unknown error')}")
        return 1

    # Try to get more detailed info
    try:
        metadata = from_uri(args.uri)

        _print_section("Metadata Summary")

        # Handle different metadata structures
        ome_data = None
        if hasattr(metadata, "attributes") and hasattr(metadata.attributes, "ome"):
            ome_data = metadata.attributes.ome
        elif hasattr(metadata, "multiscales") or hasattr(metadata, "plate"):
            ome_data = metadata

        if ome_data:
            # Show relevant information based on type
            if hasattr(ome_data, "multiscales"):
                multiscales = ome_data.multiscales
                print(f"Number of multiscales: {len(multiscales)}")
                for i, ms in enumerate(multiscales):
                    if hasattr(ms, "name") and ms.name:
                        print(f"  Multiscale {i}: {ms.name}")
                    if hasattr(ms, "axes"):
                        axis_names = [ax.name for ax in ms.axes if hasattr(ax, "name")]
                        print(f"    Axes: {', '.join(axis_names)}")
                    if hasattr(ms, "datasets"):
                        print(f"    Resolution levels: {len(ms.datasets)}")

            elif hasattr(ome_data, "plate"):
                plate = ome_data.plate
                if hasattr(plate, "rows") and hasattr(plate, "columns"):
                    print(
                        f"Plate dimensions: {len(plate.rows)} rows x "
                        f"{len(plate.columns)} columns"
                    )
                if hasattr(plate, "wells"):
                    print(f"Number of wells: {len(plate.wells)}")
                if hasattr(plate, "name") and plate.name:
                    print(f"Plate name: {plate.name}")

            elif hasattr(ome_data, "well"):
                well = ome_data.well
                if hasattr(well, "images"):
                    print(f"Number of fields: {len(well.images)}")

            elif hasattr(ome_data, "labels"):
                labels = ome_data.labels
                print(f"Number of label images: {len(labels)}")
                for label in labels:
                    print(f"  - {label}")

    except Exception as e:
        print(f"\nError getting detailed information: {e}")
        return 1

    print()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Handle the validate subcommand."""
    _print_header("OME-ZARR Validation")

    print(f"URI: {args.uri}")
    level = "Metadata + Storage" if args.storage else "Metadata Only"
    print(f"Validation Level: {level}")

    # Step 1: Basic accessibility
    _print_section("Accessibility Check")
    info = _get_zarr_info(args.uri)

    if not info["accessible"]:
        error_msg = info.get("error", "Unknown error")
        _print_result(False, f"Cannot access store: {error_msg}")
        return 1

    _print_result(True, "Store is accessible")

    # Step 2: Metadata validation
    _print_section("Metadata Validation")

    try:
        metadata = from_uri(args.uri)
        _print_result(True, f"Successfully loaded {type(metadata).__name__} metadata")

        # Validate the pydantic model itself
        try:
            # The from_uri already validates, but let's be explicit
            if hasattr(metadata, "model_validate"):
                metadata.model_validate(metadata.model_dump())
            _print_result(True, "Metadata schema validation passed")
        except Exception as e:
            _print_result(False, f"Metadata schema validation failed: {e}")
            return 1

    except Exception as e:
        _print_result(False, f"Failed to load metadata: {e}")
        return 1

    # Step 3: Storage validation (if requested)
    if args.storage:
        _print_section("Storage Structure Validation")

        # Check if zarr is available
        try:
            import zarr  # noqa: F401
        except ImportError:
            _print_result(
                False, "Storage validation requires 'zarr' package to be installed"
            )
            print("Install with: pip install zarr")
            return 1

        # Storage validation works for both local and remote paths via zarr-python

        # Determine version and run appropriate validation
        version = info.get("version", "unknown")

        if version == "0.5":
            try:
                from yaozarrs.v05._storage import validate_zarr_store

                result = validate_zarr_store(metadata)
            except Exception as e:
                _print_result(False, f"Storage validation failed: {e}")
                return 1
        elif version == "0.4":
            try:
                result = v04.validate_storage(metadata)
            except Exception as e:
                _print_result(False, f"Storage validation failed: {e}")
                return 1
        else:
            _print_result(
                False, f"Storage validation not supported for version: {version}"
            )
            return 1

        # Report results
        if result.valid:
            _print_result(True, "Storage structure validation passed")
        else:
            _print_result(False, "Storage structure validation failed")

            if result.errors:
                print("\nErrors found:")
                for error in result.errors:
                    print(f"  ✗ {error.path}")
                    print(f"    {error.message}")

            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings:
                    print(f"  ⚠ {warning.path}")
                    print(f"    {warning.message}")

            return 1

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                _print_warning(f"{warning.path}: {warning.message}")

    # Summary
    _print_section("Summary")
    _print_result(True, "Validation completed successfully")

    if args.storage:
        print("Store passed both metadata and storage validation")
    else:
        print("Store passed metadata validation")
        print("Use --storage flag to also validate storage structure")

    print()
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="yaozarrs",
        description="OME-ZARR validation and inspection tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yaozarrs info /path/to/image.zarr
  yaozarrs validate https://example.com/image.zarr
  yaozarrs validate /path/to/plate.zarr --storage
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"yaozarrs {yaozarrs.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info subcommand
    info_parser = subparsers.add_parser(
        "info", help="Show information about an OME-ZARR store"
    )
    info_parser.add_argument(
        "uri", help="URI to the OME-ZARR store (local path or URL)"
    )

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate", help="Validate an OME-ZARR store"
    )
    validate_parser.add_argument(
        "uri", help="URI to the OME-ZARR store (local path or URL)"
    )
    validate_parser.add_argument(
        "--storage",
        action="store_true",
        help="Also validate storage structure (requires local path and zarr package)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "info":
            return cmd_info(args)
        elif args.command == "validate":
            return cmd_validate(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if "--debug" in sys.argv:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
