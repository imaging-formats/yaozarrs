"""Command-line interface for yaozarrs."""

from __future__ import annotations

import argparse
import sys

from yaozarrs._storage import StorageValidationError, validate_zarr_store
from yaozarrs._zarr import open_group


def print_zarr_info(path: str) -> None:
    """Print basic information about a valid zarr group.

    Parameters
    ----------
    path : str
        Path to the zarr store
    """
    group = open_group(path)
    ome_version = group.ome_version()

    # Determine group type from metadata
    ome_meta = group.ome_metadata()
    if ome_meta is None:  # pragma: no cover
        group_type = "unknown"
    else:
        group_type = type(ome_meta).__name__

    print("✓ Valid OME-Zarr store")
    print(f"  Version: {ome_version}")
    print(f"  Type: {group_type}")


def validate_command(args: argparse.Namespace) -> int:
    """Execute the validate subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    int
        Exit code (0 for success, 1 for validation failure, 2 for other errors)
    """
    try:
        validate_zarr_store(args.path)
        print_zarr_info(args.path)
        return 0
    except ImportError as e:  # pragma: no cover
        print(f"ImportError: {e}", file=sys.stderr)
        return 2
    except StorageValidationError as e:
        print(f"✗ Validation failed for: {args.path}\n", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (defaults to sys.argv[1:])

    Returns
    -------
    int
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="yaozarrs",
        description="CLI tools for validating OME-Zarr stores",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an OME-Zarr store",
    )
    validate_parser.add_argument(
        "path",
        help="Path or URI to the OME-Zarr store",
    )
    validate_parser.set_defaults(func=validate_command)

    # Parse arguments
    args = parser.parse_args(argv)

    # Show help if no command specified
    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    # Execute the command
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
