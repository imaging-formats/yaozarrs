# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "ome-zarr",
#     "yaozarrs",
#     "zarr",
# ]
#
# [tool.uv.sources]
# yaozarrs = { path = "../" }
# ///
"""A script to create demo OME-ZARR datasets from the command line."""

import json
from pathlib import Path

from yaozarrs._demo_data import write_ome_image, write_ome_labels, write_ome_plate


def main():
    """Main function to create demo datasets from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Create demo OME-ZARR datasets.")
    parser.add_argument(
        "--version", choices=["0.4", "0.5"], default="0.5", help="OME-ZARR version"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists",
    )
    # 3 different commands
    subparsers = parser.add_subparsers(dest="command", required=True)
    img_parser = subparsers.add_parser("image", help="Create an OME-ZARR image")
    img_parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[2, 4, 32, 32],
        help="Shape of the image data",
    )
    img_parser.add_argument(
        "--axes", type=str, default="czyx", help="Axis order string for the image"
    )
    img_parser.add_argument(
        "--dtype", type=str, default="uint16", help="Data type for the image"
    )
    img_parser.add_argument(
        "--num-levels", type=int, default=2, help="Number of resolution levels"
    )
    img_parser.add_argument(
        "--scale-factor", type=float, default=2.0, help="Downsampling factor"
    )
    img_parser.add_argument(
        "--channels",
        type=str,
        nargs="*",
        default=None,
        help="Channel names (space-separated)",
    )
    img_parser.add_argument(
        "--colors",
        type=lambda s: int(s, 16),
        nargs="*",
        default=None,
        help="Channel colors as hex integers (space-separated)",
    )

    lbl_parser = subparsers.add_parser("labels", help="Create an OME-ZARR labels")
    lbl_parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[32, 32],
        help="Shape of the label data",
    )
    lbl_parser.add_argument(
        "--axes", type=str, default="yx", help="Axis order string for the labels"
    )
    lbl_parser.add_argument(
        "--dtype", type=str, default="uint32", help="Data type for the labels"
    )
    lbl_parser.add_argument(
        "--num-levels", type=int, default=2, help="Number of resolution levels"
    )
    lbl_parser.add_argument(
        "--scale-factor", type=float, default=2.0, help="Downsampling factor"
    )
    lbl_parser.add_argument(
        "--num-labels", type=int, default=5, help="Number of distinct labels"
    )
    lbl_parser.add_argument(
        "--label-colors",
        type=lambda s: tuple(int(c) for c in s.split(",")),
        nargs="*",
        default=None,
        help="Label colors as RGBA tuples (e.g. 255,0,0,255)",
    )
    lbl_parser.add_argument(
        "--parent-image",
        type=str,
        default=None,
        help="Path to parent image if segmentation of an image",
    )

    plate_parser = subparsers.add_parser("plate", help="Create an OME-ZARR plate")
    plate_parser.add_argument(
        "--rows",
        type=str,
        nargs="*",
        default=None,
        help="Row names (space-separated)",
    )
    plate_parser.add_argument(
        "--columns",
        type=str,
        nargs="*",
        default=None,
        help="Column names (space-separated)",
    )
    plate_parser.add_argument(
        "--image-shape",
        type=int,
        nargs="+",
        default=[1, 64, 64],
        help="Shape of each field image",
    )
    plate_parser.add_argument(
        "--image-axes", type=str, default="cyx", help="Axis order string for images"
    )
    plate_parser.add_argument(
        "--dtype", type=str, default="uint16", help="Data type for the images"
    )
    plate_parser.add_argument(
        "--num-levels", type=int, default=2, help="Number of resolution levels"
    )
    plate_parser.add_argument(
        "--scale-factor", type=float, default=2.0, help="Downsampling factor"
    )
    plate_parser.add_argument(
        "--fields-per-well", type=int, default=1, help="Number of fields per well"
    )
    plate_parser.add_argument(
        "--acquisitions",
        type=str,
        nargs="*",
        default=None,
        help='Acquisition metadata as JSON strings (e.g. \'{"id":0,"name":"acq1"}\')',
    )

    parser.add_argument("output_dir", type=Path, help="Output directory for datasets")

    args = parser.parse_args()

    output_path = args.output_dir
    if output_path.exists():
        if args.overwrite:
            import shutil

            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output directory {output_path} already exists. "
                "Use --overwrite to replace."
            )
    output_path.mkdir(parents=True, exist_ok=True)
    version = args.version
    command = args.command
    if command == "image":
        write_ome_image(
            output_path / "image",
            version=version,
            shape=tuple(args.shape),
            axes=args.axes,
            dtype=args.dtype,
            num_levels=args.num_levels,
            scale_factor=args.scale_factor,
            channel_names=args.channels,
            channel_colors=args.colors,
        )
        print(f"Created OME-ZARR image at {output_path / 'image'}")
    elif command == "labels":
        write_ome_labels(
            output_path / "labels",
            version=version,
            shape=tuple(args.shape),
            axes=args.axes,
            dtype=args.dtype,
            num_levels=args.num_levels,
            scale_factor=args.scale_factor,
            num_labels=args.num_labels,
            label_colors=args.label_colors,
            parent_image_path=args.parent_image,
        )
        print(f"Created OME-ZARR labels at {output_path / 'labels'}")
    elif command == "plate":
        acquisitions = (
            [json.loads(acq) for acq in args.acquisitions]
            if args.acquisitions
            else None
        )
        write_ome_plate(
            output_path / "plate",
            version=version,
            rows=args.rows,
            columns=args.columns,
            image_shape=tuple(args.image_shape),
            image_axes=args.image_axes,
            dtype=args.dtype,
            num_levels=args.num_levels,
            scale_factor=args.scale_factor,
            fields_per_well=args.fields_per_well,
            acquisitions=acquisitions,
        )
        print(f"Created OME-ZARR plate at {output_path / 'plate'}")
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
