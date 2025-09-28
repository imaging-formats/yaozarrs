# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "ome-zarr",
#     "zarr",
# ]
# ///

"""Functions to create small OME-ZARR datasets for testing.

These functions create minimal valid OME-ZARR datasets that can be used for testing
both v0.4 and v0.5 formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import zarr
from ome_zarr import writer
from ome_zarr.format import CurrentFormat, FormatV04
from ome_zarr.io import parse_url


def write_ome_image(
    path: str | Path,
    *,
    version: Literal["0.4", "0.5"] = "0.5",
    shape: tuple[int, ...] = (1, 1, 64, 64),
    axes: str = "czyx",
    dtype: str = "uint16",
    chunks: tuple[int, ...] | None = None,
    num_levels: int = 2,
    scale_factor: int = 2,
    channel_names: list[str] | None = None,
    channel_colors: list[int] | None = None,
) -> None:
    """Write a simple OME-ZARR image dataset.

    Parameters
    ----------
    path : str | Path
        Destination path for the dataset
    version : Literal["0.4", "0.5"], default "0.5"
        OME-ZARR version to write
    shape : tuple[int, ...], default (1, 1, 64, 64)
        Shape of the image data
    axes : str, default "czyx"
        Axis order string
    dtype : str, default "uint16"
        Data type for the image
    chunks : tuple[int, ...] | None, default None
        Chunk sizes. If None, uses (1, 1, 32, 32) for 4D data
    num_levels : int, default 2
        Number of resolution levels
    scale_factor : int, default 2
        Downsampling factor between levels
    channel_names : list[str] | None, default None
        Names for channels
    channel_colors : list[int] | None, default None
        Colors for channels (RGB hex values)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Default chunks
    if chunks is None:
        if len(shape) == 4:
            chunks = (1, 1, min(32, shape[-2]), min(32, shape[-1]))
        elif len(shape) == 3:
            chunks = (1, min(32, shape[-2]), min(32, shape[-1]))
        else:
            chunks = tuple(min(32, s) for s in shape)

    # Create sample data
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, size=shape, dtype=dtype)

    # Setup format
    fmt = FormatV04() if version == "0.4" else CurrentFormat()

    # Open zarr store
    if not (parsed := parse_url(str(path), mode="w", fmt=fmt)):
        raise ValueError(f"Could not parse path: {path}")

    store = parsed.store
    root = zarr.group(store=store)

    # Write image
    writer.write_image(
        image=data,
        group=root,
        axes=axes,
        storage_options={"chunks": chunks},
        scaler=writer.Scaler(
            downscale=scale_factor,
            max_layer=num_levels - 1,
        ),
    )

    # Add channel metadata if provided
    if channel_names or channel_colors:
        n_channels = shape[axes.index("c")] if "c" in axes else 1
        channels = []

        for i in range(n_channels):
            channel = {}
            if channel_names and i < len(channel_names):
                channel["label"] = channel_names[i]
            if channel_colors and i < len(channel_colors):
                channel["color"] = f"{channel_colors[i]:06x}"
            channels.append(channel)

        omero_metadata = {"omero": {"channels": channels}}
        writer.add_metadata(root, omero_metadata)


def write_ome_labels(
    path: str | Path,
    *,
    version: Literal["0.4", "0.5"] = "0.5",
    shape: tuple[int, ...] = (64, 64),
    axes: str = "yx",
    dtype: str = "uint32",
    chunks: tuple[int, ...] | None = None,
    num_levels: int = 2,
    scale_factor: int = 2,
    num_labels: int = 5,
    label_colors: list[tuple[int, int, int, int]] | None = None,
    parent_image_path: str | None = None,
) -> None:
    """Write a simple OME-ZARR labels dataset.

    Parameters
    ----------
    path : str | Path
        Destination path for the dataset
    version : Literal["0.4", "0.5"], default "0.5"
        OME-ZARR version to write
    shape : tuple[int, ...], default (64, 64)
        Shape of the label data
    axes : str, default "yx"
        Axis order string
    dtype : str, default "uint32"
        Data type for the labels
    chunks : tuple[int, ...] | None, default None
        Chunk sizes. If None, uses (32, 32) for 2D data
    num_levels : int, default 2
        Number of resolution levels
    scale_factor : int, default 2
        Downsampling factor between levels
    num_labels : int, default 5
        Number of distinct labels to create
    label_colors : list[tuple[int, int, int, int]] | None, default None
        RGBA colors for each label
    parent_image_path : str | None, default None
        Path to parent image if this is a segmentation of an image
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Default chunks
    if chunks is None:
        if len(shape) == 2:
            chunks = (min(32, shape[0]), min(32, shape[1]))
        else:
            chunks = tuple(min(32, s) for s in shape)

    # Create sample label data with distinct regions
    rng = np.random.default_rng(42)
    data = np.zeros(shape, dtype=dtype)

    # Create some random labeled regions
    for label_id in range(1, num_labels + 1):
        # Create a random elliptical region for each label
        center_y = rng.integers(shape[-2] // 4, 3 * shape[-2] // 4)
        center_x = rng.integers(shape[-1] // 4, 3 * shape[-1] // 4)
        radius_y = rng.integers(2, max(3, shape[-2] // 8))
        radius_x = rng.integers(2, max(3, shape[-1] // 8))

        y, x = np.ogrid[: shape[-2], : shape[-1]]
        mask = (y - center_y) ** 2 / radius_y**2 + (
            x - center_x
        ) ** 2 / radius_x**2 <= 1
        data[mask] = label_id

    # Setup format
    fmt = FormatV04() if version == "0.4" else CurrentFormat()

    # Open zarr store
    if not (parsed := parse_url(str(path), mode="w", fmt=fmt)):
        raise ValueError(f"Could not parse path: {path}")
    store = parsed.store
    root = zarr.group(store=store)

    # Write labels
    writer.write_labels(
        labels=data,
        group=root,
        name="labels",
        axes=axes,
        storage_options={"chunks": chunks},
        scaler=writer.Scaler(
            downscale=scale_factor,
            max_layer=num_levels - 1,
        ),
    )

    # Add label metadata
    colors = None
    if label_colors:
        colors = []
        for i, color in enumerate(label_colors[:num_labels]):
            colors.append({"label-value": i + 1, "rgba": list(color)})

    # Add additional metadata if needed
    kwargs = {}
    if parent_image_path:
        kwargs["source"] = {"image": parent_image_path}

    if colors or kwargs:
        writer.write_label_metadata(root, "labels", colors=colors, **kwargs)


def write_ome_plate(
    path: str | Path,
    *,
    version: Literal["0.4", "0.5"] = "0.5",
    plate_name: str = "test-plate",
    rows: list[str] | None = None,
    columns: list[str] | None = None,
    image_shape: tuple[int, ...] = (1, 64, 64),
    image_axes: str = "cyx",
    dtype: str = "uint16",
    chunks: tuple[int, ...] | None = None,
    num_levels: int = 2,
    scale_factor: int = 2,
    fields_per_well: int = 1,
    acquisitions: list[dict] | None = None,
) -> None:
    """Write a simple OME-ZARR plate dataset.

    Parameters
    ----------
    path : str | Path
        Destination path for the dataset
    version : Literal["0.4", "0.5"], default "0.5"
        OME-ZARR version to write
    plate_name : str, default "test-plate"
        Name of the plate
    rows : list[str] | None, default None
        Row names. If None, uses ["A", "B"]
    columns : list[str] | None, default None
        Column names. If None, uses ["1", "2"]
    image_shape : tuple[int, ...], default (1, 64, 64)
        Shape of each field image
    image_axes : str, default "cyx"
        Axis order string for images
    dtype : str, default "uint16"
        Data type for the images
    chunks : tuple[int, ...] | None, default None
        Chunk sizes for images
    num_levels : int, default 2
        Number of resolution levels for images
    scale_factor : int, default 2
        Downsampling factor between levels
    fields_per_well : int, default 1
        Number of field images per well
    acquisitions : list[dict] | None, default None
        Acquisition metadata
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Default values
    if rows is None:
        rows = ["A", "B"]
    if columns is None:
        columns = ["1", "2"]
    if chunks is None:
        if len(image_shape) == 3:
            chunks = (1, min(32, image_shape[-2]), min(32, image_shape[-1]))
        else:
            chunks = tuple(min(32, s) for s in image_shape)

    # Setup format
    fmt = FormatV04() if version == "0.4" else CurrentFormat()

    # Open zarr store
    if not (parsed := parse_url(str(path), mode="w", fmt=fmt)):
        raise ValueError(f"Could not parse path: {path}")
    store = parsed.store
    root = zarr.group(store=store)

    # Create well paths
    well_paths = []
    for row_name in rows:
        for col_name in columns:
            well_path = f"{row_name}/{col_name}"
            well_paths.append(well_path)

    # Write plate metadata first
    plate_acquisitions = (
        acquisitions if acquisitions else [{"id": 0, "name": "default"}]
    )
    writer.write_plate_metadata(
        root,
        rows,
        columns,
        well_paths,
        acquisitions=plate_acquisitions,
        name=plate_name,
    )

    # Create wells and fields
    for row_name in rows:
        for col_name in columns:
            well_path = f"{row_name}/{col_name}"

            # Create row and well groups
            row_group = root.require_group(row_name)
            well_group = row_group.require_group(col_name)

            # Create field paths
            field_paths = [str(i) for i in range(fields_per_well)]

            # Write well metadata
            well_metadata = {"images": [{"path": path} for path in field_paths]}

            if acquisitions:
                # Add acquisition references to images
                for i, img in enumerate(well_metadata["images"]):
                    img["acquisition"] = i % len(acquisitions)

            writer.write_well_metadata(well_group, field_paths)

            # Create fields in the well
            for field_idx in range(fields_per_well):
                # Create sample data for this field (different for each well/field)
                rng = np.random.default_rng(
                    hash(f"{row_name}{col_name}{field_idx}") % 2**32
                )
                field_data = rng.integers(0, 1000, size=image_shape, dtype=dtype)

                # Create field group and write image
                field_group = well_group.require_group(str(field_idx))
                writer.write_image(
                    image=field_data,
                    group=field_group,
                    axes=image_axes,
                    storage_options={"chunks": chunks},
                    scaler=writer.Scaler(
                        downscale=scale_factor,
                        max_layer=num_levels - 1,
                    ),
                )


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
