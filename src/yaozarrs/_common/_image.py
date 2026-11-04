"""Multiscale checks shared by more than one version."""

import math
from collections.abc import Sequence
from typing import Any


def check_transform_lengths(
    datasets: Sequence[Any], transforms: Sequence[Any] | None, ndim: int
) -> None:
    """Check the transformations of each dataset, and of the multiscale, fit `ndim`.

    The length of the scale and translation array MUST be the same as the length of
    "axes".
    """
    for i, ds in enumerate(datasets):
        for j, t in enumerate(ds.coordinateTransformations):
            if t.ndim != ndim:
                raise ValueError(
                    f"at datasets.[{i}].coordinateTransformations[{j}]:\n"
                    f"  The length of the transformation ({t.ndim}) does "
                    f"not match the number of axes ({ndim})."
                )
    for j, t in enumerate(transforms or ()):
        if t.ndim != ndim:
            raise ValueError(
                f"at coordinateTransformations[{j}]:\n"
                f"  The length of the transformation ({t.ndim}) "
                f"does not match the number of axes ({ndim})."
            )


def check_resolution_order(spatial_scales: Sequence[tuple[float, ...]]) -> None:
    """Check that datasets are ordered from highest to lowest resolution.

    `spatial_scales` holds the scales of each dataset along the spatial axes: per axis,
    a scale may not shrink from one level to the next (allowing for float noise).
    """
    if any(
        b < a and not math.isclose(a, b, rel_tol=1e-6)
        for prev, cur in zip(spatial_scales, spatial_scales[1:])
        for a, b in zip(prev, cur)
    ):
        raise ValueError(
            "The datasets are not ordered from highest to lowest resolution. "
            f"Found spatial scales: {spatial_scales}"
        )
