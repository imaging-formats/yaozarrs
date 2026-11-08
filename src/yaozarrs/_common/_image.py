"""Multiscale checks shared by more than one version."""

import itertools
import math
from collections.abc import Sequence
from typing import Any


def require_inline_or_path(t: Any, field: str, kind: str) -> None:
    """Check that `t` stores its parameters either inline (`field`) or at `path`.

    E.g. a 0.4/0.5 scale vector is "stored either as a list of floats (`scale`) or as
    binary data at a location in this container (`path`)", and a 0.6 affine matrix
    "either with JSON (`affine`) or as a Zarr array at a location in this container".
    """
    if (getattr(t, field) is None) == (t.path is None):
        raise ValueError(
            f"{t.type} requires exactly one of '{field}' (inline {kind}) or 'path' "
            "(Zarr array)"
        )


def check_transform_lengths(
    datasets: Sequence[Any], transforms: Sequence[Any] | None, ndim: int
) -> None:
    """Check the transformations of each dataset, and of the multiscale, fit `ndim`.

    The length of the scale and translation array MUST be the same as the length of
    "axes".
    """
    # (a transformation stored at a `path` has no length to check here)
    for i, ds in enumerate(datasets):
        for j, t in enumerate(ds.coordinateTransformations):
            if t.ndim is not None and t.ndim != ndim:
                raise ValueError(
                    f"at datasets.[{i}].coordinateTransformations[{j}]:\n"
                    f"  The length of the transformation ({t.ndim}) does "
                    f"not match the number of axes ({ndim})."
                )
    for j, t in enumerate(transforms or ()):
        if t.ndim is not None and t.ndim != ndim:
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
        for prev, cur in itertools.pairwise(spatial_scales)
        for a, b in zip(prev, cur, strict=False)
    ):
        raise ValueError(
            "The datasets are not ordered from highest to lowest resolution. "
            f"Found spatial scales: {spatial_scales}"
        )
