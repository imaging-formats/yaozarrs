"""Convenience class for specifying image dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from yaozarrs._base import _BaseModel

if TYPE_CHECKING:
    from collections.abc import Sequence


class DimSpec(_BaseModel):
    """Specification for a single dimension of an OME-Zarr image.

    !!! important

        This is a convenience class and is not part of the OME-Zarr specification.

    There are some places in the OME Zarr spec where information about a given axis must
    be entered in multiple different places (e.g., `Multiscale.axes` must agree with
    `Multiscale.datasets.CoordinateTransformation`, etc). `DimSpec` is a convenience
    class that encapsulates all the relevant information about a single dimension in one
    place, and may be used in specialized constructors (e.g. `v05.Multiscale.from_dims`)

    Examples
    --------
    5D timelapse with channels:

    >>> from yaozarrs import DimSpec, v05
    >>> dims = [
    ...     DimSpec(name="t", size=100, scale=1.0, unit="second"),
    ...     DimSpec(name="c", size=3),
    ...     DimSpec(name="z", size=50, scale=2.0, unit="micrometer"),
    ...     DimSpec(name="y", size=512, scale=0.5, unit="micrometer"),
    ...     DimSpec(name="x", size=512, scale=0.5, unit="micrometer"),
    ... ]
    >>> multiscale = v05.Multiscale.from_dims(dims, name="my_image", n_levels=3)
    """

    name: str = Field(
        description=(
            "The name of the dimension. Common names like 'x', 'y', 'z' will be "
            "inferred as spatial dimensions; 't' as time; 'c' as channel."
        )
    )
    size: int | None = Field(
        default=None,
        description=(
            "The size of the dimension (number of elements along this axis). "
            "This is not used anywhere by the OME-Zarr spec, but is useful when "
            "creating new empty datasets."
        ),
    )
    scale: float = Field(
        default=1.0,
        description="The scale factor for this dimension in physical units.",
    )
    unit: str | None = Field(
        default=None,
        description=(
            "The physical unit for this dimension (e.g., 'micrometer', 'second'). "
            "If not provided, no unit will be set on the axis."
        ),
    )
    type: Literal["space", "time", "channel"] | str | None = Field(
        default=None,
        description=(
            "The type of axis ('space', 'time', 'channel', or custom). If not "
            "provided, will be inferred from the name: x/y/z -> 'space', "
            "t -> 'time', c -> 'channel'."
        ),
    )
    scale_factor: float | None = Field(
        default=None,
        description=(
            "The scale factor for downsampling along this dimension at each "
            "multiscale level. If not provided, defaults to 2.0 for spatial "
            "dimensions (x/y/z) and 1.0 for others. To *avoid* downsampling along "
            "the z dimension (which is inferred to be a spatial dimension), "
            "set this explicitly to 1.0."
        ),
    )
    translation: float | None = Field(
        default=None,
        description=(
            "The translation offset for this dimension in physical units. If not "
            "provided, no translation transform will be applied. If only some "
            "dimensions have translation specified, all others will default to 0.0."
        ),
    )

    def infer_scale_factor(self) -> float:
        """Infer the scale factor for downsampling based on dimension type."""
        if self.scale_factor is not None:
            return self.scale_factor
        return 2.0 if self.infer_type() == "space" else 1.0

    def infer_type(self) -> str | None:
        """Infer the axis type from the dimension name.

        Returns
        -------
        str | None
            The inferred type: 'space' for x/y/z, 'time' for t, 'channel' for c,
            or the explicitly set type if provided.
        """
        if self.type is not None:
            return self.type
        name_lower = self.name.lower()
        if name_lower in ("x", "y", "z"):
            return "space"
        if name_lower in {"t", "time"}:
            return "time"
        if name_lower in {"c", "channel"}:
            return "channel"
        return None


def _axes_datasets(dim_specs: Sequence[DimSpec], nlevels: int) -> dict[str, list[dict]]:
    axes: list[dict] = [
        {"name": d.name, "type": d.infer_type(), "unit": d.unit} for d in dim_specs
    ]

    datasets = []
    for level in range(nlevels):
        scale_tform = [d.scale * d.infer_scale_factor() ** level for d in dim_specs]
        tforms = [{"scale": scale_tform}]
        if any(d.translation is not None for d in dim_specs):
            tforms.append({"translation": [d.translation or 0.0 for d in dim_specs]})
        datasets.append({"path": str(level), "coordinateTransformations": tforms})
    return {"axes": axes, "datasets": datasets}
