"""Storage validation for OME-ZARR v0.6 hierarchies.

This module provides functions to validate that OME-ZARR v0.6 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.  The traversal itself is shared
(`yaozarrs._common._storage.StorageValidator`).
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Any

from yaozarrs._common._storage import StorageValidator, _build_fs_path
from yaozarrs._storage import StorageErrorType, ValidationResult
from yaozarrs._validate import validate_ome_object, validate_ome_uri
from yaozarrs._zarr import ZarrArray, ZarrGroup
from yaozarrs.v06._bf2raw import Bf2Raw, Series
from yaozarrs.v06._image import Image, Multiscale
from yaozarrs.v06._labels import LabelImage, LabelsGroup
from yaozarrs.v06._plate import Plate, Well
from yaozarrs.v06._transforms import check_transform_ndim
from yaozarrs.v06._zarr_json import OMEAttributes, OMEZarrGroupJSON

if TYPE_CHECKING:
    from collections.abc import Mapping

    from yaozarrs._common._storage import Loc
    from yaozarrs.v06._coordinate_systems import CoordinateSystem


class StorageValidatorV06(StorageValidator):
    """Concrete implementation of storage validator. for OME-ZARR v0.6 spec."""

    __slots__ = ()

    VERSION = "0.6"
    Image = Image
    LabelImage = LabelImage
    LabelsGroup = LabelsGroup
    Plate = Plate
    Well = Well
    Bf2Raw = Bf2Raw
    Series = Series

    def _parse_ome(self, attrs: Mapping[str, Any]) -> Any:
        return validate_ome_object(attrs, OMEAttributes).ome

    def _parse_ome_uri(self, uri: str) -> Any:
        return validate_ome_uri(uri, OMEZarrGroupJSON).attributes.ome

    # NOTE: 0.6 has no rule about the arrays' `dimension_names` (0.5.2 required them
    # to be present and match the axes names), so there's no `_validate_dataset_array`.

    def visit_image(
        self, zarr_group: ZarrGroup, image_model: Image, loc_prefix: Loc
    ) -> ValidationResult:
        # The base class looks for a child group named `labels`: the spec calls it
        # "the `labels` group", and nothing else in the metadata needs to point to it,
        # so its name is the only practical way to find it.  In 0.6, a multiscale
        # transformation MAY point into a child labels group by `path`, so we also
        # visit labels groups found that way, known by their metadata (not their name).
        result = super().visit_image(zarr_group, image_model, loc_prefix)
        referenced: set[str] = set()
        for ms_idx, multiscale in enumerate(image_model.multiscales):
            ms_loc = (*loc_prefix, "multiscales", ms_idx)
            result = result.merge(
                self._validate_label_references(
                    zarr_group, multiscale, ms_loc, referenced
                )
            )
        for name in sorted(referenced - {"labels"}):
            group = zarr_group.get(name)
            labels_model = self._child_labels_group(zarr_group, name)
            if not isinstance(group, ZarrGroup) or labels_model is None:
                continue  # reported by _validate_label_references
            result = result.merge(
                self.visit_labels_group(
                    group, labels_model, (*loc_prefix, name), image_model
                )
            )
        return result

    def _child_labels_group(self, zarr_group: ZarrGroup, name: str) -> Any:
        """Return the LabelsGroup model of child group `name`, if it is one."""
        if not isinstance(group := zarr_group.get(name), ZarrGroup):
            return None
        try:
            model = group.ome_metadata(version=self.VERSION)
        except ValueError:
            return None
        return model if isinstance(model, LabelsGroup) else None

    def _visit_multiscale_no_prefetch(
        self, zarr_group: ZarrGroup, multiscale: Multiscale, loc_prefix: Loc
    ) -> ValidationResult:
        result = super()._visit_multiscale_no_prefetch(
            zarr_group, multiscale, loc_prefix
        )
        # Every Zarr array referred to by a `path` MUST have the same number of
        # dimensions and datatype
        # (the number of dimensions is checked against the axes for each array)
        first_dtype: str | None = None
        for ds_idx, dataset in enumerate(multiscale.datasets):
            if not isinstance(arr := zarr_group.get(dataset.path), ZarrArray):
                continue  # reported by the base class
            if first_dtype is None:
                first_dtype = arr.dtype
            elif arr.dtype != first_dtype:
                result.add_error(
                    StorageErrorType.dataset_dtype_mismatch,
                    (*loc_prefix, "datasets", ds_idx, "path"),
                    f"Dataset '{dataset.path}' has dtype '{arr.dtype}', but the "
                    f"first dataset has '{first_dtype}'",
                    ctx={
                        "fs_path": _build_fs_path(zarr_group, dataset.path),
                        "actual": arr.dtype,
                        "expected": first_dtype,
                    },
                )
        return result

    def _validate_label_references(
        self,
        zarr_group: ZarrGroup,
        multiscale: Multiscale,
        loc_prefix: Loc,
        referenced: set[str],
    ) -> ValidationResult:
        """Resolve multiscale transformations to coordinate systems in label images.

        Adds the child groups that the references go through to `referenced`.
        """
        result = ValidationResult()
        intrinsic = multiscale.intrinsic_coordinate_system
        for t_idx, t in enumerate(multiscale.coordinateTransformations or ()):
            # The other MUST reference a named coordinate system ... in a child
            # labels group (by `name` and `path`).
            # (the model checked everything that doesn't need the referenced group)
            side = "input" if t.input and t.input.path else "output"
            ref = getattr(t, side)
            if ref is None or ref.path is None or ref.name is None:
                continue
            loc = (*loc_prefix, "coordinateTransformations", t_idx, side)
            referenced.add(posixpath.normpath(ref.path).split("/", 1)[0])
            target = self._resolve_coordinate_system(
                zarr_group, ref.path, ref.name, loc, result
            )
            if target is None:
                continue
            n_in, n_out = (
                (target.ndim, intrinsic.ndim)
                if side == "input"
                else (intrinsic.ndim, target.ndim)
            )
            try:
                check_transform_ndim(t, n_in, n_out)
            except ValueError as e:
                result.add_error(
                    StorageErrorType.transform_dimension_mismatch,
                    (*loc_prefix, "coordinateTransformations", t_idx),
                    f"Transformation {side} {ref.name!r} in '{ref.path}': {e}",
                    ctx={"path": ref.path, "name": ref.name, "error": str(e)},
                )
        return result

    def _resolve_coordinate_system(
        self,
        zarr_group: ZarrGroup,
        path: str,
        name: str,
        loc: Loc,
        result: ValidationResult,
    ) -> CoordinateSystem | None:
        """Find coordinate system `name` in the label image at `path`, or add errors."""
        norm = posixpath.normpath(path)
        if (group := zarr_group.get(norm)) is None:
            result.add_error(
                StorageErrorType.coordinate_system_path_not_found,
                (*loc, "path"),
                f"Path '{path}' not found",
                ctx={"fs_path": _build_fs_path(zarr_group, path), "expected": "group"},
            )
            return None
        # ... in a child labels group: the child group that `path` goes through must
        # be a labels group (whatever its name)
        head, _, rest = norm.partition("/")
        if not rest or self._child_labels_group(zarr_group, head) is None:
            result.add_error(
                StorageErrorType.coordinate_system_path_invalid,
                (*loc, "path"),
                f"Path '{path}' is not in a child labels group of this image",
                ctx={"path": path, "expected": "<labels group>/<label image>"},
            )
            return None
        model: Any = None
        if isinstance(group, ZarrGroup):
            try:
                model = group.ome_metadata(version=self.VERSION)
            except ValueError as e:
                model = e
        if not isinstance(model, Image):
            ctx: dict = {"fs_path": _build_fs_path(zarr_group, path)}
            if isinstance(model, Exception):
                ctx["error"] = model
            else:
                ctx["type"] = type(model).__name__
            result.add_error(
                StorageErrorType.coordinate_system_path_invalid,
                (*loc, "path"),
                f"Path '{path}' does not contain valid Image ('multiscales') metadata",
                ctx=ctx,
            )
            return None
        for ms in model.multiscales:
            for cs in ms.coordinateSystems:
                if cs.name == name:
                    return cs
        result.add_error(
            StorageErrorType.coordinate_system_not_found,
            (*loc, "name"),
            f"Coordinate system {name!r} not found in '{path}'",
            ctx={
                "path": path,
                "name": name,
                "available": [
                    cs.name for ms in model.multiscales for cs in ms.coordinateSystems
                ],
            },
        )
        return None
