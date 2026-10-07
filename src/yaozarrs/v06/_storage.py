"""Storage validation for OME-ZARR v0.6 hierarchies.

This module provides functions to validate that OME-ZARR v0.6 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.

The logic shared with v0.4/v0.5 lives in
[`yaozarrs._storage_base.BaseStorageValidator`][]. This subclass adds the v0.6
(RFC-5) checks -- coordinate transformations, scenes, vector fields -- and
overrides the base hooks where the v0.6 spec differs (hierarchy version
consistency, dataset dtype consistency, `dimension_names` as a warning, label
dataset-count equality, intermediate label-group metadata, well acquisitions).
"""

from __future__ import annotations

import posixpath
from collections import defaultdict
from typing import Any

from yaozarrs._storage import StorageErrorType, ValidationResult
from yaozarrs._storage_base import BaseStorageValidator, Loc, _build_fs_path
from yaozarrs._zarr import ZarrArray, ZarrGroup
from yaozarrs.v06._bf2raw import Bf2Raw, Series
from yaozarrs.v06._image import Image, Multiscale
from yaozarrs.v06._labels import LabelImage, LabelsGroup
from yaozarrs.v06._plate import Plate, Well
from yaozarrs.v06._scene import Scene
from yaozarrs.v06._transforms import (
    AffineTransformation,
    BijectionTransformation,
    ByDimensionTransformation,
    CoordinatesTransformation,
    DisplacementsTransformation,
    InputOutput,
    RotationTransformation,
    SequenceTransformation,
    Transformation,
)
from yaozarrs.v06._version import CURRENT_VERSION
from yaozarrs.v06._zarr_json import OMEAttributes, OMEZarrGroupJSON

__all__ = ["StorageValidatorV06"]


class StorageValidatorV06(BaseStorageValidator):
    """Concrete implementation of storage validator for OME-ZARR v0.6 spec."""

    __slots__ = ("_root_version",)

    OME_VERSION = CURRENT_VERSION
    Image = Image
    LabelImage = LabelImage
    LabelsGroup = LabelsGroup
    Plate = Plate
    Well = Well
    Bf2Raw = Bf2Raw
    Series = Series
    OMEAttributes = OMEAttributes
    OMEZarrGroupJSON = OMEZarrGroupJSON

    # ------------------------------------------------------------------
    # base-class hook overrides
    # ------------------------------------------------------------------

    def _init_root(self, ome_metadata: Any) -> None:
        # The version declared at the root of this hierarchy. Children are
        # force-parsed against this same version (not a hardcoded literal), and
        # checked for consistency against it (spec index.md: "the OME-Zarr
        # version MUST be consistent within a hierarchy").
        self._root_version = getattr(ome_metadata, "version", CURRENT_VERSION)

    def _dispatch(
        self, zarr_group: ZarrGroup, ome_metadata: Any, loc_prefix: Loc
    ) -> ValidationResult:
        if isinstance(ome_metadata, Scene):
            return self.visit_scene(zarr_group, ome_metadata, loc_prefix)
        return super()._dispatch(zarr_group, ome_metadata, loc_prefix)

    def _visit_multiscale(
        self, zarr_group: ZarrGroup, multiscale: Multiscale, ms_loc: Loc
    ) -> ValidationResult:
        result = self._visit_multiscale_no_prefetch(zarr_group, multiscale, ms_loc)
        # v0.6: additional transformations may reference on-disk arrays
        # (affine/rotation matrices, displacement/coordinate fields) and
        # coordinate systems in child labels groups.
        return result.merge(
            self._visit_multiscale_transforms(zarr_group, multiscale, ms_loc)
        )

    def _check_dataset_array(
        self,
        zarr_group: ZarrGroup,
        multiscale: Multiscale,
        dataset: Any,
        arr: ZarrArray,
        ds_loc: Loc,
        result: ValidationResult,
        previous: list[tuple[str, ZarrArray]],
    ) -> None:
        # spec: every array referred to by a dataset path MUST have the same
        # datatype (compared against the first dataset array found).
        dtype = str(arr.dtype)
        if previous and dtype != (first_dtype := str(previous[0][1].dtype)):
            result.add_error(
                StorageErrorType.dataset_dtype_mismatch,
                ds_loc,
                f"Dataset '{dataset.path}' has dtype '{dtype}' but "
                f"'{previous[0][0]}' has dtype '{first_dtype}'. All "
                "datasets in a multiscale must have the same datatype.",
                ctx={
                    "fs_path": _build_fs_path(zarr_group, dataset.path),
                    "dtype": dtype,
                    "expected_dtype": first_dtype,
                },
            )
        # NB: a warning (not an error): the 0.5 spec required this, but the
        # 0.6 draft doesn't mention dimension_names at all.
        self._check_dimension_names(arr, multiscale, ds_loc, result, as_warning=True)

    def _check_label_path(
        self,
        labels_group: ZarrGroup,
        label_path: str,
        label_loc: Loc,
        result: ValidationResult,
    ) -> None:
        # Intermediate groups between `labels` and the images within it are
        # allowed, but these MUST NOT contain metadata.
        parts = label_path.split("/")
        for depth in range(1, len(parts)):
            inter_path = "/".join(parts[:depth])
            inter = labels_group.get(inter_path)
            if isinstance(inter, ZarrGroup) and "ome" in dict(inter.attrs):
                result.add_error(
                    StorageErrorType.labels_intermediate_metadata,
                    label_loc,
                    f"Intermediate group '{inter_path}' between the labels "
                    "group and its label images must not contain OME "
                    "metadata",
                    ctx={"fs_path": _build_fs_path(labels_group, inter_path)},
                )

    def _compare_label_to_parent(
        self,
        label_path: str,
        label_loc: Loc,
        label_image_model: Image,
        parent_image_model: Image,
        result: ValidationResult,
    ) -> None:
        # (NB: the spec constrains the *dataset* counts, not the number of
        # multiscales objects, so mismatched multiscales counts are allowed
        # and any extras are simply not compared.)
        for ms_idx, (lbl_ms, img_ms) in enumerate(
            zip(label_image_model.multiscales, parent_image_model.multiscales)
        ):
            n_lbl_ds = len(lbl_ms.datasets)
            n_img_ds = len(img_ms.datasets)
            # spec: the label image MUST have the *same* number of entries
            # (scale levels) as the original unlabeled image.
            if n_lbl_ds != n_img_ds:
                self._add_label_dataset_count_error(
                    label_path, label_loc, ms_idx, n_lbl_ds, n_img_ds, result
                )

    def _check_well_in_plate(
        self, plate_model: Plate, well_model: Well, well_loc: Loc
    ) -> ValidationResult:
        return self._validate_well_acquisitions(plate_model, well_model, well_loc)

    # ------------------------------------------------------------------
    # v0.6-only checks
    # ------------------------------------------------------------------

    def _child_metadata(
        self, child_group: ZarrGroup, loc: Loc, result: ValidationResult
    ) -> object:
        """Force-parse a child group's OME metadata as v0.6, checking version.

        Uses the hierarchy's root version (not a hardcoded literal) to parse the
        child, and records a `version_mismatch` error if the child's own
        declared version differs (spec: version MUST be consistent within a
        hierarchy). Returns either the parsed metadata object, or the
        `ValueError` raised while parsing (so callers can report either an
        `isinstance` mismatch or the underlying error).
        """
        declared = dict(child_group.attrs).get("ome", {})
        declared_version = (
            declared.get("version") if isinstance(declared, dict) else None
        )
        if declared_version is not None and declared_version != self._root_version:
            result.add_error(
                StorageErrorType.version_mismatch,
                loc,
                f"Group has version {declared_version!r}, but the root of this "
                f"hierarchy declares version {self._root_version!r}. The "
                "OME-Zarr version MUST be consistent within a hierarchy.",
                ctx={"declared": declared_version, "root": self._root_version},
            )
        try:
            return child_group.ome_metadata(version=self._root_version)
        except ValueError as e:
            return e

    def _visit_multiscale_transforms(
        self, zarr_group: ZarrGroup, multiscale: Multiscale, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate `multiscales > coordinateTransformations` against storage.

        Resolves labels-linked coordinate systems (`input.path` or
        `output.path` -- the model allows the labels link on either side as of
        v0.6rc0, see `Multiscale._post_validate`) and any path-backed transform
        parameters (affine/rotation matrices, displacement and coordinate
        fields).
        """
        result = ValidationResult()
        cs_dims = {cs.name: len(cs.axes) for cs in multiscale.coordinateSystems}

        def _resolve(io: InputOutput | None, io_loc: Loc) -> int | None:
            if io is None or io.name is None:
                return None
            if io.path is None:
                return cs_dims.get(io.name)
            # references a coordinate system in a child labels group: the path
            # MUST resolve to a multiscale image group declaring a coordinate
            # system with that name.
            target = self._load_image_target(zarr_group, io.path, io_loc, result)
            if target is None:
                return None
            cs = _find_image_cs(target, io.name)
            if cs is None:
                result.add_error(
                    StorageErrorType.transform_target_invalid,
                    io_loc,
                    f"Group '{io.path}' does not declare a coordinate system "
                    f"named {io.name!r}",
                    ctx={"path": io.path, "name": io.name},
                )
                return None
            return len(cs.axes)

        for t_idx, t in enumerate(multiscale.coordinateTransformations or []):
            t_loc = (*loc_prefix, "coordinateTransformations", t_idx)
            n_in = _resolve(t.input, (*t_loc, "input"))
            n_out = _resolve(t.output, (*t_loc, "output"))
            result = result.merge(
                self._validate_transform_params(zarr_group, t, t_loc, n_in, n_out)
            )
        return result

    def visit_scene(
        self, zarr_group: ZarrGroup, scene_model: Scene, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a scene group.

        - every transform `input`/`output` must resolve to a coordinate system,
          either declared in the scene itself (no `path`) or declared by a
          multiscale image subgroup at `path`;
        - path-backed transform parameters must resolve to valid arrays/groups;
        - the coordinate systems + transformations must form a fully connected
          graph (spec "Graph connectedness").
        """
        result = ValidationResult()
        scene_def = scene_model.scene
        scene_cs = {cs.name: cs for cs in (scene_def.coordinateSystems or [])}
        transforms = scene_def.coordinateTransformations

        # prefetch all endpoint image groups in one batch
        endpoint_paths = {
            io.path
            for t in transforms
            for io in (t.input, t.output)
            if io is not None and io.path and _is_relative_downward(io.path)
        }
        zarr_group.prefetch_children(endpoint_paths)
        image_cache: dict[str, Image | None] = {}

        def _endpoint_dims(io: InputOutput | None, io_loc: Loc) -> int | None:
            """Resolve an endpoint to its coordinate system dimensionality."""
            nonlocal result
            if io is None or io.name is None:  # enforced by the Scene model
                return None  # pragma: no cover
            if not io.path:
                if io.name not in scene_cs:
                    result.add_error(
                        StorageErrorType.transform_target_invalid,
                        io_loc,
                        f"Coordinate system {io.name!r} is not declared in the "
                        "scene (and no 'path' was given)",
                        ctx={"name": io.name},
                    )
                    return None
                return len(scene_cs[io.name].axes)
            if not _is_relative_downward(io.path):
                result.add_warning(
                    StorageErrorType.transform_target_not_found,
                    io_loc,
                    f"Path {io.path!r} is not a relative downward path; not validated",
                    ctx={"path": io.path},
                )
                return None
            if io.path not in image_cache:
                image = self._load_image_target(zarr_group, io.path, io_loc, result)
                image_cache[io.path] = image
                # spec: a scene's endpoint images are themselves ordinary image
                # groups -- their own datasets/transforms/labels MUST be valid.
                # Recurse into them (once per unique path) so e.g. a missing
                # array or an invalid child labels group is actually caught,
                # not just the endpoint's own metadata shape.
                if image is not None:
                    target = zarr_group.get(io.path)
                    if isinstance(target, ZarrGroup):
                        result = result.merge(self.visit_image(target, image, io_loc))
            if (image := image_cache[io.path]) is None:
                return None
            if (cs := _find_image_cs(image, io.name)) is None:
                result.add_error(
                    StorageErrorType.transform_target_invalid,
                    io_loc,
                    f"Image '{io.path}' does not declare a coordinate system "
                    f"named {io.name!r}",
                    ctx={"path": io.path, "name": io.name},
                )
                return None
            return len(cs.axes)

        # (path, name) pairs identify coordinate systems in the graph;
        # scene-level systems have path "".
        nodes: set[tuple[str, str]] = {("", name) for name in scene_cs}
        adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

        for t_idx, t in enumerate(transforms):
            t_loc = (*loc_prefix, "scene", "coordinateTransformations", t_idx)
            n_in = _endpoint_dims(t.input, (*t_loc, "input"))
            n_out = _endpoint_dims(t.output, (*t_loc, "output"))
            result = result.merge(
                self._validate_transform_params(zarr_group, t, t_loc, n_in, n_out)
            )
            if (
                t.input is not None
                and t.input.name is not None
                and t.output is not None
                and t.output.name is not None
            ):
                u = (t.input.path or "", t.input.name)
                v = (t.output.path or "", t.output.name)
                nodes.update((u, v))
                adjacency[u].add(v)
                adjacency[v].add(u)

        # Graph connectedness: any two coordinate systems in the metadata MUST
        # be connected by a sequence of transformations (direction-agnostic).
        if len(nodes) > 1:
            seen = {next(iter(nodes))}
            stack = list(seen)
            while stack:
                for neighbor in adjacency[stack.pop()]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            if unreached := nodes - seen:
                names = sorted("/".join(filter(None, n[::-1])) for n in unreached)
                result.add_error(
                    StorageErrorType.transform_graph_disconnected,
                    (*loc_prefix, "scene", "coordinateTransformations"),
                    "The scene's coordinate systems and transformations do not "
                    f"form a fully connected graph. Disconnected from the rest: "
                    f"{names}",
                    ctx={"disconnected": names},
                )

        return result

    def _validate_transform_params(
        self,
        zarr_group: ZarrGroup,
        transform: Transformation,
        loc: Loc,
        n_in: int | None,
        n_out: int | None,
    ) -> ValidationResult:
        """Validate on-disk parameters of a transform (recursively).

        `n_in`/`n_out` are the dimensionalities of the input/output coordinate
        systems when known (None when unresolvable, e.g. mid-sequence).
        """
        result = ValidationResult()
        if isinstance(transform, AffineTransformation):
            if transform.path is not None:
                # spec: 2D array of shape (M)x(N+1)
                result = result.merge(
                    self._check_matrix_array(
                        zarr_group,
                        transform.path,
                        loc,
                        expected_rows=n_out,
                        expected_cols=None if n_in is None else n_in + 1,
                        kind="affine",
                    )
                )
        elif isinstance(transform, RotationTransformation):
            if transform.path is not None:
                # spec: 2D array of shape NxN (input and output dims identical)
                n = n_in if n_in is not None else n_out
                result = result.merge(
                    self._check_matrix_array(
                        zarr_group,
                        transform.path,
                        loc,
                        expected_rows=n,
                        expected_cols=n,
                        kind="rotation",
                    )
                )
        elif isinstance(
            transform, (DisplacementsTransformation, CoordinatesTransformation)
        ):
            result = result.merge(
                self._check_vector_field(zarr_group, transform, loc, n_in, n_out)
            )
        elif isinstance(transform, SequenceTransformation):
            children = transform.transformations
            for c_idx, child in enumerate(children):
                # only the outermost dims are known: the first child's input is
                # the sequence's input, the last child's output its output.
                c_in = n_in if c_idx == 0 else None
                c_out = n_out if c_idx == len(children) - 1 else None
                result = result.merge(
                    self._validate_transform_params(
                        zarr_group, child, (*loc, "transformations", c_idx), c_in, c_out
                    )
                )
        elif isinstance(transform, BijectionTransformation):
            result = result.merge(
                self._validate_transform_params(
                    zarr_group, transform.forward, (*loc, "forward"), n_in, n_out
                )
            )
            result = result.merge(
                self._validate_transform_params(
                    zarr_group, transform.inverse, (*loc, "inverse"), n_out, n_in
                )
            )
        elif isinstance(transform, ByDimensionTransformation):
            for c_idx, item in enumerate(transform.transformations):
                result = result.merge(
                    self._validate_transform_params(
                        zarr_group,
                        item.transformation,
                        (*loc, "transformations", c_idx, "transformation"),
                        len(item.input_axes),
                        len(item.output_axes),
                    )
                )
        return result

    def _check_matrix_array(
        self,
        zarr_group: ZarrGroup,
        path: str,
        loc: Loc,
        expected_rows: int | None,
        expected_cols: int | None,
        kind: str,
    ) -> ValidationResult:
        """Check that a path-backed affine/rotation matrix is a valid 2D array."""
        result = ValidationResult()
        if not _is_relative_downward(path):
            result.add_warning(
                StorageErrorType.transform_path_not_found,
                (*loc, "path"),
                f"Path {path!r} is not a relative downward path; not validated",
                ctx={"path": path},
            )
            return result

        arr = zarr_group.get(path)
        if arr is None:
            result.add_error(
                StorageErrorType.transform_path_not_found,
                (*loc, "path"),
                f"The {kind} transform's path '{path}' was not found",
                ctx={"fs_path": _build_fs_path(zarr_group, path), "kind": kind},
            )
            return result
        if not isinstance(arr, ZarrArray):
            result.add_error(
                StorageErrorType.transform_array_invalid,
                (*loc, "path"),
                f"The {kind} transform's path '{path}' is not a zarr array",
                ctx={"path": path, "expected": "array", "found": "group"},
            )
            return result

        shape = arr.metadata.shape
        if shape is None or len(shape) != 2:
            result.add_error(
                StorageErrorType.transform_array_invalid,
                (*loc, "path"),
                f"The {kind} matrix at '{path}' must be 2-dimensional, "
                f"got shape {list(shape or ())}",
                ctx={"path": path, "shape": list(shape or ())},
            )
            return result

        rows, cols = shape
        if (expected_rows is not None and rows != expected_rows) or (
            expected_cols is not None and cols != expected_cols
        ):
            want = (
                f"({expected_rows or '?'}, {expected_cols or '?'})"
                if kind == "affine"
                else f"({expected_rows}, {expected_cols})"
            )
            result.add_error(
                StorageErrorType.transform_array_invalid,
                (*loc, "path"),
                f"The {kind} matrix at '{path}' has shape {list(shape)}, "
                f"but the input/output coordinate systems require {want}",
                ctx={"path": path, "shape": list(shape)},
            )
        return result

    def _check_vector_field(
        self,
        zarr_group: ZarrGroup,
        transform: DisplacementsTransformation | CoordinatesTransformation,
        loc: Loc,
        n_in: int | None,
        n_out: int | None,
    ) -> ValidationResult:
        """Check a displacements/coordinates field (a multiscale group at `path`).

        Spec constraints: the multiscale image MUST have N+1 dimensions (N =
        input dims); exactly one axis MUST have type "displacement"/"coordinate";
        the array length along that axis MUST equal N (displacements, with M=N)
        or M (coordinates).
        """
        result = ValidationResult()
        path = transform.path
        kind = transform.type  # "displacements" | "coordinates"
        vec_type = "displacement" if kind == "displacements" else "coordinate"

        if not _is_relative_downward(path):
            result.add_warning(
                StorageErrorType.transform_path_not_found,
                (*loc, "path"),
                f"Path {path!r} is not a relative downward path; not validated",
                ctx={"path": path},
            )
            return result

        image = self._load_image_target(zarr_group, path, (*loc, "path"), result)
        if image is None:
            return result

        target = zarr_group.get(path)
        multiscale = image.multiscales[0]
        if isinstance(target, ZarrGroup):
            # spec (0.6rc0 changelog): displacements/coordinates vector fields
            # are stored as a *normal* multiscale group with the same metadata
            # as other multiscales -- validate it exactly like `visit_image`
            # would (dataset arrays exist, ndim/dtype match, etc). Without this
            # a missing or wrong-shaped dataset array silently passes: only the
            # *metadata* was being checked before, not the arrays on disk.
            result = result.merge(
                self._visit_multiscale_no_prefetch(target, multiscale, (*loc, "path"))
            )

        if kind == "displacements" and None not in (n_in, n_out) and n_in != n_out:
            result.add_error(
                StorageErrorType.vector_field_invalid,
                loc,
                f"A displacements transform requires input and output coordinate "
                f"systems of equal dimensionality, got {n_in} and {n_out}",
                ctx={"path": path, "n_in": n_in, "n_out": n_out},
            )

        axes = multiscale.axes
        if n_in is not None and len(axes) != n_in + 1:
            result.add_error(
                StorageErrorType.vector_field_invalid,
                (*loc, "path"),
                f"The {kind} field at '{path}' must have {n_in + 1} dimensions "
                f"(input dimensionality + 1), but has {len(axes)} axes",
                ctx={"path": path, "ndim": len(axes), "expected": n_in + 1},
            )

        vec_idxs = [
            i for i, ax in enumerate(axes) if getattr(ax, "type", None) == vec_type
        ]
        if len(vec_idxs) != 1:
            result.add_error(
                StorageErrorType.vector_field_invalid,
                (*loc, "path"),
                f"The {kind} field at '{path}' must have exactly one axis of "
                f"type {vec_type!r}, found {len(vec_idxs)}",
                ctx={"path": path, "found": len(vec_idxs)},
            )
        else:
            idx = vec_idxs[0]
            vec_axis = axes[idx]
            # spec (0.6rc0 changelog): the displacement/coordinate axis is now
            # required to set `discrete: true`.
            if not getattr(vec_axis, "discrete", False):
                result.add_error(
                    StorageErrorType.vector_field_invalid,
                    (*loc, "path"),
                    f"The {kind} field's {vec_type!r} axis ({vec_axis.name!r}) "
                    "must set 'discrete': true",
                    ctx={"path": path, "axis": vec_axis.name},
                )
            # the length along the vector axis must equal M (coordinates) or
            # N (displacements). check against the highest-resolution level.
            expected_len = n_in if kind == "displacements" else n_out
            if expected_len is not None and isinstance(target, ZarrGroup):
                arr = target.get(multiscale.datasets[0].path)
                if isinstance(arr, ZarrArray) and (shape := arr.metadata.shape):
                    if len(shape) == len(axes) and shape[idx] != expected_len:
                        result.add_error(
                            StorageErrorType.vector_field_invalid,
                            (*loc, "path"),
                            f"The {kind} field at '{path}' has length "
                            f"{shape[idx]} along its {vec_type!r} axis, "
                            f"but the transform requires {expected_len}",
                            ctx={
                                "path": path,
                                "found": shape[idx],
                                "expected": expected_len,
                            },
                        )
        return result

    def _load_image_target(
        self, zarr_group: ZarrGroup, path: str, loc: Loc, result: ValidationResult
    ) -> Image | None:
        """Resolve `path` to a multiscale image group, recording errors."""
        target = zarr_group.get(path)
        if target is None:
            result.add_error(
                StorageErrorType.transform_target_not_found,
                loc,
                f"Path '{path}' was not found in the zarr group",
                ctx={"fs_path": _build_fs_path(zarr_group, path)},
            )
            return None
        if not isinstance(target, ZarrGroup):
            result.add_error(
                StorageErrorType.transform_target_invalid,
                loc,
                f"Path '{path}' is not a zarr group",
                ctx={"path": path, "expected": "group", "found": "array"},
            )
            return None
        meta = self._child_metadata(target, loc, result)
        if not isinstance(meta, Image):
            ctx: dict = {"path": path}
            if isinstance(meta, Exception):
                ctx["error"] = str(meta)
            else:
                ctx["type"] = type(meta).__name__
            result.add_error(
                StorageErrorType.transform_target_invalid,
                loc,
                f"Path '{path}' does not contain valid Image ('multiscales') metadata",
                ctx=ctx,
            )
            return None
        return meta

    def _validate_well_acquisitions(
        self, plate_model: Plate, well_model: Well, loc_prefix: Loc
    ) -> ValidationResult:
        """Cross-check well field-of-view acquisition ids against the plate.

        Spec: if multiple acquisitions were performed in the plate, each field
        of view MUST contain an `acquisition` key whose value MUST match one of
        the acquisitions defined in the plate metadata.
        """
        result = ValidationResult()
        if (acquisitions := plate_model.plate.acquisitions) is None:
            return result
        acq_ids = {acq.id for acq in acquisitions}
        for field_idx, field_image in enumerate(well_model.well.images):
            field_loc = (*loc_prefix, "well", "images", field_idx, "acquisition")
            if field_image.acquisition is None:
                if len(acquisitions) > 1:
                    result.add_error(
                        StorageErrorType.well_acquisition_invalid,
                        field_loc,
                        f"Field '{field_image.path}' has no 'acquisition' key, "
                        "but the plate defines multiple acquisitions",
                        ctx={"path": field_image.path},
                    )
            elif field_image.acquisition not in acq_ids:
                result.add_error(
                    StorageErrorType.well_acquisition_invalid,
                    field_loc,
                    f"Field '{field_image.path}' references acquisition "
                    f"{field_image.acquisition}, which is not defined in the "
                    f"plate metadata (defined: {sorted(acq_ids)})",
                    ctx={
                        "path": field_image.path,
                        "acquisition": field_image.acquisition,
                        "defined": sorted(acq_ids),
                    },
                )
        return result


# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------


def _is_relative_downward(path: str) -> bool:
    """Whether `path` stays inside the current group (validatable).

    Normalizes first so a path that only *looks* downward at the start but
    escapes via a later `..` segment (e.g. `"a/../../outside"`) is caught too,
    not just one starting with `"../"` or `"/"`.
    """
    if path.startswith("/"):
        return False
    normalized = posixpath.normpath(path)
    return normalized != ".." and not normalized.startswith("../")


def _find_image_cs(image: Image, name: str):
    """Find a coordinate system by name across an image's multiscales."""
    for multiscale in image.multiscales:
        for cs in multiscale.coordinateSystems:
            if cs.name == name:
                return cs
    return None
