"""Storage validation for OME-ZARR v0.6 hierarchies.

This module provides functions to validate that OME-ZARR v0.6 storage structures
conform to the specification requirements for directory layout, file existence,
and metadata consistency.  The traversal itself is shared
(`yaozarrs._common._storage.StorageValidator`).
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Any, TypeAlias

from yaozarrs._common._storage import StorageValidator, _build_fs_path
from yaozarrs._storage import StorageErrorType, ValidationResult
from yaozarrs._validate import validate_ome_object, validate_ome_uri
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
    RotationTransformation,
    SequenceTransformation,
    _output_ndim,
    _own_ndim,
    check_transform_ndim,
)
from yaozarrs.v06._zarr_json import OMEAttributes, OMEZarrGroupJSON

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from yaozarrs._common._storage import Loc
    from yaozarrs.v06._coordinate_systems import CoordinateSystem
    from yaozarrs.v06._transforms import CoordinateTransformation

# a node of the transformations graph:
# (image path, or None for the scene itself; multiscales index; coordinate system name)
Node: TypeAlias = tuple[str | None, int, str]

_FIELDS = (DisplacementsTransformation, CoordinatesTransformation)


class StorageValidatorV06(StorageValidator):
    """Concrete implementation of storage validator. for OME-ZARR v0.6 spec."""

    # (images already visited and parameter checks already done, by normalized path:
    # references can point back to a group being validated, or to the same group twice)
    __slots__ = ("_checked", "_visited")

    def __init__(self) -> None:
        self._visited: set[str] = set()
        self._checked: set[tuple] = set()

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

    def visit_root(
        self, zarr_group: ZarrGroup, ome_metadata: Any, loc_prefix: Loc
    ) -> ValidationResult:
        if isinstance(ome_metadata, Scene):
            return self.visit_scene(zarr_group, ome_metadata, loc_prefix)
        return super().visit_root(zarr_group, ome_metadata, loc_prefix)

    # ------------------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------------------

    def visit_image(
        self, zarr_group: ZarrGroup, image_model: Image, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate an image that is not part of a scene."""
        first_visit = _norm(zarr_group.path) not in self._visited
        result = self._visit_image(zarr_group, image_model, loc_prefix)
        if first_visit:
            result = result.merge(self._warn_if_disconnected(image_model, loc_prefix))
        return result

    def visit_label_image(
        self, zarr_group: ZarrGroup, label_image_model: LabelImage, loc_prefix: Loc
    ) -> ValidationResult:
        # The `zarr.json` file for the label image MUST implement the multiscales
        # specification.
        # (so it gets the image checks too; the base class only checks what is
        # specific to label images)
        key = f"label:{_norm(zarr_group.path)}"
        if key in self._visited:
            return ValidationResult()
        self._visited.add(key)
        result = self.visit_image(zarr_group, label_image_model, loc_prefix)
        return result.merge(
            super().visit_label_image(zarr_group, label_image_model, loc_prefix)
        )

    def _visit_image(
        self, zarr_group: ZarrGroup, image_model: Image, loc_prefix: Loc
    ) -> ValidationResult:
        # The base class looks for a child group named `labels`: the spec calls it
        # "the `labels` group", and nothing else in the metadata needs to point to it,
        # so its name is the only practical way to find it.  In 0.6, a multiscale
        # transformation MAY point into a child labels group by `path`, so we also
        # visit labels groups found that way, known by their metadata (not their name).
        if (key := _norm(zarr_group.path)) in self._visited:
            return ValidationResult()
        self._visited.add(key)
        result = super().visit_image(zarr_group, image_model, loc_prefix)
        referenced: set[str] = set()
        for ms_idx, multiscale in enumerate(image_model.multiscales):
            ms_loc = (*loc_prefix, "multiscales", ms_idx)
            result = result.merge(
                self._validate_multiscale_transforms(
                    zarr_group, multiscale, ms_loc, referenced
                )
            )
        for name in sorted(referenced - {"labels"}):
            group = zarr_group.get(name)
            labels_model = self._child_labels_group(zarr_group, name)
            if not isinstance(group, ZarrGroup) or labels_model is None:
                continue  # reported by _validate_multiscale_transforms
            result = result.merge(
                self.visit_labels_group(
                    group, labels_model, (*loc_prefix, name), image_model
                )
            )
        return result

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

    def _validate_multiscale_transforms(
        self,
        zarr_group: ZarrGroup,
        multiscale: Multiscale,
        loc_prefix: Loc,
        referenced: set[str],
    ) -> ValidationResult:
        """Check a multiscale's transformations against the referenced groups.

        Adds the child groups that references into labels go through to `referenced`.
        """
        result = ValidationResult()
        systems = {cs.name: cs for cs in multiscale.coordinateSystems}
        for t_idx, t in enumerate(multiscale.coordinateTransformations or ()):
            t_loc = (*loc_prefix, "coordinateTransformations", t_idx)
            ndims: list[int | None] = []
            for side in ("input", "output"):
                ref = getattr(t, side)
                if ref is None or ref.name is None:  # pragma: no cover
                    ndims.append(None)  # (rejected by the model)
                elif ref.path is None:
                    cs = systems.get(ref.name)
                    ndims.append(cs.ndim if cs else None)
                else:
                    # The other MUST reference a named coordinate system ... in a
                    # child labels group (by `name` and `path`).
                    if (head := _child_head(ref.path)) is not None:
                        referenced.add(head)
                    cs = self._resolve_label_coordinate_system(
                        zarr_group, ref.path, ref.name, (*t_loc, side), result
                    )
                    ndims.append(cs.ndim if cs else None)
            n_in, n_out = ndims
            has_ref = any(getattr(t, side).path for side in ("input", "output"))
            if has_ref and n_in is not None and n_out is not None:
                # (the model could only assume the label image's dimensionality)
                try:
                    check_transform_ndim(t, n_in, n_out)
                except ValueError as e:
                    result.add_error(
                        StorageErrorType.transform_dimension_mismatch,
                        t_loc,
                        f"Transformation to a label image: {e}",
                        ctx={"error": str(e)},
                    )
            result = result.merge(
                self._validate_parameters(zarr_group, t, n_in, n_out, t_loc)
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

    def _resolve_label_coordinate_system(
        self,
        zarr_group: ZarrGroup,
        path: str,
        name: str,
        loc: Loc,
        result: ValidationResult,
    ) -> CoordinateSystem | None:
        """Find coordinate system `name` in the label image at `path`, or add errors."""
        norm = posixpath.normpath(path)
        if _child_head(path) is None:
            result.add_error(
                StorageErrorType.coordinate_system_path_invalid,
                (*loc, "path"),
                f"Path '{path}' is not in a child labels group of this image",
                ctx={"path": path, "expected": "<labels group>/<label image>"},
            )
            return None
        if zarr_group.get(norm) is None:
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
        image = self._image_at(zarr_group, norm, loc, result)
        return None if image is None else self._find_cs(image, path, name, loc, result)

    def _image_at(
        self, zarr_group: ZarrGroup, path: str, loc: Loc, result: ValidationResult
    ) -> Image | None:
        """Return the Image model of the (existing) group at `path`, or add errors."""
        group = zarr_group.get(path)
        model: Any = None
        if isinstance(group, ZarrGroup):
            try:
                model = group.ome_metadata(version=self.VERSION)
            except ValueError as e:
                model = e
        if isinstance(model, Image):
            return model
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

    def _find_cs(
        self, image: Image, path: str, name: str, loc: Loc, result: ValidationResult
    ) -> CoordinateSystem | None:
        if (found := _find_coordinate_system(image, name)) is not None:
            return found[1]
        result.add_error(
            StorageErrorType.coordinate_system_not_found,
            (*loc, "name"),
            f"Coordinate system {name!r} not found in '{path}'",
            ctx={
                "path": path,
                "name": name,
                "available": [
                    cs.name for ms in image.multiscales for cs in ms.coordinateSystems
                ],
            },
        )
        return None

    # ------------------------------------------------------------------------------
    # Graph connectedness
    # ------------------------------------------------------------------------------

    def _warn_if_disconnected(self, image: Image, loc_prefix: Loc) -> ValidationResult:
        # The graph MUST be connected (but need not be complete)
        # (that rule is about the multiscales and scene metadata combined, so outside
        # of a scene we only warn. Each multiscales entry is a graph of its own, as
        # names are resolved within one entry.)
        result = ValidationResult()
        for ms_idx, multiscale in enumerate(image.multiscales):
            nodes, edges = _multiscale_graph(None, ms_idx, multiscale)
            if len(groups := _components(nodes, edges)) > 1:
                result.add_warning(
                    StorageErrorType.graph_not_connected,
                    (*loc_prefix, "multiscales", ms_idx, "coordinateSystems"),
                    "Coordinate systems are not all linked by coordinate "
                    f"transformations: {groups}",
                    ctx={"groups": groups},
                )
        return result

    # ------------------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------------------

    def visit_scene(
        self, zarr_group: ZarrGroup, scene_model: Scene, loc_prefix: Loc
    ) -> ValidationResult:
        """Validate a scene group and the images its transformations reference."""
        result = ValidationResult()
        scene = scene_model.scene
        scene_loc = (*loc_prefix, "scene")
        systems = {cs.name: cs for cs in scene.coordinateSystems or ()}
        nodes: set[Node] = {(None, 0, name) for name in systems}
        edges: list[tuple[Node, Node]] = []
        images: dict[str, Image | None] = {}
        entries: set[tuple[str, int]] = set()  # (image path, multiscales index)
        resolved_all = True

        for t_idx, t in enumerate(scene.coordinateTransformations):
            t_loc = (*scene_loc, "coordinateTransformations", t_idx)
            ends: list[tuple[Node, CoordinateSystem] | None] = []
            for side in ("input", "output"):
                ends.append(
                    self._resolve_scene_ref(
                        zarr_group,
                        getattr(t, side),
                        systems,
                        images,
                        (*t_loc, side),
                        result,
                    )
                )
            start, end = ends
            if start is None or end is None:
                resolved_all = False
                n_in = n_out = None
            else:
                (a, cs_in), (b, cs_out) = start, end
                edges.append((a, b))
                entries |= {(n[0], n[1]) for n in (a, b) if n[0] is not None}
                n_in, n_out = cs_in.ndim, cs_out.ndim
                try:
                    check_transform_ndim(t, n_in, n_out)
                except ValueError as e:
                    result.add_error(
                        StorageErrorType.transform_dimension_mismatch,
                        t_loc,
                        f"Transformation between {_fmt(a)!r} and {_fmt(b)!r}: {e}",
                        ctx={"error": str(e)},
                    )
            result = result.merge(
                self._validate_parameters(
                    zarr_group, t, n_in, n_out, t_loc, in_scene=True
                )
            )

        for path, image in images.items():
            if image is None:
                continue
            if not isinstance(group := zarr_group.get(path), ZarrGroup):
                continue  # pragma: no cover  (checked when resolving the reference)
            result = result.merge(self._visit_image(group, image, (*loc_prefix, path)))
            for ms_idx, multiscale in enumerate(image.multiscales):
                # (only the multiscales entries the scene references: the others are
                # alternative representations of the image, which a scene reference
                # can't name; names resolve to the first entry that has them)
                if (path, ms_idx) not in entries:
                    continue
                ms_nodes, ms_edges = _multiscale_graph(path, ms_idx, multiscale)
                nodes |= ms_nodes
                edges += ms_edges
                edges += _label_edges(path, ms_idx, multiscale, images)

        # The coordinate systems defined in the multiscales metadata and the `scene`
        # metadata combined with the coordinate transformations form a
        # transformations graph. ... The graph MUST be connected
        # (the multiscales are those of the images the scene references, as discussed
        # in ngff-spec#155; we only check once every reference resolved, so a missing
        # image isn't reported twice)
        if resolved_all and len(groups := _components(nodes, edges)) > 1:
            result.add_error(
                StorageErrorType.graph_not_connected,
                scene_loc,
                "The coordinate systems of the scene and its images are not all "
                f"linked by coordinate transformations: {groups}",
                ctx={"groups": groups},
            )
        return result

    def _resolve_scene_ref(
        self,
        zarr_group: ZarrGroup,
        ref: Any,
        systems: dict[str, CoordinateSystem],
        images: dict[str, Image | None],
        loc: Loc,
        result: ValidationResult,
    ) -> tuple[Node, CoordinateSystem] | None:
        """Resolve a scene transformation's input or output, or add errors."""
        if ref.path is None:
            # (the model checked that it is one of the scene's coordinate systems)
            return (None, 0, ref.name), systems[ref.name]
        # If `name` refers to a coordinate system in the multiscale image subgroup
        # specified by `path`, both `path` and `name` MUST be provided.
        path = posixpath.normpath(ref.path)
        if path not in images:
            images[path] = self._scene_image(zarr_group, path, loc, result)
        if (image := images[path]) is None:
            return None
        if (found := _find_coordinate_system(image, ref.name)) is None:
            self._find_cs(image, path, ref.name, loc, result)  # (adds the error)
            return None
        ms_idx, cs = found
        return (path, ms_idx, ref.name), cs

    def _scene_image(
        self, zarr_group: ZarrGroup, path: str, loc: Loc, result: ValidationResult
    ) -> Image | None:
        if path.startswith(("..", "/")):
            result.add_error(
                StorageErrorType.coordinate_system_path_invalid,
                (*loc, "path"),
                f"Path '{path}' is not a subgroup of the scene",
                ctx={"path": path},
            )
            return None
        if zarr_group.get(path) is None:
            result.add_error(
                StorageErrorType.coordinate_system_path_not_found,
                (*loc, "path"),
                f"Path '{path}' not found",
                ctx={"fs_path": _build_fs_path(zarr_group, path), "expected": "group"},
            )
            return None
        return self._image_at(zarr_group, path, loc, result)

    # ------------------------------------------------------------------------------
    # Transformation parameters stored in Zarr (matrices, displacement fields)
    # ------------------------------------------------------------------------------

    def _validate_parameters(
        self,
        holder: ZarrGroup,
        t: CoordinateTransformation,
        n_in: int | None,
        n_out: int | None,
        loc: Loc,
        in_scene: bool = False,
    ) -> ValidationResult:
        """Check the Zarr arrays/groups that `t` (or its parts) names by `path`.

        `holder` is the group whose metadata holds `t`; `n_in` and `n_out` are the
        dimensionality of its input and output, where known.
        """
        result = ValidationResult()
        for sub, sub_in, sub_out, sub_loc in _walk(t, n_in, n_out, loc):
            is_matrix = isinstance(sub, (AffineTransformation, RotationTransformation))
            if not (is_matrix or isinstance(sub, _FIELDS)) or sub.path is None:
                continue
            node = self._find_parameters(holder, sub, sub_loc, in_scene, result)
            if node is None:
                continue
            # (the same parameters for the same dimensionality are checked once, e.g.
            # one field used by both directions of a bijection)
            if (key := (_norm(node.path), sub.type, sub_in, sub_out)) in self._checked:
                continue
            self._checked.add(key)
            if is_matrix:
                result = result.merge(
                    self._validate_matrix(holder, node, sub, sub_in, sub_out, sub_loc)
                )
            else:
                result = result.merge(
                    self._validate_field(node, sub, sub_in, sub_out, sub_loc)
                )
        return result

    def _find_parameters(
        self,
        holder: ZarrGroup,
        t: Any,
        loc: Loc,
        in_scene: bool,
        result: ValidationResult,
    ) -> ZarrGroup | ZarrArray | None:
        """Return the node at `t.path`, or add an error or warning."""
        # (the spec only says "a location in this container": a path is resolved
        # against the group whose metadata holds the transformation, as in the spec's
        # examples. Like ngff-zarr, which refuses them on read and write, we reject a
        # leading "/" and empty, "." or ".." segments: nothing defines what they'd
        # point to)
        path: str = t.path
        ctx: dict = {"path": path}
        if path.startswith("/") or any(p in ("", ".", "..") for p in path.split("/")):
            result.add_error(
                StorageErrorType.transform_parameters_invalid,
                (*loc, "path"),
                f"Path '{path}' (for {t.type}) must be a relative path inside the "
                "group, with no empty, '.' or '..' segments",
                ctx=ctx,
            )
            return None
        if (node := holder.get(path)) is None:
            result.add_error(
                StorageErrorType.transform_parameters_not_found,
                (*loc, "path"),
                f"Path '{path}' (for {t.type}) not found",
                ctx={**ctx, "fs_path": _build_fs_path(holder, path)},
            )
            return None
        if in_scene and path.split("/", 1)[0] != "coordinateTransformations":
            # For transformations that store data or parameters in a Zarr array, those
            # Zarr arrays SHOULD be stored in a Zarr group on the same level as images
            # called "coordinateTransformations".
            result.add_warning(
                StorageErrorType.transform_parameters_location,
                (*loc, "path"),
                f"Parameters of {t.type} at '{path}' SHOULD be stored in the scene's "
                "'coordinateTransformations' group",
                ctx=ctx,
            )
        return node

    def _validate_matrix(
        self,
        holder: ZarrGroup,
        node: ZarrGroup | ZarrArray,
        t: AffineTransformation | RotationTransformation,
        n_in: int | None,
        n_out: int | None,
        loc: Loc,
    ) -> ValidationResult:
        result = ValidationResult()
        is_affine = isinstance(t, AffineTransformation)
        # The matrix MUST be stored as a 2D array ... The array at this path MUST be
        # 2D whose shape MUST be `(M)x(N+1)` (affine) / `N x N` (rotation).
        expected: tuple[int | None, int | None] = (
            (n_out, None if n_in is None else n_in + 1) if is_affine else (n_in, n_in)
        )
        shape = node.metadata.shape if isinstance(node, ZarrArray) else None
        problem = None
        if shape is None:
            problem = "is not a Zarr array"
        elif len(shape) != 2:
            problem = f"has {len(shape)} dimensions, not 2"
        elif not is_affine and shape[0] != shape[1]:
            problem = f"has shape {list(shape)}, which is not square"
        elif any(e is not None and e != s for e, s in zip(expected, shape)):
            want = "x".join("?" if e is None else str(e) for e in expected)
            problem = f"has shape {list(shape)}, expected {want}"
        if problem:
            result.add_error(
                StorageErrorType.transform_parameters_invalid,
                (*loc, "path"),
                f"The {t.type} matrix at '{t.path}' {problem}",
                ctx={"path": t.path, "fs_path": _build_fs_path(holder, t.path or "")},
            )
        return result

    def _validate_field(
        self,
        node: ZarrGroup | ZarrArray,
        t: DisplacementsTransformation | CoordinatesTransformation,
        n_in: int | None,
        n_out: int | None,
        loc: Loc,
    ) -> ValidationResult:
        """Validate the multiscale image holding a displacements/coordinates field."""
        result = ValidationResult()
        kind = (
            "displacement"
            if isinstance(t, DisplacementsTransformation)
            else "coordinate"
        )

        def _error(msg: str, **ctx: Any) -> None:
            result.add_error(
                StorageErrorType.transform_parameters_invalid,
                (*loc, "path"),
                f"The {t.type} field at '{t.path}' {msg}",
                ctx={"path": t.path, **ctx},
            )

        # `coordinates` and `displacements` transformations store a vector field ...
        # in an ome-zarr multiscale group
        # (since ngff-spec#145; the schema descriptions still say "zarr array")
        if not isinstance(node, ZarrGroup):
            _error("is a Zarr array, but must be a multiscale image group")
            return result
        try:
            model: Any = node.ome_metadata(version=self.VERSION)
        except ValueError as e:
            model = e
        if not isinstance(model, Image):
            reason = (
                {"error": model}
                if isinstance(model, Exception)
                else {"type": type(model).__name__}
            )
            _error("does not contain valid Image ('multiscales') metadata", **reason)
            return result
        result = result.merge(self._visit_image(node, model, (*loc, "path")))

        for multiscale in model.multiscales:
            axes = multiscale.axes
            # MUST specify `"type": "displacement"` or `"type": "coordinate"` ... and
            # `"discrete": true` in the multiscale image's axes metadata for the
            # dimension corresponding to the coordinate or displacement vector.
            vector = [i for i, ax in enumerate(axes) if ax.type == kind]
            if len(vector) != 1:
                _error(f"must have exactly one axis of type {kind!r}")
                continue
            idx = vector[0]
            if axes[idx].discrete is not True:
                _error(f"axis {axes[idx].name!r} must specify 'discrete: true'")
            # If the input coordinate system has `N` axes, the multiscale image at
            # location `path` MUST have `N+1` dimensions.
            if n_in is not None and len(axes) != n_in + 1:
                _error(
                    f"has {len(axes)} dimensions, expected {n_in + 1} (input has "
                    f"{n_in} axes)"
                )
                continue
            # For `coordinates` ... the length of the array along the `coordinate`
            # dimension MUST equal `M` ... For `displacements` ... MUST equal `N`
            length = n_in if kind == "displacement" else n_out
            if length is None:
                continue
            for dataset in multiscale.datasets:
                arr = node.get(dataset.path)
                if not isinstance(arr, ZarrArray) or arr.metadata.shape is None:
                    continue  # reported when validating the image
                if (actual := arr.metadata.shape[idx]) != length:
                    _error(
                        f"array '{dataset.path}' has length {actual} along the "
                        f"{kind} axis, expected {length}"
                    )
        return result


# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------


def _norm(path: str) -> str:
    """Normalize a path within the store ("" is the root)."""
    return "" if (norm := posixpath.normpath(path or ".")) == "." else norm


def _child_head(path: str) -> str | None:
    """The first component of `path` if it stays inside the group, else None."""
    norm = posixpath.normpath(path)
    if path.startswith("/") or norm == "." or norm.split("/", 1)[0] == "..":
        return None
    return norm.split("/", 1)[0]


def _label_edges(
    path: str, ms_idx: int, multiscale: Multiscale, images: dict[str, Image | None]
) -> list[tuple[Node, Node]]:
    """Edges from a multiscales entry into label images that a scene references."""
    edges: list[tuple[Node, Node]] = []
    for t in multiscale.coordinateTransformations or ():
        if t.input is None or t.output is None:  # pragma: no cover
            continue
        for ref, other in ((t.input, t.output), (t.output, t.input)):
            if ref.path is None or other.path is not None or not ref.name:
                continue
            target = _norm(posixpath.join(path, ref.path))
            if (label := images.get(target)) is None or other.name is None:
                continue
            if (found := _find_coordinate_system(label, ref.name)) is not None:
                edges.append(((path, ms_idx, other.name), (target, found[0], ref.name)))
    return edges


def _find_coordinate_system(
    image: Image, name: str
) -> tuple[int, CoordinateSystem] | None:
    """(multiscales index, coordinate system) of the first one named `name`."""
    for ms_idx, multiscale in enumerate(image.multiscales):
        for cs in multiscale.coordinateSystems:
            if cs.name == name:
                return ms_idx, cs
    return None


def _multiscale_graph(
    path: str | None, ms_idx: int, multiscale: Multiscale
) -> tuple[set[Node], list[tuple[Node, Node]]]:
    """Nodes and edges of a multiscales entry's own transformations graph."""
    nodes = {(path, ms_idx, cs.name) for cs in multiscale.coordinateSystems}
    edges = [
        ((path, ms_idx, t.input.name), (path, ms_idx, t.output.name))
        for t in multiscale.coordinateTransformations or ()
        # (references into child label images only matter when a scene references
        # the label image too: see _label_edges)
        if t.input
        and t.output
        and t.input.path is None
        and t.output.path is None
        and t.input.name
        and t.output.name
    ]
    return nodes, edges


def _components(
    nodes: Iterable[Node], edges: Iterable[tuple[Node, Node]]
) -> list[list[str]]:
    """Connected components of the (undirected) graph, as sorted lists of names."""
    parent: dict[Node, Node] = {n: n for n in nodes}

    def find(n: Node) -> Node:
        while parent.setdefault(n, n) != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[Node, list[str]] = {}
    for n in parent:
        groups.setdefault(find(n), []).append(_fmt(n))
    return sorted(sorted(g) for g in groups.values())


def _fmt(node: Node) -> str:
    path, ms_idx, name = node
    if path is None:
        return name
    return f"{path}:{name}" if ms_idx == 0 else f"{path}[{ms_idx}]:{name}"


def _walk(
    t: Any, n_in: int | None, n_out: int | None, loc: Loc
) -> Iterator[tuple[Any, int | None, int | None, Loc]]:
    """Yield `t` and every transformation nested in it, with their dimensionality.

    (inferred as in `check_transform_ndim`; None where it can't be)
    """
    yield t, n_in, n_out, loc
    if isinstance(t, BijectionTransformation):
        yield from _walk(t.forward, n_in, n_out, (*loc, "forward"))
        yield from _walk(t.inverse, n_out, n_in, (*loc, "inverse"))
    elif isinstance(t, SequenceTransformation):
        n = n_in
        steps = t.transformations
        for i, step in enumerate(steps):
            if n is None:
                n = _own_ndim(step)[0]
            n_next = n_out if i == len(steps) - 1 else _output_ndim(step, n)
            yield from _walk(step, n, n_next, (*loc, "transformations", i))
            n = n_next
    elif isinstance(t, ByDimensionTransformation):
        for i, item in enumerate(t.transformations):
            yield from _walk(
                item.transformation,
                len(item.inputAxes),
                len(item.outputAxes),
                (*loc, "transformations", i, "transformation"),
            )
