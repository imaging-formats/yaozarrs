"""Tree representation for zarr groups.

Provides a tree view showing metadata files (zarr.json for v3, .zgroup/.zattrs/.zarray
for v2) with OME type annotations. Uses rich for enhanced rendering if available,
otherwise falls back to standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from yaozarrs import v04, v05
from yaozarrs._zarr import ZarrArray, ZarrGroup

if TYPE_CHECKING:
    from rich.tree import Tree


# Icons for different node types
ICON_ARRAY = "📊"  # Array nodes
ICON_OME_GROUP = "🅾️"  # OME-zarr group nodes (microscope for bio data)
ICON_GROUP = "📁"  # Regular (non-ome-zarr) group nodes
ICON_ELLIPSIS = "⋯"  # Ellipsis for truncated children

# Tree drawing characters (for non-rich output)
TREE_BRANCH = "├── "
TREE_LAST = "└── "
TREE_PIPE = "│   "
TREE_SPACE = "    "


# =============================================================================
# Intermediate Representation
# =============================================================================


@dataclass
class TreeNode:
    """Intermediate representation for tree nodes.

    This class captures all the data needed for rendering a tree node,
    independent of the rendering method (plain text or rich).
    """

    name: str
    icon: str
    node_type: str  # "group" or "array"
    # For groups: list of (filename, ome_annotation) tuples
    metadata_files: list[tuple[str, str]] = field(default_factory=list)
    children: list[TreeNode] = field(default_factory=list)
    truncated: bool = False
    # For arrays
    dtype: str | None = None
    shape: tuple[int, ...] | None = None


# =============================================================================
# Data Fetching (builds intermediate representation)
# =============================================================================


def _natural_sort_key(s: str) -> list:
    """Return a key for natural sorting (numeric-aware).

    Splits string into text and numeric parts for proper ordering:
    "A/1", "A/2", "A/10" instead of "A/1", "A/10", "A/2"
    """
    import re

    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _get_node_icon(node: ZarrGroup | ZarrArray) -> str:
    """Get the icon for a node based on its type."""
    if isinstance(node, ZarrArray):
        return ICON_ARRAY
    elif isinstance(node, ZarrGroup):
        # Check if it's an OME-zarr group
        try:
            if node._local_ome_version() is not None:
                return ICON_OME_GROUP
        except Exception:
            pass
        return ICON_GROUP
    return ""  # pragma: no cover


def _get_ome_type_annotation(node: ZarrGroup | ZarrArray) -> str:
    """Get the OME type annotation for a node's metadata file.

    Returns a string like '<- v05.Image' or empty string if no OME metadata.
    """
    if not isinstance(node, ZarrGroup):
        return ""

    try:
        ome_meta = node.ome_metadata()
    except Exception:
        return ""

    if ome_meta is None:
        return ""

    # Get the class name and module
    cls = type(ome_meta)
    module = cls.__module__

    # Extract version from module (e.g., 'yaozarrs.v05._image' -> 'v05')
    if ".v05." in module:
        version = "v05"
    elif ".v04." in module:
        version = "v04"
    else:
        version = ""

    if version:
        return f"  <- {version}.{cls.__name__}"
    return f"  <- {cls.__name__}"


def _get_metadata_files(node: ZarrGroup | ZarrArray) -> list[str]:
    """Get the list of metadata files for a node."""
    if node.zarr_format >= 3:
        return ["zarr.json"]
    else:
        # v2: different files for groups vs arrays
        if isinstance(node, ZarrArray):
            return [".zarray", ".zattrs"]
        else:
            return [".zgroup", ".zattrs"]


def _get_children_from_ome_metadata(group: ZarrGroup) -> list[str] | None:
    """Extract child keys from OME metadata.

    For remote HTTP stores where directory listing isn't available,
    we can infer children from the OME metadata structure.

    Returns None if no OME metadata or can't determine children.
    """
    ome_meta = group.ome_metadata()
    if ome_meta is None:
        return None

    children: list[str] = []

    # v0.5 Image or LabelImage: children are multiscale dataset paths
    if isinstance(ome_meta, v05.Image):
        for ms in ome_meta.multiscales:
            for ds in ms.datasets:
                if ds.path not in children:
                    children.append(ds.path)
        # Also check for labels subgroup
        if "labels" in group:
            children.append("labels")
        return children

    # v0.5 Plate: children are well paths (like "A/1", "B/2")
    if isinstance(ome_meta, v05.Plate):
        for well in ome_meta.plate.wells:
            if well.path not in children:
                children.append(well.path)
        return children

    # v0.5 Well: children are field-of-view paths
    if isinstance(ome_meta, v05.Well):
        for img in ome_meta.well.images:
            if img.path not in children:
                children.append(img.path)
        return children

    # v0.5 LabelsGroup: children are label names
    if isinstance(ome_meta, v05.LabelsGroup):
        return list(ome_meta.labels)

    # v0.4 Image: similar to v0.5
    if isinstance(ome_meta, v04.Image):
        for ms in ome_meta.multiscales:
            for ds in ms.datasets:
                if ds.path not in children:
                    children.append(ds.path)
        # Check for labels
        if "labels" in group:
            children.append("labels")
        return children

    # v0.4 Plate: children are well paths (like "A/1", "B/2")
    if isinstance(ome_meta, v04.Plate):
        if ome_meta.plate:
            for well in ome_meta.plate.wells:
                if well.path not in children:
                    children.append(well.path)
        return children

    # v0.4 Well
    if isinstance(ome_meta, v04.Well):
        if ome_meta.well:
            for img in ome_meta.well.images:
                if img.path not in children:
                    children.append(img.path)
        return children

    # v0.4 Bf2Raw or bioformats2raw: probe for numbered children
    if isinstance(ome_meta, v04.Bf2Raw):
        # bioformats2raw layouts have numbered children (0, 1, 2, ...)
        for i in range(100):  # reasonable upper bound
            if str(i) in group:
                children.append(str(i))
            else:
                break
        return children if children else None

    # v0.4 Labels
    if hasattr(ome_meta, "labels") and ome_meta.labels:
        return list(ome_meta.labels)  # ty: ignore

    return None


def _get_child_keys(group: ZarrGroup) -> list[str]:
    """Get sorted list of child keys from a zarr group.

    First tries filesystem listing (works for local stores).
    Falls back to OME metadata extraction for remote HTTP stores.
    """
    children: set[str] = set()
    store = group._store

    # Try filesystem listing (works for local and some remote stores)
    # Skip for HTTP/cloud stores that don't support directory listing
    if hasattr(store, "_fsmap") and hasattr(store._fsmap, "fs"):
        fs = store._fsmap.fs

        # Check filesystem protocol - skip listing for HTTP/cloud stores
        fs_protocol = getattr(fs, "protocol", "")
        protocols = [fs_protocol] if isinstance(fs_protocol, str) else list(fs_protocol)
        skip_listing = any(p in ("http", "https", "s3", "gs", "az") for p in protocols)

        if not skip_listing:
            root = store._fsmap.root
            prefix = f"{group._path}/" if group._path else ""
            full_path = f"{root}/{prefix}".rstrip("/") if prefix else root

            try:
                entries = fs.ls(full_path, detail=False)
                for entry in entries:
                    name = entry.rstrip("/").rsplit("/", 1)[-1]
                    # Skip metadata files
                    if name.startswith(".") or name == "zarr.json":
                        continue
                    children.add(name)
            except Exception:
                pass

    # If filesystem listing didn't work, try OME metadata
    if not children:
        ome_children = _get_children_from_ome_metadata(group)
        if ome_children:
            children.update(ome_children)

    # Sort with natural ordering (numeric awareness)
    child_list = list(children)
    return sorted(child_list, key=_natural_sort_key)


def _prefetch_plate_hierarchy(
    group: ZarrGroup,
    depth: int | None,
    max_per_level: int | None,
) -> None:
    """Prefetch all metadata for a plate hierarchy in stages.

    This uses a multi-stage approach:
    1. Prefetch all well metadata (known from plate metadata)
    2. After wells are cached, prefetch all image metadata
    3. After images are cached, prefetch all array metadata

    This batches requests at each level for efficient network usage.
    """
    ome_meta = group.ome_metadata()
    if ome_meta is None:
        return

    # Stage 1: Get well paths from plate metadata
    if isinstance(ome_meta, (v05.Plate, v04.Plate)):
        wells = (
            ome_meta.plate.wells
            if isinstance(ome_meta, v05.Plate)
            else (ome_meta.plate.wells if ome_meta.plate else [])
        )
        well_paths = sorted([w.path for w in wells], key=_natural_sort_key)
        if max_per_level is not None:
            well_paths = well_paths[:max_per_level]

        if not well_paths:
            return

        # Prefetch all well metadata at once
        group.prefetch_children(well_paths)

        # Stage 2: If depth allows, get image paths from well metadata
        if depth is not None and depth <= 1:
            return

        image_paths: list[str] = []
        for wp in well_paths:
            try:
                well = group[wp]
                well_meta = well.ome_metadata()
                if isinstance(well_meta, (v05.Well, v04.Well)):
                    images = (
                        well_meta.well.images
                        if isinstance(well_meta, v05.Well)
                        else (well_meta.well.images if well_meta.well else [])
                    )
                    img_paths = [img.path for img in images]
                    img_paths = sorted(img_paths, key=_natural_sort_key)
                    if max_per_level is not None:
                        img_paths = img_paths[:max_per_level]
                    for ip in img_paths:
                        image_paths.append(f"{wp}/{ip}")
            except Exception:
                continue

        if image_paths:
            group.prefetch_children(image_paths)

        # Stage 3: If depth allows, get array paths from image metadata
        if depth is not None and depth <= 2:
            return

        array_paths: list[str] = []
        for img_path in image_paths:
            try:
                image = group[img_path]
                img_meta = image.ome_metadata()
                if isinstance(img_meta, (v05.Image, v04.Image)):
                    for ms in img_meta.multiscales:
                        ds_paths = sorted(
                            [ds.path for ds in ms.datasets], key=_natural_sort_key
                        )
                        if max_per_level is not None:
                            ds_paths = ds_paths[:max_per_level]
                        for dp in ds_paths:
                            array_paths.append(f"{img_path}/{dp}")
            except Exception:
                continue

        if array_paths:
            group.prefetch_children(array_paths)

    elif isinstance(ome_meta, (v05.Image, v04.Image)):
        # For images, just prefetch array paths
        array_paths: list[str] = []
        for ms in ome_meta.multiscales:
            ds_paths = sorted([ds.path for ds in ms.datasets], key=_natural_sort_key)
            if max_per_level is not None:
                ds_paths = ds_paths[:max_per_level]
            array_paths.extend(ds_paths)

        if array_paths:
            group.prefetch_children(array_paths)


def _build_tree(
    group: ZarrGroup,
    depth: int | None,
    max_per_level: int | None,
    current_depth: int = 0,
    name: str | None = None,
    _prefetched: bool = False,
) -> TreeNode:
    """Build the intermediate tree representation.

    Parameters
    ----------
    group : ZarrGroup
        The group to build a tree for.
    depth : int | None
        Maximum depth to traverse (None for unlimited).
    max_per_level : int | None
        Maximum children per level (None for unlimited).
    current_depth : int
        Current traversal depth.
    name : str | None
        Name to display for this node (defaults to store path basename).
    _prefetched : bool
        Internal flag indicating if descendants have already been prefetched.

    Returns
    -------
    TreeNode
        The intermediate tree representation.
    """
    # On first call (root), prefetch all descendants using staged approach
    if not _prefetched and current_depth == 0:
        _prefetch_plate_hierarchy(group, depth, max_per_level)

    # Determine node name
    if name is None:
        name = group.store_path.rsplit("/", 1)[-1] or group.store_path

    # Get icon and OME annotation
    icon = _get_node_icon(group)
    ome_annotation = _get_ome_type_annotation(group)

    # Build metadata files list with annotations
    metadata_file_names = _get_metadata_files(group)
    metadata_files: list[tuple[str, str]] = []
    for mf in metadata_file_names:
        if mf in ("zarr.json", ".zattrs"):
            metadata_files.append((mf, ome_annotation))
        else:
            metadata_files.append((mf, ""))

    # Create the node
    node = TreeNode(
        name=name,
        icon=icon,
        node_type="group",
        metadata_files=metadata_files,
    )

    # Get children if within depth limit
    if depth is None or current_depth < depth:
        child_keys = _get_child_keys(group)

        # Apply max_per_level limit
        if max_per_level is not None and len(child_keys) > max_per_level:
            child_keys = child_keys[:max_per_level]
            node.truncated = True

        # Build child nodes (metadata should already be prefetched)
        for key in child_keys:
            try:
                child = group[key]
            except (KeyError, Exception):
                continue

            if isinstance(child, ZarrGroup):
                child_node = _build_tree(
                    child,
                    depth,
                    max_per_level,
                    current_depth + 1,
                    key,
                    _prefetched=True,
                )
                node.children.append(child_node)
            elif isinstance(child, ZarrArray):
                # Array node
                array_node = TreeNode(
                    name=key,
                    icon=_get_node_icon(child),
                    node_type="array",
                    dtype=str(child._metadata.data_type),
                    shape=child._metadata.shape,
                )
                node.children.append(array_node)

    return node


# =============================================================================
# Rendering Functions
# =============================================================================


def _render_plain(
    node: TreeNode,
    prefix: str = "",
    is_last: bool = True,
    is_root: bool = True,
) -> list[str]:
    """Render a TreeNode to plain text lines.

    Parameters
    ----------
    node : TreeNode
        The tree node to render.
    prefix : str
        Current line prefix for tree drawing.
    is_last : bool
        Whether this is the last item at its level.
    is_root : bool
        Whether this is the root node.

    Returns
    -------
    list[str]
        Lines of the tree representation.
    """
    lines: list[str] = []

    # Render the node header
    if is_root:
        lines.append(f"{node.icon} {node.name}")
        node_prefix = ""
    else:
        lines.append(f"{prefix}{node.icon} {node.name}")
        node_prefix = prefix[:-4] + (TREE_SPACE if is_last else TREE_PIPE)

    # For arrays, add dtype and shape info and return (no children/metadata)
    if node.node_type == "array":
        # Modify the last line to include array info
        if node.dtype and node.shape:
            lines[-1] = lines[-1] + f" ({node.dtype}, {node.shape})"
        return lines

    # Build list of all items to render (metadata files + children)
    all_items: list[tuple[str, object]] = []  # (type, data)
    for mf_name, mf_annotation in node.metadata_files:
        all_items.append(("meta", (mf_name, mf_annotation)))
    for child in node.children:
        all_items.append(("child", child))

    # Render each item
    total_items = len(all_items)
    for i, (item_type, item_data) in enumerate(all_items):
        is_item_last = (i == total_items - 1) and not node.truncated

        if is_root:
            line_prefix = TREE_LAST if is_item_last else TREE_BRANCH
        else:
            line_prefix = node_prefix + (TREE_LAST if is_item_last else TREE_BRANCH)

        if item_type == "meta":
            mf_name, mf_annotation = item_data  # type: ignore[misc]
            lines.append(f"{line_prefix}{mf_name}{mf_annotation}")
        else:
            # Child node - recurse
            child_node: TreeNode = item_data  # type: ignore[assignment]
            child_lines = _render_plain(
                child_node,
                prefix=line_prefix,
                is_last=is_item_last,
                is_root=False,
            )
            lines.extend(child_lines)

    # Add ellipsis if truncated
    if node.truncated:
        if is_root:
            lines.append(f"{TREE_LAST}{ICON_ELLIPSIS} ...")
        else:
            lines.append(f"{node_prefix}{TREE_LAST}{ICON_ELLIPSIS} ...")

    return lines


def _render_rich(node: TreeNode) -> Tree:
    """Render a TreeNode to a rich Tree object.

    Parameters
    ----------
    node : TreeNode
        The tree node to render.

    Returns
    -------
    Tree
        Rich Tree object that can be printed directly.
    """
    from rich.tree import Tree

    def add_contents(tree: Tree, tree_node: TreeNode) -> None:
        """Recursively add contents to a rich tree."""
        # For arrays, nothing more to add (info is in the label)
        if tree_node.node_type == "array":
            return

        # Add metadata files
        for mf_name, mf_annotation in tree_node.metadata_files:
            if mf_annotation:
                tree.add(f"[dim]{mf_name}[/dim][cyan]{mf_annotation}[/cyan]")
            else:
                tree.add(f"[dim]{mf_name}[/dim]")

        # Add children
        for child in tree_node.children:
            if child.node_type == "array":
                label = (
                    f"[bold]{child.icon} {child.name}[/bold] "
                    f"[dim]({child.dtype}, {child.shape})[/dim]"
                )
                tree.add(label)
            else:
                child_tree = tree.add(f"[bold]{child.icon} {child.name}[/bold]")
                add_contents(child_tree, child)

        # Add truncation indicator
        if tree_node.truncated:
            tree.add(f"[dim italic]{ICON_ELLIPSIS} ...[/dim italic]")

    # Create root tree
    root = Tree(f"[bold]{node.icon} {node.name}[/bold]")
    add_contents(root, node)
    return root


# =============================================================================
# Public API
# =============================================================================


def print_tree(
    group: ZarrGroup,
    depth: int | None = None,
    max_per_level: int | None = None,
) -> None:
    """Print a tree representation of the zarr group hierarchy.

    Uses rich library for colored output if available,
    otherwise falls back to plain text.

    Parameters
    ----------
    group : ZarrGroup
        The zarr group to render as a tree.
    depth : int | None, optional
        Maximum depth to traverse. None for unlimited depth.
    max_per_level : int | None, optional
        Maximum number of children to show at each level.
    """
    tree_data = _build_tree(group, depth, max_per_level)

    try:
        from rich import print as rprint

        rich_tree = _render_rich(tree_data)
        rprint(rich_tree)
    except ImportError:
        lines = _render_plain(tree_data)
        print("\n".join(lines))


def render_tree(
    group: ZarrGroup,
    depth: int | None = None,
    max_per_level: int | None = None,
) -> str:
    """Render a tree representation of the zarr group hierarchy as a string.

    Parameters
    ----------
    group : ZarrGroup
        The zarr group to render as a tree.
    depth : int | None, optional
        Maximum depth to traverse. None for unlimited depth.
        Using a smaller depth improves performance for large hierarchies.
    max_per_level : int | None, optional
        Maximum number of children to show at each level.
        Additional children are indicated with an ellipsis.
        None for unlimited children.

    Returns
    -------
    str
        String representation of the tree (without ANSI colors).

    Notes
    -----
    For colored output, use `print_tree()` instead.

    Shows metadata files (zarr.json for v3, .zgroup/.zattrs/.zarray for v2)
    with OME type annotations like '<- v05.Image'.

    Icons:
    - 📊 Array nodes
    - 🔬 OME-zarr group nodes (groups with OME metadata)
    - 📁 Regular group nodes
    - ⋯  Indicates truncated children (when max_per_level is exceeded)
    """
    tree_data = _build_tree(group, depth, max_per_level)
    lines = _render_plain(tree_data)
    return "\n".join(lines)
