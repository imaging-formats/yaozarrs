"""Tree representation for zarr groups.

Provides a tree view showing metadata files (zarr.json for v3, .zgroup/.zattrs/.zarray
for v2) with OME type annotations. Uses rich for enhanced rendering if available,
otherwise falls back to standard library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.tree import Tree

    from yaozarrs._zarr import ZarrArray, ZarrGroup

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
    from yaozarrs._zarr import ZarrArray, ZarrGroup

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
    return ""


def _get_ome_type_annotation(node: ZarrGroup | ZarrArray) -> str:
    """Get the OME type annotation for a node's metadata file.

    Returns a string like '<- v05.Image' or empty string if no OME metadata.
    """
    from yaozarrs._zarr import ZarrGroup

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
    from yaozarrs._zarr import ZarrArray

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
    from yaozarrs import v04, v05

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

    # Try filesystem listing first (works for local and some remote stores)
    if hasattr(store, "_fsmap") and hasattr(store._fsmap, "fs"):
        fs = store._fsmap.fs
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


def _build_tree_plain(
    group: ZarrGroup,
    depth: int | None,
    max_per_level: int | None,
    current_depth: int = 0,
    prefix: str = "",
    is_last: bool = True,
    name: str | None = None,
) -> list[str]:
    """Build plain text tree representation with metadata files.

    Parameters
    ----------
    group : ZarrGroup
        The group to render.
    depth : int | None
        Maximum depth to traverse (None for unlimited).
    max_per_level : int | None
        Maximum children per level (None for unlimited).
    current_depth : int
        Current traversal depth.
    prefix : str
        Prefix for the current line (for tree drawing).
    is_last : bool
        Whether this is the last child at this level.
    name : str | None
        Name to display for this node.

    Returns
    -------
    list[str]
        Lines of the tree representation.
    """
    from yaozarrs._zarr import ZarrArray, ZarrGroup

    lines: list[str] = []

    # Root node
    if name is None:
        name = group.store_path.rsplit("/", 1)[-1] or group.store_path
        icon = _get_node_icon(group)
        lines.append(f"{icon} {name}")
        node_prefix = ""
    else:
        icon = _get_node_icon(group)
        lines.append(f"{prefix}{icon} {name}")
        node_prefix = prefix[:-4] + (TREE_SPACE if is_last else TREE_PIPE)

    # Get children and metadata files
    if depth is None or current_depth < depth:
        child_keys = _get_child_keys(group)
    else:
        child_keys = []
    metadata_files = _get_metadata_files(group)
    ome_annotation = _get_ome_type_annotation(group)

    # Determine what items we have at this level
    # For groups: metadata files first, then child nodes
    all_items: list[tuple[str, str]] = []  # (type, name)
    for mf in metadata_files:
        all_items.append(("meta", mf))
    for ck in child_keys:
        all_items.append(("child", ck))

    # Prefetch children for efficiency
    if child_keys:
        group.prefetch_children(child_keys)

    # Apply max_per_level limit to children only
    truncated = False
    if max_per_level is not None and len(child_keys) > max_per_level:
        # Rebuild items with truncated children
        all_items = []
        for mf in metadata_files:
            all_items.append(("meta", mf))
        for ck in child_keys[:max_per_level]:
            all_items.append(("child", ck))
        truncated = True

    # Process each item
    for i, (item_type, item_name) in enumerate(all_items):
        is_item_last = (i == len(all_items) - 1) and not truncated

        if current_depth == 0:
            line_prefix = TREE_LAST if is_item_last else TREE_BRANCH
        else:
            line_prefix = node_prefix + (TREE_LAST if is_item_last else TREE_BRANCH)

        if item_type == "meta":
            # Metadata file - add OME annotation for .zattrs or zarr.json
            if item_name in ("zarr.json", ".zattrs"):
                lines.append(f"{line_prefix}{item_name}{ome_annotation}")
            else:
                lines.append(f"{line_prefix}{item_name}")
        else:
            # Child node
            try:
                child = group[item_name]
            except (KeyError, Exception):
                continue

            if isinstance(child, ZarrGroup):
                child_lines = _build_tree_plain(
                    child,
                    depth,
                    max_per_level,
                    current_depth + 1,
                    line_prefix,
                    is_item_last,
                    item_name,
                )
                lines.extend(child_lines)
            elif isinstance(child, ZarrArray):
                # Array node (no metadata files shown for arrays)
                icon = _get_node_icon(child)
                shape = child._metadata.shape
                dtype = child._metadata.data_type
                lines.append(f"{line_prefix}{icon} {item_name} ({dtype}, {shape})")

    # Add ellipsis if truncated
    if truncated:
        if current_depth == 0:
            lines.append(f"{TREE_LAST}{ICON_ELLIPSIS} ...")
        else:
            lines.append(f"{node_prefix}{TREE_LAST}{ICON_ELLIPSIS} ...")

    return lines


def _build_rich_tree(
    group: ZarrGroup,
    depth: int | None,
    max_per_level: int | None,
) -> Tree:
    """Build a rich Tree object for the zarr group hierarchy.

    Parameters
    ----------
    group : ZarrGroup
        The group to render.
    depth : int | None
        Maximum depth to traverse (None for unlimited).
    max_per_level : int | None
        Maximum children per level (None for unlimited).

    Returns
    -------
    Tree
        Rich Tree object that can be printed directly.
    """
    from rich.tree import Tree

    from yaozarrs._zarr import ZarrArray, ZarrGroup

    def add_node_contents(
        tree_node: Tree,
        zarr_node: ZarrGroup | ZarrArray,
        current_depth: int,
    ) -> None:
        """Add metadata files and children to a tree node."""
        # Only show metadata files for groups, not arrays
        if isinstance(zarr_node, ZarrArray):
            return

        metadata_files = _get_metadata_files(zarr_node)
        ome_annotation = _get_ome_type_annotation(zarr_node)

        # Add metadata files for groups
        for mf in metadata_files:
            if mf in ("zarr.json", ".zattrs") and ome_annotation:
                tree_node.add(f"[dim]{mf}[/dim][cyan]{ome_annotation}[/cyan]")
            else:
                tree_node.add(f"[dim]{mf}[/dim]")

        # Check depth limit for groups
        if depth is not None and current_depth >= depth:
            return

        child_keys = _get_child_keys(zarr_node)
        if not child_keys:
            return

        # Prefetch children for efficiency
        zarr_node.prefetch_children(child_keys)

        # Apply max_per_level limit
        truncated = False
        if max_per_level is not None and len(child_keys) > max_per_level:
            child_keys = child_keys[:max_per_level]
            truncated = True

        for key in child_keys:
            try:
                child = zarr_node[key]
            except (KeyError, Exception):
                continue

            icon = _get_node_icon(child)

            if isinstance(child, ZarrGroup):
                child_tree = tree_node.add(f"[bold]{icon} {key}[/bold]")
                add_node_contents(child_tree, child, current_depth + 1)
            elif isinstance(child, ZarrArray):
                shape = child._metadata.shape
                dtype = child._metadata.data_type
                label = f"[bold]{icon} {key}[/bold] [dim]({dtype}, {shape})[/dim]"
                tree_node.add(label)

        if truncated:
            tree_node.add(f"[dim italic]{ICON_ELLIPSIS} ...[/dim italic]")

    # Create root tree
    root_name = group.store_path.rsplit("/", 1)[-1] or group.store_path
    icon = _get_node_icon(group)
    tree = Tree(f"[bold]{icon} {root_name}[/bold]")

    add_node_contents(tree, group, 0)

    return tree


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
    try:
        from rich import print as rprint

        tree = _build_rich_tree(group, depth, max_per_level)
        rprint(tree)
    except ImportError:
        lines = _build_tree_plain(group, depth, max_per_level)
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
    lines = _build_tree_plain(group, depth, max_per_level)
    return "\n".join(lines)
