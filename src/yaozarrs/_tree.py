"""Tree representation for zarr groups.

Provides a tree view showing metadata files (zarr.json for v3, .zgroup/.zattrs/.zarray
for v2) with OME type annotations. Uses rich for enhanced rendering if available,
otherwise falls back to standard library.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from yaozarrs._zarr import (
    _BF2RAW,
    _IMAGE,
    _LABELS_GROUP,
    _SCENE,
    ZarrArray,
    ZarrGroup,
    _numbered_children,
    bioformats2raw_series,
    nodes_below,
)

if TYPE_CHECKING:
    from rich.tree import Tree

ICON_ARRAY = "📦"
ICON_IMAGE = "🖼️"
ICON_LABEL = "🏷️"
ICON_OME_GROUP = "🅾️"
ICON_GROUP = "📁"
ICON_ELLIPSIS = "⋯"

TREE_BRANCH = "├── "
TREE_LAST = "└── "
TREE_PIPE = "│   "
TREE_SPACE = "    "


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


def _natural_sort_key(s: str) -> list:
    """Natural sort key: "A/1", "A/2", "A/10" instead of "A/1", "A/10", "A/2"."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


def _get_node_icon(node: ZarrGroup | ZarrArray) -> str:
    """Get the icon for a node based on its type."""
    if isinstance(node, ZarrArray):
        return ICON_ARRAY
    try:
        ome = node.ome_metadata()
    except Exception:
        return ICON_GROUP
    if ome is None:
        return ICON_GROUP
    if isinstance(ome, _IMAGE):
        return ICON_IMAGE
    if isinstance(ome, _LABELS_GROUP):
        return ICON_LABEL
    return ICON_OME_GROUP


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
    match = re.search(r"\.(v\d\d)\.", module)
    version = match.group(1) if match else ""

    if version:
        return f"  <- {version}.{cls.__name__}"
    return f"  <- {cls.__name__}"


def _get_metadata_files(node: ZarrGroup | ZarrArray) -> list[str]:
    """Get the list of metadata files for a node."""
    if node.zarr_format >= 3:
        return ["zarr.json"]
    if isinstance(node, ZarrArray):
        return [".zarray", ".zattrs"]
    return [".zgroup", ".zattrs"]


def _metadata_children(group: ZarrGroup) -> tuple[list[str], list[str]]:
    """Return the children of `group` that its OME metadata names (or may have).

    For stores that can't be listed (e.g. http): the first list holds the children
    that the metadata names, the second those `group` may also have (e.g. `labels`),
    which have to be looked for.
    """
    try:
        ome_meta = group.ome_metadata()
    except Exception:
        return [], []

    if isinstance(ome_meta, _BF2RAW):
        try:
            return bioformats2raw_series(group), []
        except Exception:  # (e.g. invalid `OME` metadata: as validation does)
            return _numbered_children(group), []

    named, maybe = nodes_below(ome_meta)
    if isinstance(ome_meta, _SCENE):
        # (where the spec says to keep the arrays its transformations use)
        maybe.append("coordinateTransformations")
    return named, maybe


def _listed_children(group: ZarrGroup) -> list[str]:
    """Return the names of the children of `group` from a directory listing.

    Empty if the store can't be listed (http and cloud stores, and mappings).
    """
    mapper = group._store._mapper
    if not hasattr(mapper, "fs"):
        return []
    protocol = getattr(mapper.fs, "protocol", "")
    protocols = [protocol] if isinstance(protocol, str) else list(protocol)
    if any(p in ("http", "https", "s3", "gs", "az") for p in protocols):
        return []

    full_path = f"{mapper.root}/{group._path}" if group._path else mapper.root
    try:
        entries = mapper.fs.ls(full_path, detail=False)
    except Exception:
        return []
    names = (entry.rstrip("/").rsplit("/", 1)[-1] for entry in entries)
    # (skip metadata files)
    return [n for n in names if not n.startswith(".") and n != "zarr.json"]


def _group_node(group: ZarrGroup, name: str) -> TreeNode:
    ome_annotation = _get_ome_type_annotation(group)
    return TreeNode(
        name=name,
        icon=_get_node_icon(group),
        node_type="group",
        metadata_files=[
            (mf, ome_annotation if mf in ("zarr.json", ".zattrs") else "")
            for mf in _get_metadata_files(group)
        ],
    )


def _array_node(array: ZarrArray, name: str) -> TreeNode:
    return TreeNode(
        name=name,
        icon=ICON_ARRAY,
        node_type="array",
        dtype=str(array._metadata.data_type),
        shape=array._metadata.shape,
    )


def _build_tree(
    group: ZarrGroup, depth: int | None, max_per_level: int | None
) -> TreeNode:
    """Build the intermediate tree representation.

    One level at a time, fetching the metadata of all the nodes in a level at once
    (a couple of round trips per level on remote stores).

    Parameters
    ----------
    group : ZarrGroup
        The group to build a tree for.
    depth : int | None
        Maximum depth to traverse (None for unlimited).
    max_per_level : int | None
        Maximum children per level (None for unlimited).

    Returns
    -------
    TreeNode
        The intermediate tree representation.
    """
    display_name = group._display_name()
    root = _group_node(group, display_name.rsplit("/", 1)[-1] or display_name)

    def _join(rel: str, key: str) -> str:
        return f"{rel}/{key}" if rel else key

    # (path relative to `group`, group, its tree node) for every group in a level
    level: list[tuple[str, ZarrGroup, TreeNode]] = [("", group, root)]
    # (normalized paths of the groups queued so far: metadata can point back up)
    seen = {"."}
    current_depth = 0
    while level and (depth is None or current_depth < depth):
        children = [
            (keys, []) if (keys := _listed_children(grp)) else _metadata_children(grp)
            for _, grp, _ in level
        ]
        if group._store._batched:  # (else the nodes are read as they're built)
            # (one batch for the level: each group's first children, and those it
            # may have, which can only push others out of the first `max_per_level`)
            group.prefetch_children(
                _join(rel, key)
                for (rel, _, _), (keys, maybe) in zip(level, children, strict=True)
                for key in [
                    *sorted(set(keys), key=_natural_sort_key)[:max_per_level],
                    *maybe,
                ]
            )

        next_level: list[tuple[str, ZarrGroup, TreeNode]] = []
        for (rel, grp, node), (keys, maybe) in zip(level, children, strict=True):
            keys = sorted(
                {*keys, *(k for k in maybe if k in grp)}, key=_natural_sort_key
            )
            if max_per_level is not None and len(keys) > max_per_level:
                keys = keys[:max_per_level]
                node.truncated = True
            for key in keys:
                try:
                    child = grp[key]
                except Exception:
                    continue
                if isinstance(child, ZarrGroup):
                    child_node = _group_node(child, key)
                    path = _join(rel, key)
                    if (norm := posixpath.normpath(path)) not in seen:
                        seen.add(norm)
                        next_level.append((path, child, child_node))
                else:
                    child_node = _array_node(child, key)
                node.children.append(child_node)
        level = next_level
        current_depth += 1
    return root


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

    n_meta = len(node.metadata_files)
    total_items = n_meta + len(node.children)

    def _line_prefix(is_item_last: bool) -> str:
        if is_root:
            return TREE_LAST if is_item_last else TREE_BRANCH
        return node_prefix + (TREE_LAST if is_item_last else TREE_BRANCH)

    # Render metadata files
    for i, (mf_name, mf_annotation) in enumerate(node.metadata_files):
        is_item_last = (i == total_items - 1) and not node.truncated
        lines.append(f"{_line_prefix(is_item_last)}{mf_name}{mf_annotation}")

    # Render child nodes
    for j, child_node in enumerate(node.children):
        is_item_last = (n_meta + j == total_items - 1) and not node.truncated
        child_lines = _render_plain(
            child_node,
            prefix=_line_prefix(is_item_last),
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
    - 📦 Array nodes
    - 🖼️ Image groups (multiscales)
    - 🏷️ Label image groups
    - 🅾️ Other OME-zarr group nodes
    - 📁 Regular group nodes
    - ⋯  Indicates truncated children (when max_per_level is exceeded)
    """
    tree_data = _build_tree(group, depth, max_per_level)
    lines = _render_plain(tree_data)
    return "\n".join(lines)
