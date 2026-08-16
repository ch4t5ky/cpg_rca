from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath

from src.offline.log import TrieNode, WILDCARD


__all__ = ["visualize_trie_matplotlib"]

_LEVEL_HEIGHT = 3.25
_MIN_LEAF_WIDTH = 3.0
_CHILD_GAP = 0.70
_EDGE_LABEL_MARGIN = 0.24
_MAX_TOKEN_CHARS = 26
_MAX_CALL_IDS = 3


def _clip_text(value: str, limit: int = _MAX_TOKEN_CHARS) -> str:
    """Return a compact one-line edge label suitable for dense Trie diagrams."""
    value = " ".join(str(value or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _format_terminal_ids(call_ids: Iterable[str]) -> str:
    """Format a bounded list of terminal CPG CALL ids for the node annotation."""
    values = list(call_ids)
    if not values:
        return ""
    shown = values[:_MAX_CALL_IDS]
    suffix = f" +{len(values) - len(shown)}" if len(values) > len(shown) else ""
    return "CALL: " + ", ".join(shown) + suffix


def _trie_bfs(root: TrieNode):
    """Yield Trie nodes in BFS order with renderer-local stable numeric ids."""
    counter = 0

    def new_id() -> int:
        nonlocal counter
        counter += 1
        return counter

    root_id = new_id()
    queue = deque([(root, root_id, None, None)])
    while queue:
        node, node_id, parent_id, edge_key = queue.popleft()
        yield node_id, parent_id, edge_key, node
        for key, child in node.children.items():
            queue.append((child, new_id(), node_id, key))


def _tree_layout(
    edges: List[Tuple[int, int]],
    root_id: int,
    node_labels: Dict[int, str],
    edge_labels: Dict[Tuple[int, int], str],
) -> Dict[int, Tuple[float, float]]:
    """Compute a top-down layout whose subtree widths account for label length."""
    children: Dict[int, List[int]] = defaultdict(list)
    parent: Dict[int, int] = {}
    for source, target in edges:
        children[source].append(target)
        parent[target] = source

    visit_order: List[int] = []
    queue = deque([root_id])
    seen = {root_id}
    while queue:
        node_id = queue.popleft()
        visit_order.append(node_id)
        for child_id in children[node_id]:
            if child_id not in seen:
                seen.add(child_id)
                queue.append(child_id)

    def own_width(node_id: int) -> float:
        node_width = max(_MIN_LEAF_WIDTH, 0.18 * len(node_labels.get(node_id, "")) + 1.4)
        parent_id = parent.get(node_id)
        if parent_id is None:
            return node_width
        edge_width = max(
            _MIN_LEAF_WIDTH,
            0.17 * len(edge_labels.get((parent_id, node_id), "")) + 1.2,
        )
        return max(node_width, edge_width)

    widths: Dict[int, float] = {}
    for node_id in reversed(visit_order):
        child_ids = children[node_id]
        if not child_ids:
            widths[node_id] = own_width(node_id)
            continue
        children_width = sum(widths[child_id] for child_id in child_ids)
        children_width += _CHILD_GAP * max(0, len(child_ids) - 1)
        widths[node_id] = max(own_width(node_id), children_width)

    positions: Dict[int, Tuple[float, float]] = {}
    x_start: Dict[int, float] = {root_id: -widths[root_id] / 2}
    depth: Dict[int, int] = {root_id: 0}

    for node_id in visit_order:
        positions[node_id] = (
            x_start[node_id] + widths[node_id] / 2,
            -depth[node_id] * _LEVEL_HEIGHT,
        )
        cursor = x_start[node_id]
        for child_id in children[node_id]:
            depth[child_id] = depth[node_id] + 1
            x_start[child_id] = cursor
            cursor += widths[child_id] + _CHILD_GAP

    return positions


def _draw_curved_edge(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    """Draw a smooth inter-level Trie edge behind labels and nodes."""
    middle_y = (y0 + y1) / 2
    path = MplPath(
        [
            (x0, y0 - 0.40),
            (x0, middle_y + 0.25),
            (x1, middle_y - 0.25),
            (x1, y1 + 0.42),
        ],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(
        path, facecolor="none", edgecolor="#94a3b8", linewidth=1.45,
        alpha=0.80, zorder=1,
    ))


def _draw_circle(ax, x: float, y: float, label: str = "") -> None:
    """Draw an internal Trie node."""
    ax.add_patch(Circle(
        (x, y), radius=0.42, facecolor="#2563eb", edgecolor="#1e3a8a",
        linewidth=1.8, zorder=4,
    ))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                color="white", family="DejaVu Sans Mono", weight="bold", zorder=5)


def _draw_diamond(ax, x: float, y: float) -> None:
    """Draw a terminal Trie node representing one or more log templates."""
    size = 0.56
    ax.add_patch(Polygon(
        [(x, y + size), (x + size, y), (x, y - size), (x - size, y)],
        closed=True, facecolor="#059669", edgecolor="#064e3b", linewidth=1.8, zorder=4,
    ))
    ax.text(x, y, "LOG", ha="center", va="center", fontsize=7.5,
            color="white", family="DejaVu Sans Mono", weight="bold", zorder=5)


def _draw_call_box(ax, x: float, y: float, text: str) -> None:
    """Draw compact CPG CALL metadata below a terminal Trie node."""
    if not text:
        return
    width = max(1.6, min(5.8, 0.095 * len(text) + 0.45))
    height = 0.42
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=0.9, edgecolor="#86efac", facecolor="#ecfdf5", zorder=3,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.8,
            color="#065f46", family="DejaVu Sans Mono", zorder=5)


def visualize_trie_matplotlib(root: TrieNode, output_path: str = "trie.png") -> None:
    """
    Render a Log Template Trie to PNG.

    The layout dynamically allocates horizontal space from subtree and label
    widths, reducing collisions between branches, tokens, and terminal labels.
    """
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[tuple[int, int, str]] = []
    for node_id, parent_id, edge_key, node in _trie_bfs(root):
        node_rows.append({
            "id": node_id,
            "is_terminal": bool(node.terminals),
            "call_ids": [template.call_node_id for template in node.terminals],
            "label": "ROOT" if parent_id is None else "",
        })
        if parent_id is not None:
            edge_rows.append((parent_id, node_id, edge_key))

    if not node_rows:
        raise ValueError("Cannot visualize an empty Trie.")

    root_id = node_rows[0]["id"]
    node_labels = {row["id"]: row["label"] for row in node_rows}
    edge_labels = {
        (source, target): _clip_text(WILDCARD if key == WILDCARD else key)
        for source, target, key in edge_rows
    }
    positions = _tree_layout(
        [(source, target) for source, target, _ in edge_rows],
        root_id,
        node_labels,
        edge_labels,
    )

    all_x = [x for x, _ in positions.values()]
    all_y = [y for _, y in positions.values()]
    graph_width = max(all_x) - min(all_x) if all_x else 1.0
    graph_height = max(all_y) - min(all_y) if all_y else 1.0
    figure_width = min(64.0, max(18.0, graph_width * 1.05 + 6.0))
    figure_height = min(42.0, max(10.0, graph_height * 0.88 + 5.5))
    label_font_size = max(6.0, min(8.5, 10.0 - graph_width / 45.0))

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    for source, target, _ in edge_rows:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        _draw_curved_edge(ax, x0, y0, x1, y1)
        label_x = (x0 + x1) / 2 + (0.14 if x1 >= x0 else -0.14)
        label_y = y0 * 0.40 + y1 * 0.60 - _EDGE_LABEL_MARGIN
        ax.text(
            label_x, label_y, edge_labels[(source, target)],
            ha="center", va="center", fontsize=label_font_size, color="#334155", zorder=2,
            bbox={
                "boxstyle": "round,pad=0.20", "facecolor": "#ffffff",
                "edgecolor": "#cbd5e1", "linewidth": 0.7, "alpha": 0.96,
            },
        )

    for row in node_rows:
        x, y = positions[row["id"]]
        if row["is_terminal"]:
            _draw_diamond(ax, x, y)
        else:
            _draw_circle(ax, x, y, "ROOT" if row["id"] == root_id else "")
        _draw_call_box(ax, x, y - 0.95, _format_terminal_ids(row["call_ids"]))

    ax.text(0.5, 1.045, "Log Template Trie", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=18, weight="bold", color="#111827")
    ax.text(
        0.5, 1.005,
        "Blue circle: internal Trie node  |  Green diamond: terminal log template  |  <*>: wildcard",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#64748b",
    )

    padding_x = max(3.0, graph_width * 0.06)
    padding_y = max(2.5, graph_height * 0.12)
    ax.set_xlim(min(all_x) - padding_x, max(all_x) + padding_x)
    ax.set_ylim(min(all_y) - padding_y - 0.8, max(all_y) + padding_y)
    ax.axis("off")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Trie] Saved PNG -> {output}")