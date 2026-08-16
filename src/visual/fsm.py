from pathlib import Path
from typing import Dict, List, Tuple

import graphviz
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, Patch

from src.offline.finite_state_machine import StaticLogFSM


__all__ = ["visualize_log_fsm", "draw_log_fsm_graphviz"]


_KIND_STYLE = {
    "START_SEGMENT": {"fillcolor": "#059669", "fontcolor": "white"},
    "BETWEEN_LOGS": {"fillcolor": "#2563eb", "fontcolor": "white"},
    "RETURN_SEGMENT": {"fillcolor": "#7c3aed", "fontcolor": "white"},
    "INCOMPLETE_SEGMENT": {"fillcolor": "#dc2626", "fontcolor": "white"},
}
_MAX_LABEL_METHODS = 4
_MAX_LABEL_CHARS = 60


def short_method_name(full_name: str) -> str:
    """Return the final component of a fully qualified method name."""
    return full_name.rsplit(".", 1)[-1] if full_name else "unknown"


def shorten_label(text: str, max_length: int) -> str:
    """Normalize, truncate, and wrap a graph label for Matplotlib output."""
    import textwrap

    normalized = " ".join((text or "").split())
    if len(normalized) > max_length:
        normalized = normalized[: max_length - 1].rstrip() + "…"
    return "\n".join(
        textwrap.wrap(
            normalized,
            width=24,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _safe_id(raw_id: str) -> str:
    """Create a Graphviz node id without characters interpreted as ports."""
    return raw_id.replace(":", "_").replace(" ", "_")


def _truncate(text: str, limit: int = _MAX_LABEL_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _state_label(state) -> str:
    lines = [state.id, state.kind]
    methods = list(state.direct_methods) + list(state.external_calls)
    if methods:
        shown = [method.rsplit(".", 1)[-1] for method in methods[:_MAX_LABEL_METHODS]]
        suffix = ", ..." if len(methods) > _MAX_LABEL_METHODS else ""
        lines.append(_truncate(", ".join(shown) + suffix))
    if state.conditions:
        lines.append(_truncate("if: " + " | ".join(state.conditions), 50))
    return "\n".join(lines)


def _transition_label(transition) -> str:
    template = _truncate(transition.template, 45)
    return f"{template}\n(score={transition.static_score})" if transition.static_score else template


def draw_log_fsm_graphviz(
    fsm: StaticLogFSM,
    filename: str = "log_fsm",
    output_dir: str = "output",
    fmt: str = "png",
    rankdir: str = "LR",
    show_warnings: bool = True,
) -> str:
    """Render StaticLogFSM with Graphviz and return the generated file path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_label = fsm.summary()
    if show_warnings and fsm.warnings:
        graph_label += "\\n" + "\\n".join(f"WARNING: {warning}" for warning in fsm.warnings)

    dot = graphviz.Digraph(
        name=filename,
        format=fmt,
        graph_attr={
            "rankdir": rankdir,
            "label": graph_label,
            "labelloc": "t",
            "fontsize": "11",
            "fontname": "Helvetica",
            "bgcolor": "white",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Helvetica",
            "fontsize": "10",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
            "color": "#6b7280",
        },
    )

    start_ids = fsm.start_states
    terminal_ids = fsm.terminals
    for state_id, state in fsm.states.items():
        style = dict(_KIND_STYLE.get(state.kind, {"fillcolor": "#94a3b8", "fontcolor": "black"}))
        peripheries = "2" if state_id in start_ids or state_id in terminal_ids else "1"
        dot.node(
            _safe_id(state_id),
            label=_state_label(state),
            fillcolor=style["fillcolor"],
            fontcolor=style["fontcolor"],
            peripheries=peripheries,
        )

    for transition in fsm.transitions:
        dot.edge(
            _safe_id(transition.source_segment_id),
            _safe_id(transition.target_segment_id),
            label=_transition_label(transition),
        )

    return dot.render(filename=filename, directory=str(out_dir), cleanup=True)


def _node_kind_priority(kind: str) -> int:
    priority = {"start": 0, "logpoint": 1, "return": 2, "incomplete": 3, "unknown": 4}
    return priority.get(kind, 99)


def _compute_levels(graph: nx.DiGraph, fsm: StaticLogFSM) -> Dict[str, int]:
    """Assign BFS layers from all start segments; put unreachable nodes last."""
    start_ids = [state_id for state_id, state in fsm.states.items() if state.is_start and state_id in graph]
    levels: Dict[str, int] = {}
    for start_id in start_ids:
        for node_id, distance in nx.single_source_shortest_path_length(graph, start_id).items():
            if node_id not in levels or distance < levels[node_id]:
                levels[node_id] = distance

    max_level = max(levels.values(), default=0)
    for offset, node_id in enumerate(sorted(set(graph.nodes) - set(levels)), start=1):
        levels[node_id] = max_level + offset
    return levels


def _layout(graph: nx.DiGraph, fsm: StaticLogFSM) -> Dict[str, Tuple[float, float]]:
    """Use layered layout first, then Graphviz dot, then deterministic spring layout."""
    try:
        levels = _compute_levels(graph, fsm)
        groups: Dict[int, List[str]] = {}
        for node_id, level in levels.items():
            groups.setdefault(level, []).append(node_id)

        positions: Dict[str, Tuple[float, float]] = {}
        for level, node_ids in sorted(groups.items()):
            ordered = sorted(
                node_ids,
                key=lambda node_id: (
                    _node_kind_priority(graph.nodes[node_id].get("kind", "")),
                    str(graph.nodes[node_id].get("label", "")),
                    node_id,
                ),
            )
            width = len(ordered)
            for index, node_id in enumerate(ordered):
                positions[node_id] = ((index - (width - 1) / 2) * 4.3, -level * 2.8)
        return positions
    except Exception:
        pass

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        raw = graphviz_layout(graph, prog="dot")
        return {node_id: (float(x), -float(y)) for node_id, (x, y) in raw.items()}
    except Exception:
        raw = nx.spring_layout(graph, seed=42, k=2.5, iterations=100)
        return {node_id: (float(x) * 10.0, float(y) * 10.0) for node_id, (x, y) in raw.items()}


def _draw_edges(ax, graph: nx.DiGraph, positions: Dict[str, Tuple[float, float]]) -> None:
    edges = [(source, target) for source, target, _ in graph.edges(data=True)]
    nx.draw_networkx_edges(
        graph, positions, ax=ax, edgelist=edges,
        arrows=True, arrowstyle="-|>", arrowsize=18, width=1.8,
        edge_color="#64748b", connectionstyle="arc3,rad=0.05",
        min_source_margin=26, min_target_margin=26,
    )
    labels = {
        (source, target): data["label"]
        for source, target, data in graph.edges(data=True)
        if data.get("label")
    }
    if labels:
        nx.draw_networkx_edge_labels(
            graph, positions, edge_labels=labels, ax=ax,
            font_size=8, font_color="#334155", rotate=False, label_pos=0.5,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.95),
        )


def _draw_nodes(ax, graph: nx.DiGraph, positions: Dict[str, Tuple[float, float]]) -> None:
    colors = {
        "start": {"face": "#059669", "edge": "#064e3b", "text": "#ffffff"},
        "logpoint": {"face": "#2563eb", "edge": "#1e3a8a", "text": "#ffffff"},
        "return": {"face": "#166534", "edge": "#14532d", "text": "#ffffff"},
        "incomplete": {"face": "#f59e0b", "edge": "#92400e", "text": "#1f2937"},
        "unknown": {"face": "#94a3b8", "edge": "#475569", "text": "#0f172a"},
    }
    for node_id, attributes in graph.nodes(data=True):
        x, y = positions[node_id]
        kind = attributes.get("kind", "unknown")
        label = attributes.get("label", node_id)
        palette = colors.get(kind, colors["unknown"])
        lines = label.splitlines()
        widest = max((len(line) for line in lines), default=8)
        width = max(2.0, min(5.8, 0.085 * widest + 0.75))
        height = max(0.85, min(1.85, 0.42 * len(lines) + 0.38))
        if kind in {"start", "return"}:
            width, height = max(width, 1.8), max(height, 0.78)
        ax.add_patch(FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.10,rounding_size=0.16",
            linewidth=1.8, edgecolor=palette["edge"], facecolor=palette["face"], zorder=3,
        ))
        ax.text(
            x, y, label, ha="center", va="center", fontsize=9,
            fontweight="bold" if kind != "logpoint" else "normal",
            color=palette["text"], family="DejaVu Sans", zorder=4, wrap=True,
        )


def _draw_legend(ax) -> None:
    handles = [
        Patch(facecolor="#059669", edgecolor="#064e3b", label="START"),
        Patch(facecolor="#2563eb", edgecolor="#1e3a8a", label="Segment (between logs)"),
        Patch(facecolor="#166534", edgecolor="#14532d", label="RETURN"),
        Patch(facecolor="#f59e0b", edgecolor="#92400e", label="Incomplete path"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.96,
              facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8)


def visualize_log_fsm(
    fsm: StaticLogFSM,
    output_path: str = "logfsm.png",
    max_label_length: int = 42,
    figsize: Tuple[float, float] | None = None,
) -> None:
    """Render StaticLogFSM with Matplotlib as an alternative to Graphviz."""
    graph = nx.DiGraph()
    for state_id, state in fsm.states.items():
        method = short_method_name(state.method_full_name)
        label = shorten_label(state.label, max_label_length)
        if state.is_terminal:
            kind = "return" if state.kind == "RETURN_SEGMENT" else "incomplete"
        elif state.is_start:
            kind = "start"
        else:
            kind = "logpoint"
        graph.add_node(state_id, kind=kind, label=f"{method}\n{label}")

    for transition in fsm.transitions:
        for state_id in (transition.source_segment_id, transition.target_segment_id):
            if state_id not in graph:
                graph.add_node(state_id, kind="unknown", label=state_id)
        graph.add_edge(
            transition.source_segment_id,
            transition.target_segment_id,
            label=shorten_label(transition.template, max_label_length),
        )

    if not graph.nodes:
        raise ValueError("FSM is empty — nothing to visualize.")

    if figsize is None:
        node_count = max(1, len(graph.nodes))
        figsize = (min(28.0, max(13.0, node_count * 2.2)), min(20.0, max(8.0, node_count * 1.35)))

    positions = _layout(graph, fsm)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    ax.set_title(f"Static Log FSM — {fsm.entrypoint_full_name}", fontsize=16,
                 fontweight="bold", color="#0f172a", pad=22)

    _draw_edges(ax, graph, positions)
    _draw_nodes(ax, graph, positions)
    _draw_legend(ax)

    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    padding_x = max(1.5, (max(xs) - min(xs)) * 0.12)
    padding_y = max(1.5, (max(ys) - min(ys)) * 0.16)
    ax.set_xlim(min(xs) - padding_x, max(xs) + padding_x)
    ax.set_ylim(min(ys) - padding_y - 0.5, max(ys) + padding_y)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)