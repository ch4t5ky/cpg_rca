import os
import textwrap
import graphviz
from src.offline.finite_state_machine import StaticLogFSM
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
import pandas as pd
import os
from pathlib import Path
import json
from dataclasses import dataclass, field


def short_method(full_name: str) -> str:
    if not full_name:
        return ""
    return full_name.rsplit(".", 1)[-1] if "." in full_name else full_name


def shorten(s: str, width: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return textwrap.shorten(s, width=width, placeholder="...") if s else ""


def draw_branching_call_flow_graphviz(
    result,
    method_full_name: str,
    method_graph,
    filename: str = None,
    fmt: str = "png",
    max_condition_len: int = 28,
    max_paths_to_render: int = 200,
    rankdir: str = "LR",
    expand_links: bool = True,
    output_dir: str = "output",
    _rendered_registry: dict = None,
):
    """
    Строит структурный граф ветвления метода по method_paths.
    Узлы графа:
    - вызовы методов
    - RETURN
    - ромбы-ветвления по condition

    Если у вызванного метода есть собственные method_paths в result,
    для него создаётся отдельный файл.
    """
    _rendered_registry = _rendered_registry if _rendered_registry is not None else {}

    paths = result.method_paths.get(method_full_name, [])
    if not paths:
        raise ValueError(f"No paths for {method_full_name}")

    if len(paths) > max_paths_to_render:
        paths = paths[:max_paths_to_render]

    cs_map = {cs.node_id: cs for cs in getattr(method_graph, "call_sites", [])}

    def segment_calls(seg):
        calls = []
        seen = set()

        for cn in seg.nodes:
            if cn.label.upper() != "CALL":
                continue

            cs = cs_map.get(cn.node_id)
            callee_full = getattr(cs, "method_full_name", "") if cs else ""
            if not callee_full:
                continue

            display = short_method(callee_full)
            if not display or callee_full in seen:
                continue

            seen.add(callee_full)
            calls.append((callee_full, display))

        return tuple(calls)

    def extract_condition_text(prev_seg):
        if prev_seg is None or not prev_seg.nodes:
            return ""

        for cn in reversed(prev_seg.nodes):
            if cn.label.upper() == "CALL":
                code = (cn.code or "").strip()
                if code:
                    return shorten(code, max_condition_len)

        return ""

    def normalize_path(path):
        normalized = []
        prev_seg = None

        for seg in path.segments:
            calls = segment_calls(seg)
            cond = seg.condition or ""
            cond_text = extract_condition_text(prev_seg) if cond else ""
            is_terminal = bool(seg.is_terminal)

            if not calls and not cond and not is_terminal:
                prev_seg = seg
                continue

            if calls:
                for i, (callee_full, display) in enumerate(calls):
                    normalized.append({
                        "condition": cond if i == 0 else "",
                        "key": callee_full,
                        "display": display,
                        "is_terminal": False,
                        "condition_text": cond_text if i == 0 else "",
                        "callee_full": callee_full,
                    })

                if is_terminal:
                    normalized.append({
                        "condition": "",
                        "key": "__return__",
                        "display": "RETURN",
                        "is_terminal": True,
                        "condition_text": "",
                        "callee_full": None,
                    })
            else:
                normalized.append({
                    "condition": cond,
                    "key": "__return__" if is_terminal else "",
                    "display": "RETURN" if is_terminal else "",
                    "is_terminal": is_terminal,
                    "condition_text": cond_text,
                    "callee_full": None,
                })

            prev_seg = seg

        return normalized

    normalized_paths = []
    for path in paths:
        norm = normalize_path(path)
        if not norm:
            continue

        sigs = tuple(
            (
                item["condition"],
                item["is_terminal"],
                item["key"],
                item["condition_text"],
                item["display"],
                item["callee_full"],
            )
            for item in norm
        )
        normalized_paths.append(sigs)

    if not normalized_paths:
        raise ValueError(f"No normalized paths for {method_full_name}")

    class TrieNode:
        __slots__ = ("key", "children", "count", "terminal_count", "depth", "uid")
        _counter = [0]

        def __init__(self, key=None, depth=0):
            self.key = key
            self.children = {}
            self.count = 0
            self.terminal_count = 0
            self.depth = depth
            TrieNode._counter[0] += 1
            self.uid = f"n{TrieNode._counter[0]}"

    root = TrieNode(depth=0)

    for sigs in normalized_paths:
        cur = root
        cur.count += 1
        for sig in sigs:
            nxt = cur.children.get(sig)
            if nxt is None:
                nxt = TrieNode(key=sig, depth=cur.depth + 1)
                cur.children[sig] = nxt
            cur = nxt
            cur.count += 1
        cur.terminal_count += 1

    def cond_rank(sig):
        cond = sig[0]
        if cond in ("TRUE", "LOOP_TRUE"):
            return 0
        if cond in ("FALSE", "LOOP_FALSE"):
            return 1
        if cond == "LOOP_BODY":
            return 2
        return 3

    _ordered_cache = {}

    def ordered_children(node):
        cache_key = id(node)
        if cache_key in _ordered_cache:
            return _ordered_cache[cache_key]

        ordered = sorted(
            node.children.items(),
            key=lambda kv: (
                cond_rank(kv[0]),
                str(kv[0][2]),
                str(kv[0][3]),
                str(kv[0][1]),
            ),
        )
        _ordered_cache[cache_key] = ordered
        return ordered

    def box_text(sig):
        cond, is_terminal, key, cond_text, display, callee_full = sig
        if is_terminal:
            return "RETURN"
        if callee_full:
            return display
        return ""

    def should_draw_node(sig):
        cond, is_terminal, key, cond_text, display, callee_full = sig
        return bool(callee_full) or is_terminal

    def edge_label(sig):
        cond, is_terminal, key, cond_text, display, callee_full = sig
        if cond and cond_text:
            return f"{cond}\n{cond_text}"
        return cond or cond_text or None

    EDGE_COLOR = {
        "TRUE": "#22c55e",
        "LOOP_TRUE": "#22c55e",
        "FALSE": "#ef4444",
        "LOOP_FALSE": "#ef4444",
        "LOOP_BODY": "#f59e0b",
    }

    def edge_color(cond):
        return EDGE_COLOR.get(cond, "#94a3b8")

    g = graphviz.Digraph(
        "flow",
        graph_attr={
            "bgcolor": "#0f172a",
            "rankdir": rankdir,
            "splines": "spline",
            "nodesep": "0.35",
            "ranksep": "0.9",
            "fontname": "Helvetica",
            "label": f"{method_full_name}\nStructural call-flow graph",
            "labelloc": "t",
            "fontcolor": "white",
            "fontsize": "20",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "11",
            "fontcolor": "white",
            "style": "rounded,filled",
            "penwidth": "1.3",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
            "fontcolor": "white",
            "penwidth": "1.6",
        },
        format=fmt,
    )

    g.node(
        "root",
        label=method_graph.name,
        shape="box",
        style="rounded,filled",
        fillcolor="#0b3b2e",
        color="#34d399",
    )

    def merge_conditions(pending, new):
        p_label, p_cond = pending
        n_label, n_cond = new
        label = "\n".join(x for x in [p_label, n_label] if x) or None
        cond = p_cond or n_cond
        return (label, cond)

    def has_own_flow(callee_full):
        return bool(callee_full) and callee_full in result.method_paths

    def get_callee_method_graph(callee_full):
        for e in result.sequence:
            if e.method_graph.full_name == callee_full:
                return e.method_graph
        return None

    def visit(node, parent_uid, pending=(None, "")):
        children = ordered_children(node)
        if not children:
            return

        if len(children) > 1:
            diamond_uid = f"d_{node.uid}"
            p_label, p_cond = pending

            first_cond_text = next((sig[3] for sig, _ in children if sig[3]), "")
            diamond_label = p_label or first_cond_text or "?"

            g.node(
                diamond_uid,
                label=diamond_label,
                shape="diamond",
                fillcolor="#334155",
                color="#f8fafc",
                fontsize="9",
            )
            g.edge(
                parent_uid,
                diamond_uid,
                label=p_cond if p_cond else None,
                color=edge_color(p_cond) if p_cond else "#cbd5e1",
            )
            branch_from = diamond_uid
            pending = (None, "")
        else:
            branch_from = parent_uid

        for sig, child in children:
            cond, is_terminal, key, cond_text, display, callee_full = sig
            this_label = edge_label(sig)
            combined = merge_conditions(pending, (this_label, cond))

            if should_draw_node(sig):
                if is_terminal:
                    fillcolor, color = "#3f1d2e", "#f472b6"
                    shape = "box"
                else:
                    fillcolor, color = "#132a3a", "#38bdf8"
                    shape = "box"

                node_label = box_text(sig)
                child_has_flow = has_own_flow(callee_full)
                if child_has_flow:
                    node_label = f"{node_label}\n[expand]"
                    color = "#a78bfa"

                g.node(
                    child.uid,
                    label=node_label,
                    shape=shape,
                    style="rounded,filled",
                    fillcolor=fillcolor,
                    color=color,
                )

                _, cond_c = combined
                g.edge(
                    branch_from,
                    child.uid,
                    label=cond_c or None,
                    color=edge_color(cond_c),
                )

                if child_has_flow and expand_links and callee_full not in _rendered_registry:
                    _rendered_registry[callee_full] = True
                    callee_mg = get_callee_method_graph(callee_full)

                    if callee_mg is not None:
                        safe_name = "".join(
                            c if c.isalnum() else "_" for c in short_method(callee_full)
                        ) or "subgraph"

                        try:
                            draw_branching_call_flow_graphviz(
                                result=result,
                                method_full_name=callee_full,
                                method_graph=callee_mg,
                                filename=f"sub_{safe_name}",
                                fmt=fmt,
                                max_condition_len=max_condition_len,
                                max_paths_to_render=max_paths_to_render,
                                rankdir=rankdir,
                                expand_links=expand_links,
                                output_dir=output_dir,
                                _rendered_registry=_rendered_registry,
                            )
                        except ValueError:
                            pass

                visit(child, child.uid)
            else:
                visit(child, branch_from, pending=combined)

    visit(root, "root")

    os.makedirs(output_dir, exist_ok=True)
    render_name = filename or short_method(method_full_name) or "branching_graph"
    out_path = g.render(filename=render_name, directory=output_dir, cleanup=True)
    return out_path


def visualize_log_fsm(
    fsm: StaticLogFSM,
    output_path: str = "logfsm.png",
    max_label_length: int = 42,
    figsize: Tuple[float, float] = None,
) -> None:
    """Render a StaticLogFSM (segments + log transitions) to PNG."""
    import networkx as nx

    graph = nx.DiGraph()

    for state_id, state in fsm.states.items():
        method = short_method_name(state.method_full_name)
        label_text = shorten_label(state.label, max_label_length)

        if state.is_terminal:
            kind = "return" if state.kind == "RETURN_SEGMENT" else "incomplete"
        elif state.is_start:
            kind = "start"
        else:
            kind = "logpoint"

        graph.add_node(
            state_id,
            kind=kind,
            label=f"{method}\n{label_text}",
        )

    for transition in fsm.transitions:
        if transition.source_segment_id not in graph:
            graph.add_node(transition.source_segment_id, kind="unknown", label=transition.source_segment_id)
        if transition.target_segment_id not in graph:
            graph.add_node(transition.target_segment_id, kind="unknown", label=transition.target_segment_id)

        edge_label = shorten_label(transition.template, max_label_length)
        graph.add_edge(
            transition.source_segment_id,
            transition.target_segment_id,
            label=edge_label,
            partial=False,
            terminal=False,
        )

    if not graph.nodes:
        raise ValueError("FSM is empty — nothing to visualize.")

    if figsize is None:
        node_count = max(1, len(graph.nodes))
        figsize = (
            min(28.0, max(13.0, node_count * 2.2)),
            min(20.0, max(8.0, node_count * 1.35)),
        )

    positions = fsm_layout(graph, fsm)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    ax.set_title(
        f"Static Log FSM — {fsm.entrypoint_full_name}",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
        pad=22,
    )

    draw_fsm_edges(ax=ax, graph=graph, positions=positions)
    draw_fsm_nodes(ax=ax, graph=graph, positions=positions)
    draw_fsm_legend(ax)

    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    padding_x = max(1.5, (max(xs) - min(xs)) * 0.12)
    padding_y = max(1.5, (max(ys) - min(ys)) * 0.16)
    ax.set_xlim(min(xs) - padding_x, max(xs) + padding_x)
    ax.set_ylim(min(ys) - padding_y - 0.5, max(ys) + padding_y)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Static Log FSM saved: {output_path}")


def fsm_layout(graph, fsm: StaticLogFSM) -> Dict[str, Tuple[float, float]]:
    """Layered layout with START row first; falls back to dot, then spring."""
    import networkx as nx

    try:
        levels = compute_fsm_levels(graph, fsm)
        if levels:
            level_groups: Dict[int, List[str]] = {}
            for node_id, level in levels.items():
                level_groups.setdefault(level, []).append(node_id)

            positions: Dict[str, Tuple[float, float]] = {}
            for level, node_ids in sorted(level_groups.items()):
                ordered_nodes = sorted(
                    node_ids,
                    key=lambda nid: (
                        node_kind_priority(graph.nodes[nid].get("kind", "")),
                        str(graph.nodes[nid].get("label", "")),
                        nid,
                    ),
                )
                width = len(ordered_nodes)
                for index, node_id in enumerate(ordered_nodes):
                    x = (index - (width - 1) / 2)
                    y = -level * 2.8
                    positions[node_id] = (x * 4.3, y)
            return positions
    except Exception:
        pass

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        raw = graphviz_layout(graph, prog="dot")
        return {node_id: (float(x), -float(y)) for node_id, (x, y) in raw.items()}
    except Exception:
        pass

    positions = nx.spring_layout(graph, seed=42, k=2.5, iterations=100)
    return {node_id: (float(x) * 10.0, float(y) * 10.0) for node_id, (x, y) in positions.items()}


def compute_fsm_levels(graph, fsm: StaticLogFSM) -> Dict[str, int]:
    """BFS distance from every start segment; unreachable nodes go last."""
    import networkx as nx

    start_ids = [sid for sid, state in fsm.states.items() if state.is_start and sid in graph]

    levels: Dict[str, int] = {}
    for start_id in start_ids:
        for node_id, distance in nx.single_source_shortest_path_length(graph, start_id).items():
            if node_id not in levels or distance < levels[node_id]:
                levels[node_id] = distance

    max_level = max(levels.values(), default=0)
    unreachable = sorted(set(graph.nodes) - set(levels))
    for offset, node_id in enumerate(unreachable, start=1):
        levels[node_id] = max_level + offset

    return levels


def node_kind_priority(kind: str) -> int:
    priority = {"start": 0, "logpoint": 1, "return": 2, "incomplete": 3, "unknown": 4}
    return priority.get(kind, 99)


def draw_fsm_edges(ax, graph, positions: Dict[str, Tuple[float, float]]) -> None:
    import networkx as nx

    edges = [(source, target) for source, target, _ in graph.edges(data=True)]

    nx.draw_networkx_edges(
        graph, positions, ax=ax, edgelist=edges,
        arrows=True, arrowstyle="-|>", arrowsize=18, width=1.8,
        edge_color="#64748b", connectionstyle="arc3,rad=0.05",
        min_source_margin=26, min_target_margin=26,
    )

    edge_labels = {
        (source, target): data["label"]
        for source, target, data in graph.edges(data=True)
        if data.get("label")
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(
            graph, positions, edge_labels=edge_labels, ax=ax,
            font_size=8, font_color="#334155", rotate=False, label_pos=0.5,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.95),
        )


def draw_fsm_nodes(ax, graph, positions: Dict[str, Tuple[float, float]]) -> None:
    from matplotlib.patches import FancyBboxPatch

    colors = {
        "start":    {"face": "#059669", "edge": "#064e3b", "text": "#ffffff"},
        "logpoint": {"face": "#2563eb", "edge": "#1e3a8a", "text": "#ffffff"},
        "return":   {"face": "#166534", "edge": "#14532d", "text": "#ffffff"},
        "incomplete": {"face": "#f59e0b", "edge": "#92400e", "text": "#1f2937"},
        "unknown":  {"face": "#94a3b8", "edge": "#475569", "text": "#0f172a"},
    }

    for node_id, attributes in graph.nodes(data=True):
        x, y = positions[node_id]
        kind = attributes.get("kind", "unknown")
        label = attributes.get("label", node_id)
        palette = colors.get(kind, colors["unknown"])

        lines = label.splitlines()
        widest_line = max((len(line) for line in lines), default=8)
        width = max(2.0, min(5.8, 0.085 * widest_line + 0.75))
        height = max(0.85, min(1.85, 0.42 * len(lines) + 0.38))
        if kind in {"start", "return"}:
            width = max(width, 1.8)
            height = max(height, 0.78)

        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.10,rounding_size=0.16",
            linewidth=1.8, edgecolor=palette["edge"], facecolor=palette["face"], zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            x, y, label, ha="center", va="center", fontsize=9,
            fontweight="bold" if kind != "logpoint" else "normal",
            color=palette["text"], family="DejaVu Sans", zorder=4, wrap=True,
        )


def draw_fsm_legend(ax) -> None:
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#059669", edgecolor="#064e3b", label="START"),
        Patch(facecolor="#2563eb", edgecolor="#1e3a8a", label="Segment (between logs)"),
        Patch(facecolor="#166534", edgecolor="#14532d", label="RETURN"),
        Patch(facecolor="#f59e0b", edgecolor="#92400e", label="Incomplete path"),
    ]
    ax.legend(
        handles=handles, loc="upper right", frameon=True, framealpha=0.96,
        facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8,
    )


def short_method_name(full_name: str) -> str:
    if not full_name:
        return "unknown"
    return full_name.rsplit(".", 1)[-1]


def shorten_label(text: str, max_length: int) -> str:
    import textwrap

    normalized = " ".join((text or "").split())
    if len(normalized) > max_length:
        normalized = normalized[: max_length - 1].rstrip() + "…"
    return "\n".join(textwrap.wrap(normalized, width=24, break_long_words=False, break_on_hyphens=False))


def _short_method_name(fullname: str) -> str:
    if not fullname:
        return "unknown"

    return fullname.rsplit(".", 1)[-1]


def _shorten_label(text: str, max_length: int) -> str:
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


def draw_chain_timeline(results: pd.DataFrame, outpath: str = "output/fsm_chain_timeline.png") -> str:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    rows = results.copy()
    rows = rows.sort_values(["chain_id", "timestamp"]).reset_index(drop=True)

    chain_groups = defaultdict(list)
    for _, r in rows.iterrows():
        chain_groups[int(r["chain_id"])].append(r)

    fig_h = max(5, 0.45 * len(chain_groups) + 2)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    colors = {
        "continue": "#22c55e",
        "new_chain": "#ef4444",
        "unknown": "#94a3b8",
    }

    y_positions = {}
    for i, chain_id in enumerate(sorted(chain_groups.keys())):
        y_positions[chain_id] = i

    for chain_id, group in chain_groups.items():
        group = sorted(group, key=lambda r: r["timestamp"] if pd.notna(r["timestamp"]) else -1)
        ts = [r["timestamp"] for r in group if pd.notna(r["timestamp"])]
        if not ts:
            continue

        y = y_positions[chain_id]
        start_ts = min(ts)
        end_ts = max(ts)

        ax.hlines(y, start_ts, end_ts, color="#64748b", linewidth=2.2, alpha=0.8)

        for r in group:
            t = r["timestamp"]
            if pd.isna(t):
                continue
            ax.scatter(
                t,
                y,
                s=42,
                color=colors.get(str(r["verdict"]), "#94a3b8"),
                zorder=3,
            )

            if str(r["verdict"]) in ("continue", "new_chain", "unknown"):
                label = str(r["matched_state"]) if r.get("matched_state") else ""
                if label and r["verdict"] != "continue":
                    ax.text(
                        t,
                        y + 0.12,
                        label[:28],
                        fontsize=7,
                        color="#cbd5e1",
                        rotation=15,
                        ha="left",
                        va="bottom",
                    )

    ax.set_title("FSM chains timeline", color="white", fontsize=15, pad=14)
    ax.set_xlabel("timestamp", color="white")
    ax.set_ylabel("chain id", color="white")
    ax.tick_params(colors="white")

    yticks = [y_positions[cid] for cid in sorted(chain_groups.keys())]
    ylabels = [str(cid) for cid in sorted(chain_groups.keys())]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, color="white")

    legend = [
        Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=8)
        for k, v in colors.items()
    ]
    ax.legend(handles=legend, facecolor="#0f172a", edgecolor="#334155", labelcolor="white")

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath

def save_chain_store(active_chains: Dict[int, Any], outpath: str) -> str:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    payload = {}
    for cid, chain in active_chains.items():
        payload[str(cid)] = {
            "chain_id": chain.chain_id,
            "fsm_key": chain.fsm_key,
            "current_state": chain.current_state,
            "last_ts": chain.last_ts,
            "score": chain.score,
            "terminated": chain.terminated,
            "logs": chain.logs,
        }
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return outpath


def export_results(results: pd.DataFrame, active_chains: Dict[int, Any], output_dir: str, service_name: str) -> Dict[str, str]:
    out_dir = Path(output_dir) / service_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fsm_chain_timeline.csv"
    png_path = out_dir / "fsm_chain_timeline.png"
    json_path = out_dir / "active_chains.json"

    results.to_csv(csv_path, index=False)
    draw_chain_timeline(results, outpath=str(png_path))
    save_chain_store(active_chains, outpath=str(json_path))

    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "json": str(json_path),
    }

from src.offline.flow import SemanticUnit, SemanticUnitKind


@dataclass
class _TrieNode:
    key: Tuple[str, str] | None = None
    unit: SemanticUnit | None = None
    children: Dict[Tuple[str, str], "_TrieNode"] = field(default_factory=dict)
    terminal: bool = False
    uid: str = ""


def draw_semantic_flow_graphviz(
    result,
    method_full_name: str,
    method_graph,
    filename: str | None = None,
    fmt: str = "png",
    output_dir: str = "output",
    max_paths_to_render: int = 200,
    max_code_length: int = 52,
) -> str:
    """Render a method as a graph of semantic source-level operations.

    Required input: MethodPath.semantic_segments, populated by EntrypointFlow.
    Nodes are declarations, assignments, calls, conditions and returns.
    CFG branch labels are placed on graph edges.  Each node displays DEF/USE.
    """
    paths = result.method_paths.get(method_full_name, [])[:max_paths_to_render]
    if not paths:
        raise ValueError(f"No paths for {method_full_name}")
    if not any(path.semantic_segments for path in paths):
        raise ValueError(
            "Semantic paths are absent. Rebuild flow with the new flow.py first."
        )

    counter = 0

    def new_node(key=None, unit=None) -> _TrieNode:
        nonlocal counter
        counter += 1
        return _TrieNode(key=key, unit=unit, uid=f"semantic_{counter}")

    root = new_node()
    for path in paths:
        cursor = root
        for segment in path.semantic_segments:
            condition = segment.condition or ""
            units = segment.units
            if not units and segment.is_terminal:
                units = [SemanticUnit(
                    node_id="__return__",
                    kind=SemanticUnitKind.RETURN,
                    code="RETURN",
                    line="",
                )]
            for index, unit in enumerate(units):
                edge_condition = condition if index == 0 else ""
                key = (edge_condition, unit.node_id)
                child = cursor.children.get(key)
                if child is None:
                    child = new_node(key=key, unit=unit)
                    cursor.children[key] = child
                cursor = child
        if path.is_complete:
            cursor.terminal = True

    palette = {
        SemanticUnitKind.DECLARATION: ("#0f766e", "#5eead4"),
        SemanticUnitKind.ASSIGNMENT: ("#1d4ed8", "#93c5fd"),
        SemanticUnitKind.CALL: ("#7c3aed", "#c4b5fd"),
        SemanticUnitKind.LOG_CALL: ("#c2410c", "#fdba74"),
        SemanticUnitKind.CONDITION: ("#a16207", "#fde047"),
        SemanticUnitKind.RETURN: ("#166534", "#86efac"),
        SemanticUnitKind.JUMP: ("#475569", "#cbd5e1"),
        SemanticUnitKind.UNKNOWN: ("#334155", "#cbd5e1"),
    }
    edge_colors = {
        "TRUE": "#22c55e",
        "LOOP_TRUE": "#22c55e",
        "FALSE": "#ef4444",
        "LOOP_FALSE": "#ef4444",
        "LOOP_BODY": "#f59e0b",
    }

    graph = graphviz.Digraph(
        "semantic_flow",
        graph_attr={
            "rankdir": "LR",
            "bgcolor": "#0f172a",
            "splines": "spline",
            "nodesep": "0.35",
            "ranksep": "0.75",
            "fontname": "Helvetica",
            "label": f"{method_full_name}\\nSemantic flow: operations and data dependencies",
            "labelloc": "t",
            "fontsize": "19",
            "fontcolor": "white",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Helvetica",
            "fontsize": "10",
            "fontcolor": "white",
            "margin": "0.14,0.10",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
            "fontcolor": "white",
            "penwidth": "1.5",
        },
        format=fmt,
    )
    graph.node(
        "start",
        label=f"START\\n{method_graph.name}",
        fillcolor="#065f46",
        color="#6ee7b7",
        penwidth="2",
    )

    def unit_label(unit: SemanticUnit) -> str:
        code = " ".join((unit.code or "").split())
        code = textwrap.shorten(code, width=max_code_length, placeholder=" …")
        title = unit.kind.value.upper()
        line = f"L{unit.line}" if unit.line else ""
        parts = [f"{title} {line}", code]
        if unit.defines:
            parts.append("DEF: " + ", ".join(value.name for value in unit.defines))
        if unit.uses:
            parts.append("USE: " + ", ".join(value.name for value in unit.uses))
        return "\\n".join(part for part in parts if part)

    rendered: set[str] = set()

    def visit(parent_id: str, trie_node: _TrieNode) -> None:
        for (condition, _), child in trie_node.children.items():
            unit = child.unit
            if unit is None:
                continue
            fill, border = palette[unit.kind]
            if child.uid not in rendered:
                graph.node(child.uid, label=unit_label(unit), fillcolor=fill, color=border)
                rendered.add(child.uid)
            graph.edge(
                parent_id,
                child.uid,
                label=condition or None,
                color=edge_colors.get(condition, "#94a3b8"),
            )
            visit(child.uid, child)

    visit("start", root)
    os.makedirs(output_dir, exist_ok=True)
    name = filename or (method_graph.name or "semantic_flow")
    return graph.render(filename=name, directory=output_dir, cleanup=True)

from src.offline.flow import SemanticFlowGraph, SemanticUnit, SemanticUnitKind


def draw_semantic_graph_graphviz(
    semantic_graph: SemanticFlowGraph,
    filename: str | None = None,
    output_dir: str = "output",
    fmt: str = "png",
    rankdir: str = "LR",
    max_code_length: int = 58,
) -> str:
    """Render a semantic control-flow graph for one method.

    Visual semantics:
      START                  green ellipse
      CONDITION              green diamond
      LOOP                   green hexagon
      CALL INTERNAL          blue ellipse
      RETURN                 red ellipse
      all other operations   purple ellipse
    """
    graph = graphviz.Digraph(
        "semantic_flow",
        graph_attr={
            "rankdir": rankdir,
            "splines": "spline",
            "nodesep": "0.35",
            "ranksep": "0.80",
            "fontname": "Helvetica",
            "label": (
                f"{semantic_graph.method_full_name}\n"
                "Semantic control-flow graph"
            ),
            "labelloc": "t",
            "bgcolor": "#ffffff",
            "fontcolor": "#0f172a",
            "fontsize": "19",
        },
        node_attr={
            "style": "filled",
            "fontname": "Helvetica",
            "fontsize": "10",
            "fontcolor": "#0f172a",
            "margin": "0.16,0.10",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
            "fontcolor": "#334155",
            "penwidth": "1.5",
            "color": "#94a3b8",
        },
        format=fmt,
    )

    def format_parameters(
        parameters: list[tuple[str, str]],
    ) -> str:
        if not parameters:
            return "—"

        return "\\n".join(
            f"{name}: {type_name}" if name else type_name
            for name, type_name in parameters
        )


    def start_label() -> str:
        inputs = format_parameters(semantic_graph.input_parameters)
        return f"START\\nIN:\\n{inputs}"


    def return_label() -> str:
        outputs = format_parameters(semantic_graph.output_parameters)
        return f"RETURN\\nOUT:\\n{outputs}"


    graph.node(
        semantic_graph.start_node_id,
        label=start_label(),
        shape="ellipse",
        fillcolor="#059669",
        color="#6ee7b7",
        fontcolor="#ffffff",
        penwidth="2",
    )

    def short_method(full_name: str) -> str:
        return (
            full_name.rsplit(".", 1)[-1]
            if "." in full_name
            else full_name
        )

    def compact_code(code: str) -> str:
        normalized = " ".join((code or "").split())
        return textwrap.shorten(
            normalized,
            width=max_code_length,
            placeholder=" …",
        )


    def unit_style(unit: SemanticUnit) -> tuple[str, str, str]:
        if unit.kind is SemanticUnitKind.CONDITION:
            return "diamond", "#DCFCE7", "#16A34A"

        if unit.kind is SemanticUnitKind.LOOP:
            return "hexagon", "#D1FAE5", "#15803D"

        if unit.kind is SemanticUnitKind.RETURN:
            return "ellipse", "#FEE2E2", "#DC2626"

        if unit.internal_callee_full_names:
            return "ellipse", "#DBEAFE", "#2563EB"

        return "ellipse", "#EDE9FE", "#7C3AED"

    def unit_label(unit: SemanticUnit) -> str:
        if unit.kind is SemanticUnitKind.RETURN:
            return return_label()

        if unit.internal_callee_full_names:
            callees = ", ".join(
                short_method(name)
                for name in unit.internal_callee_full_names
            )
            return f"CALL INTERNAL\n{callees}"

        title = unit.kind.value.upper()
        code = compact_code(unit.code)

        return f"{title}\n{code}" if code else title

    for unit_id, unit in semantic_graph.nodes.items():
        shape, fillcolor, border = unit_style(unit)

        node_kwargs = {
            "label": unit_label(unit),
            "shape": shape,
            "fillcolor": fillcolor,
            "color": border,
        }

        if unit.kind is SemanticUnitKind.RETURN:
            node_kwargs.update(
                {
                    "fontcolor": "#991B1B",
                    "penwidth": "2.0",
                }
            )

        graph.node(unit_id, **node_kwargs)

    return_ids = [
        unit_id
        for unit_id, unit in semantic_graph.nodes.items()
        if unit.kind is SemanticUnitKind.RETURN
    ]

    if return_ids:
        with graph.subgraph() as terminal_rank:
            terminal_rank.attr(rank="sink")

            for return_id in return_ids:
                terminal_rank.node(return_id)

    edge_colors = {
        "TRUE": "#22c55e",
        "FALSE": "#ef4444",
        "body": "#22c55e",
        "next iteration": "#f59e0b",
    }

    for edge in semantic_graph.edge_list:
        graph.edge(
            edge.source_id,
            edge.target_id,
            label=edge.condition or None,
            color=edge_colors.get(edge.condition, "#94a3b8"),
        )

    os.makedirs(output_dir, exist_ok=True)

    name = filename or (
        f"semantic_{short_method(semantic_graph.method_full_name)}"
    )

    return graph.render(
        filename=name,
        directory=output_dir,
        cleanup=True,
    )

from src.offline.finite_state_machine import StaticLogFSM

_KIND_STYLE = {
    "START_SEGMENT": {"fillcolor": "#059669", "fontcolor": "white"},       # green
    "BETWEEN_LOGS": {"fillcolor": "#2563eb", "fontcolor": "white"},        # blue
    "RETURN_SEGMENT": {"fillcolor": "#7c3aed", "fontcolor": "white"},      # purple
    "INCOMPLETE_SEGMENT": {"fillcolor": "#dc2626", "fontcolor": "white"},  # red
}

_MAX_LABEL_METHODS = 4
_MAX_LABEL_CHARS = 60


def _safe_id(raw_id: str) -> str:
    """Graphviz-safe node id — no ':' (which .edge() misparses as a port)."""
    return raw_id.replace(":", "_").replace(" ", "_")


def _truncate(text: str, limit: int = _MAX_LABEL_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _state_label(state) -> str:
    lines = [state.id, state.kind]

    methods = list(state.direct_methods) + list(state.external_calls)
    if methods:
        shown = [m.rsplit(".", 1)[-1] for m in methods[:_MAX_LABEL_METHODS]]
        suffix = ", ..." if len(methods) > _MAX_LABEL_METHODS else ""
        lines.append(_truncate(", ".join(shown) + suffix))

    if state.conditions:
        lines.append(_truncate("if: " + " | ".join(state.conditions), 50))

    return "\n".join(lines)


def _edge_label(transition) -> str:
    template = _truncate(transition.template, 45)
    if transition.static_score:
        return f"{template}\n(score={transition.static_score})"
    return template


def draw_log_fsm_graphviz(
    fsm: StaticLogFSM,
    filename: str = "log_fsm",
    output_dir: str = "output",
    fmt: str = "png",
    rankdir: str = "LR",
    show_warnings: bool = True,
) -> str:
    """Render a StaticLogFSM to disk with graphviz and return the file path."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_label = fsm.summary()
    if show_warnings and fsm.warnings:
        graph_label += "\\n" + "\\n".join(f"WARNING: {w}" for w in fsm.warnings)

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
        peripheries = "2" if (state_id in start_ids or state_id in terminal_ids) else "1"
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
            label=_edge_label(transition),
        )

    rendered_path: str = dot.render(filename=filename, directory=str(out_dir), cleanup=True)
    return rendered_path