# show_flowchart.py  (v15 — line-number based matching)
"""
Ключевое изменение v15
──────────────────────
В реальном CPG node_id в cfg_order, cfg_nodes и call_sites
могут быть разными объектами (Joern генерирует их независимо).
Попытка сопоставить по node_id ненадёжна.

Решение: привязка child → diamond строится по LINE NUMBER:
  - каждый CONTROL_STRUCTURE имеет line (из CfgNode.line)
  - каждый CallSite имеет line (cs.line)
  - child_entry попадает в diamond[i] если:
      ctrl[i].line < child_call_line < ctrl[i+1].line
    (т.е. call-site встречается после открытия ветки и до следующей)

Это работает без cfg_order вообще — только номера строк.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

BG    = "#0d1117"
TXT   = "#e6edf3"
C_YES = "#4ade80"
C_NO  = "#f87171"
C_GRY = "#8b949e"

BOX_W  = 2.8
BOX_H  = 0.52
DIA_W  = 3.2
DIA_H  = 0.65
V_STEP = 1.1
H_OFF  = 4.2


def _wrap(t: str, w: int = 26) -> str:
    lines = textwrap.wrap(str(t), w)
    return "\n".join(lines) if lines else str(t)


def _to_int(v, default: int = 0) -> int:
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return default


def _box(ax, cx, cy, label, fill, border, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
        boxstyle="round,pad=0,rounding_size=0.1",
        facecolor=fill, edgecolor=border, linewidth=1.6, zorder=3))
    ax.text(cx, cy, _wrap(label), ha="center", va="center",
            fontsize=fs, color=TXT, zorder=4,
            fontweight="bold" if bold else "normal",
            multialignment="center")


def _diamond(ax, cx, cy, label):
    hw, hh = DIA_W/2, DIA_H/2
    ax.add_patch(plt.Polygon(
        [[cx, cy+hh], [cx+hw, cy], [cx, cy-hh], [cx-hw, cy]],
        closed=True, facecolor="#0a2e2a", edgecolor="#2dd4bf",
        linewidth=1.6, zorder=3))
    ax.text(cx, cy, _wrap(label, 24), ha="center", va="center",
            fontsize=6.5, color="#80ffef", zorder=4, multialignment="center")


def _arrow_down(ax, x, y_from, y_to, color=C_GRY):
    if y_from <= y_to:
        return
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, mutation_scale=10), zorder=2)


def _arrow_horiz_down(ax, x_from, x_to, y_h, y_to, color=C_NO):
    ax.plot([x_from, x_to], [y_h, y_h], color=color, lw=1.3, zorder=2)
    ax.annotate("", xy=(x_to, y_to), xytext=(x_to, y_h),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, mutation_scale=10), zorder=2)


def _lbl(ax, x, y, text, color, ha="left"):
    ax.text(x, y + 0.07, text, fontsize=6.5, color=color, zorder=5, ha=ha)


def _classify(code: str) -> str:
    return "loop" if re.match(r'^(for|while)\b', code.strip().lower()) else "branch"


@dataclass
class FNode:
    uid:   str
    kind:  str
    label: str
    yes:   Optional["FNode"] = None
    no:    Optional["FNode"] = None
    nxt:   Optional["FNode"] = None
    cx: float = 0.0
    cy: float = 0.0


# ── build tree ────────────────────────────────────────────────────────────────

def _build_tree(result, seq) -> Optional[FNode]:
    if not seq:
        return None

    def range_children(e):
        kids = []
        for e2 in seq:
            if e2.call_index <= e.call_index:
                continue
            if e2.depth < e.depth:
                break
            if e2.depth == e.depth:
                break
            if e2.depth == e.depth + 1:
                kids.append(e2)
        return kids

    def child_call_line(parent_entry, child_entry) -> int:
        """
        Номер строки CALL-сайта вызова child внутри parent.
        Ищем по method_full_name с учётом occurrence (повторные вызовы).
        Fallback: line_start самого child метода.
        """
        mg  = parent_entry.method_graph
        fn  = child_entry.method_graph.full_name
        occ = sum(
            1 for e2 in seq
            if e2.call_index < child_entry.call_index
            and e2.depth == child_entry.depth
            and e2.method_graph.full_name == fn
        )
        sites = sorted(
            [cs for cs in mg.call_sites if cs.method_full_name == fn],
            key=lambda cs: _to_int(cs.line),
        )
        if occ < len(sites):
            return _to_int(sites[occ].line)
        # fallback: берём line_start вызываемого метода
        return _to_int(child_entry.method_graph.line_start)

    def build(e) -> FNode:
        mg       = e.method_graph
        children = range_children(e)

        root_node = FNode(
            uid   = f"n_{e.call_index}",
            kind  = "root" if e.depth == 0 else "call",
            label = ("▶ " if e.depth == 0 else f"[{e.call_index}] ") + mg.name,
        )

        if not children:
            return root_node

        # CONTROL_STRUCTURE-ноды, отсортированные по номеру строки
        ctrl_nodes = sorted(
            [cn for cn in mg.cfg_nodes
             if cn.label.upper() == "CONTROL_STRUCTURE" and cn.code.strip()],
            key=lambda cn: _to_int(cn.line),
        )

        if not ctrl_nodes:
            # Нет веток — все дети идут линейно
            top: List[FNode] = [root_node]
            for ch in children:
                top.append(build(ch))
            for i in range(len(top) - 1):
                top[i].nxt = top[i + 1]
            return root_node

        # Номера строк ctrl-нод + sentinel
        ctrl_lines_sorted = [_to_int(cn.line) for cn in ctrl_nodes]
        ctrl_bounds = ctrl_lines_sorted + [999999]

        # ── Привязка child → diamond по номеру строки ─────────────────────────
        # child попадает в diamond[i] если:
        #   ctrl_lines[i] < child_line < ctrl_lines[i+1]
        ctrl_to_kids: Dict[int, list] = {}   # индекс ctrl → список детей
        pre_kids: list = []                  # дети до первого ctrl

        for ch in sorted(children, key=lambda c: child_call_line(e, c)):
            cl = child_call_line(e, ch)
            assigned = False
            for i in range(len(ctrl_nodes)):
                lo = ctrl_bounds[i]
                hi = ctrl_bounds[i + 1]
                if lo < cl <= hi:
                    ctrl_to_kids.setdefault(i, []).append(ch)
                    assigned = True
                    break
            if not assigned:
                pre_kids.append(ch)

        # ── Строим top-level список ────────────────────────────────────────────
        top: List[FNode] = [root_node]

        for ch in pre_kids:
            top.append(build(ch))

        for i, cn in enumerate(ctrl_nodes):
            chs = ctrl_to_kids.get(i)
            if not chs:
                continue
            code = cn.code.strip()
            if len(code) > 36:
                code = code[:34] + "…"

            # Yes = первый child (тело ветки), No = остальные (else/дополнительные)
            yes_chs = chs[:1]
            no_chs  = chs[1:]

            yes_chain = [build(ch) for ch in yes_chs]
            no_chain  = [build(ch) for ch in no_chs]

            for j in range(len(yes_chain) - 1):
                yes_chain[j].nxt = yes_chain[j + 1]
            for j in range(len(no_chain) - 1):
                no_chain[j].nxt = no_chain[j + 1]

            diamond = FNode(
                uid   = f"d_{e.call_index}_{cn.node_id}",
                kind  = "diamond",
                label = code,
                yes   = yes_chain[0] if yes_chain else None,
                no    = no_chain[0]  if no_chain  else None,
            )
            top.append(diamond)

        # Линейная цепочка верхнего уровня
        for i in range(len(top) - 1):
            top[i].nxt = top[i + 1]

        return root_node

    return build(seq[0])


# ── layout ────────────────────────────────────────────────────────────────────

def _layout(node: Optional[FNode], cx: float, cy: float,
            visited: Optional[set] = None) -> None:
    if node is None:
        return
    if visited is None:
        visited = set()
    if node.uid in visited:
        return
    visited.add(node.uid)

    node.cx = cx
    node.cy = cy
    h = DIA_H if node.kind == "diamond" else BOX_H

    if node.yes is not None:
        _layout(node.yes, cx, cy - h/2 - V_STEP, visited)

    if node.no is not None:
        _layout(node.no, cx + H_OFF, cy, visited)

    if node.nxt is not None:
        sub = _collect(node.yes) + _collect(node.no)
        same_col = [n for n in sub if abs(n.cx - cx) < 0.01]
        if same_col:
            min_y = min(
                n.cy - (DIA_H if n.kind == "diamond" else BOX_H) / 2
                for n in same_col
            )
            next_y = min_y - V_STEP
        else:
            next_y = cy - h/2 - V_STEP
        _layout(node.nxt, cx, next_y, visited)


def _collect(node: Optional[FNode],
             out: Optional[List[FNode]] = None,
             visited: Optional[set] = None) -> List[FNode]:
    if node is None:
        return out or []
    if out is None:
        out = []
    if visited is None:
        visited = set()
    if node.uid in visited:
        return out
    visited.add(node.uid)
    out.append(node)
    _collect(node.yes, out, visited)
    _collect(node.no,  out, visited)
    _collect(node.nxt, out, visited)
    return out


# ── render ────────────────────────────────────────────────────────────────────

def show_flowchart(result, *, out_png: str = "flowchart.png",
                   max_depth: int = -1) -> str:
    seq = result.sequence
    if max_depth >= 0:
        seq = [e for e in seq if e.depth <= max_depth]
    if not seq:
        print("[show_flowchart] Empty sequence.")
        return ""

    root = _build_tree(result, seq)
    if root is None:
        print("[show_flowchart] Could not build tree.")
        return ""

    _layout(root, 0.0, 0.0)
    nodes = _collect(root)
    if not nodes:
        return ""

    xs = [n.cx for n in nodes]
    ys = [n.cy for n in nodes]
    x_min = min(xs) - BOX_W - 0.5
    x_max = max(xs) + BOX_W + 0.5
    y_min = min(ys) - BOX_H - 1.2
    y_max = max(ys) + 1.8

    fig_w = max((x_max - x_min) * 0.82, 9)
    fig_h = max((y_max - y_min) * 0.70, 8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")

    n_dia = sum(1 for n in nodes if n.kind == "diamond")
    ax.text((x_min+x_max)/2, y_max - 0.15,
            f"{result.endpoint_name}  ·  {len(seq)} calls  ·  {n_dia} branches",
            ha="center", va="top", fontsize=10, color=TXT, fontfamily="monospace")

    drawn: set = set()

    def draw_edge(src, dst, color, yes_lbl=False, no_lbl=False):
        key = (src.uid, dst.uid)
        if key in drawn:
            return
        drawn.add(key)
        h_s = DIA_H if src.kind == "diamond" else BOX_H
        h_d = DIA_H if dst.kind == "diamond" else BOX_H
        if abs(dst.cx - src.cx) < 0.01:
            _arrow_down(ax, src.cx, src.cy - h_s/2 - 0.04,
                        dst.cy + h_d/2 + 0.04, color)
            if yes_lbl:
                _lbl(ax, src.cx + 0.1, src.cy - h_s/2 - 0.22, "Yes", C_YES)
        else:
            y_h  = src.cy
            x_f  = src.cx + DIA_W/2 + 0.1
            x_t  = dst.cx - BOX_W/2 - 0.1
            _arrow_horiz_down(ax, x_f, x_t, y_h, dst.cy + h_d/2 + 0.04, color)
            if no_lbl:
                _lbl(ax, x_f + 0.1, y_h, "No", C_NO)

    def draw_all(node, vis=None):
        if node is None:
            return
        if vis is None:
            vis = set()
        if node.uid in vis:
            return
        vis.add(node.uid)
        if node.yes is not None:
            draw_edge(node, node.yes, C_YES, yes_lbl=(node.kind == "diamond"))
            draw_all(node.yes, vis)
        if node.no is not None:
            draw_edge(node, node.no, C_NO, no_lbl=True)
            draw_all(node.no, vis)
        if node.nxt is not None:
            draw_edge(node, node.nxt, C_GRY)
            draw_all(node.nxt, vis)

    draw_all(root)

    for n in nodes:
        if n.kind == "diamond":
            _diamond(ax, n.cx, n.cy, n.label)
        elif n.kind == "root":
            _box(ax, n.cx, n.cy, n.label, "#3b1f6e", "#c084fc", fs=9.0, bold=True)
        else:
            _box(ax, n.cx, n.cy, n.label, "#102a44", "#38bdf8", fs=8.5)

    ax.legend(handles=[
        mpatches.Patch(color="#c084fc", label="Endpoint (root)"),
        mpatches.Patch(color="#38bdf8", label="Internal call"),
        mpatches.Patch(color="#2dd4bf", label="Branch (diamond)"),
        mpatches.Patch(color=C_YES,     label="Yes / loop body"),
        mpatches.Patch(color=C_NO,      label="No  / loop exit"),
    ], fontsize=7, loc="upper right", framealpha=0.85,
       facecolor="#161b22", edgecolor=C_GRY, labelcolor=TXT)

    plt.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=140, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[show_flowchart] ✓ {out_png}  ({len(nodes)} nodes, {n_dia} branches)")
    return out_png