"""
cpg_trie_visualizer_matplotlib.py (ENHANCED WITH METHOD TRACKING)
==================================================================
Builds Trie from CPG/PDG graph and visualizes it in PNG via matplotlib.


ENHANCED FEATURES:
- Tracks which METHOD each log statement belongs to
- Outputs: Log -> Template -> Method mapping
- Selective punctuation removal (keeps hyphens for compound words)
- Memory-optimized (max_depth=5, max_nodes=100)
- Deduplication of code segments
- Removes leading wildcards, keeps trailing for variable matching

FIXES:
- _build_template: stops after first LITERAL block (no DDG-chain pollution)
- _trie_match: prefix match — message can be prefix of template (trailing wildcards optional)

Usage:
------
python cpg_trie_visualizer_matplotlib.py export.dot trie.png


Output Structure:
-----------------
LogMapping now includes:
  - bucket: log group identifier
  - message: original log message
  - call_node_id: CPG CALL node ID
  - method_name: function/method name containing this log
  - method_node_id: CPG METHOD node ID
  - template: matched template pattern
  - score: static token count
  - matched: bool


Зависимости
-----------
pip install networkx matplotlib pydot
"""


from __future__ import annotations


import html
import re
import string
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle, Polygon, PathPatch
from matplotlib.path import Path as MplPath


import pydot


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


LOG_METHOD_NAMES: Set[str] = {
    # Java logging frameworks
    "info", "warn", "warning", "debug", "error", "trace", "fatal", "log",

    # System output streams (for catching shutdown hooks, startup messages, etc.)
    "println",      # System.out.println(), System.err.println()
    "print",        # System.out.print(), System.err.print()

    # Go logrus / zap / zerolog
    "infof", "warnf", "debugf", "errorf", "fatalf", "panicf",
    "infoln", "warnln", "debugln", "errorln",
    "printf",
    "msg",
    "msgf",

    # SLF4J / Logback
    "slf4j",

    # Apache Commons Logging
    "commons",

    # Log4j
    "log4j",

    # Custom logger patterns
    "write",        # Writer.write()
    "WriteLine",
    "flush",        # BufferedWriter.flush()
}

AST_LABEL = "AST"
CALL_LABEL = "CALL"
REACHING_DEF_LABEL = "REACHING_DEF"
CDG_LABEL = "CDG"
METHOD_LABEL = "METHOD"
WILDCARD = "<*>"
STRING_RE = re.compile(r'["`\'](.*?)["`\']', re.DOTALL)
LITERAL_LABELS: Set[str] = {
    "LITERAL", "STRING", "STRING_LITERAL", "NUMBER_LITERAL", "FIELD_IDENTIFIER",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LogTemplate:
    call_node_id: str
    raw_template: str
    tokens: List[str]
    static_count: int


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    terminals: List[LogTemplate] = field(default_factory=list)


@dataclass
class LogMapping:
    bucket: str
    message: str
    call_node_id: str
    method_name: str
    method_node_id: str
    template: str
    score: int
    matched: bool


# ---------------------------------------------------------------------------
# Basic graph helpers
# ---------------------------------------------------------------------------


def _clean(v) -> str:
    return html.unescape(str(v).strip().strip('\"')).strip()


def _label(G: nx.MultiDiGraph, nid) -> str:
    return _clean(G.nodes[nid].get("label", "")).upper()


def _code(G: nx.MultiDiGraph, nid) -> str:
    return _clean(G.nodes[nid].get("CODE", ""))


def _ast_children(G: nx.MultiDiGraph, nid) -> List:
    return [
        v for _, v, d in G.edges(nid, data=True)
        if _clean(d.get("label", "")).upper() == AST_LABEL
    ]


def _reaching_def_predecessors(G: nx.MultiDiGraph, nid) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for u, v, d in G.in_edges(nid, data=True):
        if _clean(d.get("label", "")).upper() == REACHING_DEF_LABEL:
            out.append((u, _clean(d.get("VARIABLE", ""))))
    return out


def _normalize_code(s: str) -> str:
    s = str(s)
    s = html.unescape(s)
    s = s.replace(r'\"', '"')
    s = s.replace(r"\'", "'")
    s = s.replace(r"\\", "\\")
    s = re.sub(r"\s+", " ", s)
    return s.strip().strip('\"').strip()


def _arg_nodes(G: nx.MultiDiGraph, call_nid) -> List:
    args = []
    for v in _ast_children(G, call_nid):
        lbl = _label(G, v)
        if lbl in {"METHOD", "TYPE_REF", "NAMESPACE_BLOCK", "BLOCK"}:
            continue

        idx = G.nodes[v].get("ARGUMENT_INDEX", 999999)
        try:
            idx = int(str(idx).strip('\"'))
        except Exception:
            idx = 999999

        args.append((idx, v))

    args.sort(key=lambda x: x[0])
    return [v for _, v in args]


def _tokenize(s: str) -> List[str]:
    return s.strip().split()


# ---------------------------------------------------------------------------
# FIND PARENT METHOD FOR LOG CALL
# ---------------------------------------------------------------------------


def _find_parent_method(G: nx.MultiDiGraph, call_node_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the parent METHOD node for a given CALL node.
    Returns (method_name, method_node_id) or (None, None)
    """
    visited = set()
    queue = deque([call_node_id])

    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)

        if _label(G, nid) == METHOD_LABEL:
            method_name = _clean(G.nodes[nid].get("NAME", ""))
            return (method_name, str(nid))

        for parent_id, _, edge_data in G.in_edges(nid, data=True):
            if parent_id not in visited:
                queue.append(parent_id)

    return (None, None)


# ---------------------------------------------------------------------------
# SELECTIVE TEXT CLEANING (KEEPS HYPHENS)
# ---------------------------------------------------------------------------


def _clean_text_selective(text: str) -> str:
    """
    Remove ALL escape sequences and special symbols EXCEPT hyphens (-).
    Keeps hyphens for compound words like "user-failed", "connection-timeout".
    """
    if not text:
        return ""

    text = str(text)

    text = text.replace(r'\\', ' ')
    text = text.replace(r'\"', ' ')
    text = text.replace(r"\'", ' ')
    text = text.replace(r'\/', ' ')
    text = re.sub(r'\[a-zA-Z]', ' ', text)

    PUNCTUATION_TO_REMOVE = set(string.punctuation)
    PUNCTUATION_TO_REMOVE.discard('-')  # KEEP hyphens

    for p in PUNCTUATION_TO_REMOVE:
        text = text.replace(p, ' ')

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.lower()
    return text


def _normalize_code_selective(s: str) -> str:
    """Normalize CODE attributes with selective cleaning"""
    s = str(s)
    s = s.strip().strip('\'"')
    s = _clean_text_selective(s)
    return s


# ---------------------------------------------------------------------------
# Backward traversal over REACHING_DEF (OPTIMIZED)
# ---------------------------------------------------------------------------


def _backward_reaching_def(
    G: nx.MultiDiGraph,
    starts: List,
    max_depth: int = 5,
    max_nodes: int = 100,
) -> List[Tuple[str, str, str]]:
    """Optimized traversal with hard limits"""
    visited = set()
    queue = deque((n, 0, "") for n in starts)
    result: List[Tuple[str, str, str]] = []

    while queue and len(visited) < max_nodes:
        nid, depth, variable = queue.popleft()
        if nid in visited or depth > max_depth:
            continue

        visited.add(nid)
        code = _code(G, nid)
        label = _label(G, nid)
        if code:
            result.append((code, label, variable))

        if depth < max_depth and len(visited) < max_nodes:
            for pred, var_name in _reaching_def_predecessors(G, nid):
                if pred not in visited:
                    queue.append((pred, depth + 1, var_name))

    return result


# ---------------------------------------------------------------------------
# FIX: Build template — stops after first LITERAL block (no DDG-chain pollution)
# ---------------------------------------------------------------------------


def _build_template(code_labels: List[Tuple[str, str, str]]) -> str:
    """
    Build template from the FIRST LITERAL block only.

    FIX: stops traversal after finding the log message literal to avoid
    DDG-chain pollution. Without this fix, REACHING_DEF edges pull in tokens
    from sibling/predecessor nodes, turning 'request started <*>' into
    'request started <*> session <*>' — which breaks trie matching.

    Removes leading wildcards, keeps trailing for variable matching.
    """
    if not code_labels:
        return ""

    max_entries = min(len(code_labels), 50)
    code_labels = code_labels[:max_entries]

    parts: List[str] = []
    seen_codes: Set[str] = set()
    need_wildcard = False
    found_literal = False  # FIX: track whether we've seen a literal

    for code, label, variable in code_labels:
        if not code or not code.strip():
            continue

        code_clean = _normalize_code_selective(code)

        if not code_clean:
            continue

        code_key = code_clean.lower()

        if code_key in seen_codes:
            continue
        seen_codes.add(code_key)

        if label in LITERAL_LABELS:
            if need_wildcard and parts and parts[-1] != WILDCARD:
                parts.append(WILDCARD)
                need_wildcard = False

            words = code_clean.split()
            if words:
                parts.extend(words)
            found_literal = True
            continue

        # FIX: stop after first literal — no more DDG pollution
        if found_literal:
            need_wildcard = True
            break

        need_wildcard = True

    while parts and parts[0] == WILDCARD:
        parts.pop(0)

    merged = []
    for p in parts:
        if not (p == WILDCARD and merged and merged[-1] == WILDCARD):
            merged.append(p)

    if merged and merged[-1] != WILDCARD:
        merged.append(WILDCARD)

    return " ".join(merged).strip() if merged else ""


@dataclass
class LogTemplateWithMethod:
    call_node_id: str
    method_name: str
    method_node_id: str
    raw_template: str
    tokens: List[str]
    static_count: int


def build_templates_from_cpg(
    G: nx.MultiDiGraph,
    max_ddg_depth: int = 5,
) -> List[LogTemplateWithMethod]:
    """Build templates and track which method each log belongs to"""
    templates: List[LogTemplateWithMethod] = []

    for nid, attrs in G.nodes(data=True):
        if _label(G, nid) != CALL_LABEL:
            continue

        call_name = _clean(attrs.get("NAME", "")).lower()
        if call_name not in LOG_METHOD_NAMES:
            continue

        method_name, method_node_id = _find_parent_method(G, nid)
        if not method_name:
            method_name = "unknown"
            method_node_id = "unknown"

        args = _arg_nodes(G, nid)
        if not args:
            continue

        code_labels = _backward_reaching_def(G, args, max_depth=max_ddg_depth)
        if not code_labels:
            continue

        raw = _build_template(code_labels)
        if not raw:
            continue

        tokens = _tokenize(raw)
        static_count = sum(1 for t in tokens if t != WILDCARD)

        templates.append(LogTemplateWithMethod(
            call_node_id=str(nid),
            method_name=method_name,
            method_node_id=str(method_node_id),
            raw_template=raw,
            tokens=tokens,
            static_count=static_count,
        ))

    return templates


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------


def build_trie(templates: List[LogTemplateWithMethod]) -> TrieNode:
    root = TrieNode()

    for tmpl in templates:
        node = root
        for tok in tmpl.tokens:
            key = WILDCARD if tok == WILDCARD else tok.lower()
            if key not in node.children:
                node.children[key] = TrieNode()
            node = node.children[key]
        node.terminals.append(LogTemplate(
            call_node_id=tmpl.call_node_id,
            raw_template=tmpl.raw_template,
            tokens=tmpl.tokens,
            static_count=tmpl.static_count,
        ))

    return root


def _trie_match(
    node: TrieNode,
    msg: List[str],
    pos: int,
    results: List[LogTemplate],
) -> None:
    """
    FIX: Prefix match support.

    If the message ends but the template has only wildcard edges remaining,
    still count it as a match. This fixes cases where DDG pollution added
    extra tokens to the template (e.g. 'request started <*> session <*>')
    but the actual log message is simply 'request started'.
    """
    if pos == len(msg):
        # Standard terminal match
        results.extend(node.terminals)
        # FIX: prefix match — collect terminals reachable only via wildcards
        queue = []
        if WILDCARD in node.children:
            queue.append(node.children[WILDCARD])
        seen = set()
        while queue:
            n = queue.pop()
            nid = id(n)
            if nid in seen:
                continue
            seen.add(nid)
            results.extend(n.terminals)
            if WILDCARD in n.children:
                queue.append(n.children[WILDCARD])
        return

    tok = msg[pos].lower()

    if tok in node.children:
        _trie_match(node.children[tok], msg, pos + 1, results)

    if WILDCARD in node.children:
        wc = node.children[WILDCARD]
        for end in range(pos + 1, len(msg) + 1):
            _trie_match(wc, msg, end, results)


def map_logs(
    log_rows: List[Tuple[str, str]],
    root: TrieNode,
    templates_with_methods: List[LogTemplateWithMethod],
    min_static: int = 1,
) -> List[LogMapping]:
    """Map logs to templates and include method information"""
    mappings: List[LogMapping] = []

    method_lookup: Dict[str, Tuple[str, str]] = {}
    for tmpl in templates_with_methods:
        method_lookup[tmpl.call_node_id] = (tmpl.method_name, tmpl.method_node_id)

    for bucket, message in log_rows:
        message_clean = _clean_text_selective(message)
        tokens = message_clean.strip().split()

        candidates: List[LogTemplate] = []
        _trie_match(root, tokens, 0, candidates)
        candidates = [c for c in candidates if c.static_count >= min_static]

        if candidates:
            best = max(candidates, key=lambda t: (t.static_count, len(t.tokens)))
            method_name, method_node_id = method_lookup.get(
                best.call_node_id,
                ("unknown", "unknown")
            )
            mappings.append(LogMapping(
                bucket=bucket,
                message=message,
                call_node_id=best.call_node_id,
                method_name=method_name,
                method_node_id=method_node_id,
                template=best.raw_template,
                score=best.static_count,
                matched=True,
            ))
        else:
            mappings.append(LogMapping(
                bucket=bucket,
                message=message,
                call_node_id="",
                method_name="",
                method_node_id="",
                template="",
                score=0,
                matched=False,
            ))

    return mappings


# ---------------------------------------------------------------------------
# Trie visualization
# ---------------------------------------------------------------------------


def _trie_bfs(root: TrieNode):
    counter = [0]

    def new_id():
        counter[0] += 1
        return counter[0]

    root_id = new_id()
    queue = deque([(root, root_id, None, None)])

    while queue:
        trie_node, nid, parent_id, edge_key = queue.popleft()
        yield nid, parent_id, edge_key, trie_node

        for key, child in trie_node.children.items():
            cid = new_id()
            queue.append((child, cid, nid, key))


def _tree_layout(
    edges: List[Tuple[int, int]],
    root_id: int,
) -> Dict[int, Tuple[float, float]]:
    adj = defaultdict(list)
    for fr, to in edges:
        adj[fr].append(to)

    visit_order = []
    visited = set()
    bfs = deque([root_id])
    visited.add(root_id)

    while bfs:
        n = bfs.popleft()
        visit_order.append(n)
        for c in adj[n]:
            if c not in visited:
                visited.add(c)
                bfs.append(c)

    node_w = 2.8
    width = {}

    for n in reversed(visit_order):
        ch = adj[n]
        width[n] = max(node_w, sum(width[c] for c in ch)) if ch else node_w

    pos = {}
    x_start = {root_id: -width[root_id] / 2}
    depth = {root_id: 0}

    for n in visit_order:
        cx = x_start[n] + width[n] / 2
        pos[n] = (cx, -depth[n] * 2.5)

        cursor = x_start[n]
        for c in adj[n]:
            depth[c] = depth[n] + 1
            x_start[c] = cursor
            cursor += width[c]

    return pos


def _draw_curved_edge(ax, x0, y0, x1, y1, color="#93c5fd", lw=1.8):
    verts = [
        (x0, y0),
        (x0, (y0 + y1) / 2),
        (x1, (y0 + y1) / 2),
        (x1, y1),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
    ]
    path = MplPath(verts, codes)
    patch = PathPatch(path, facecolor="none", edgecolor=color, lw=lw)
    ax.add_patch(patch)


def _draw_circle_node(ax, x, y, label=None,
                      radius=0.38,
                      face="#2563eb",
                      edge="#1e3a8a"):
    circ = Circle((x, y), radius=radius, facecolor=face,
                  edgecolor=edge, linewidth=2)
    ax.add_patch(circ)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=8.5, color="white",
        family="monospace", weight="bold"
    )


def _draw_diamond_node(ax, x, y, label=None,
                       size=0.52,
                       face="#059669",
                       edge="#064e3b"):
    pts = [
        (x, y + size),
        (x + size, y),
        (x, y - size),
        (x - size, y),
    ]
    poly = Polygon(pts, closed=True, facecolor=face,
                   edgecolor=edge, linewidth=2)
    ax.add_patch(poly)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=8.5, color="white",
        family="monospace", weight="bold"
    )


def visualize_trie_matplotlib(
    root: TrieNode,
    output_path: str = "trie.png",
) -> None:
    node_rows = []
    edge_rows = []

    for nid, parent_id, edge_key, trie_node in _trie_bfs(root):
        term_ids = [t.call_node_id for t in trie_node.terminals]
        disp = "ROOT" if parent_id is None else (WILDCARD if edge_key == WILDCARD else edge_key)
        node_rows.append((nid, disp, bool(trie_node.terminals), term_ids))
        if parent_id is not None:
            edge_rows.append((parent_id, nid, edge_key))

    root_id = node_rows[0][0]
    pos = _tree_layout([(fr, to) for fr, to, _ in edge_rows], root_id)

    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")

    for fr, to, key in edge_rows:
        x0, y0 = pos[fr]
        x1, y1 = pos[to]
        _draw_curved_edge(ax, x0, y0, x1, y1)

        mx = (x0 + x1) / 2
        my = y0 * 0.35 + y1 * 0.65
        disp = WILDCARD if key == WILDCARD else key
        ax.text(
            mx, my, disp,
            ha="center", va="center",
            fontsize=8.5, color="#374151",
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor="white", edgecolor="none", alpha=0.9)
        )

    for nid, label, is_term, call_ids in node_rows:
        x, y = pos[nid]
        if is_term:
            _draw_diamond_node(ax, x, y)
        else:
            _draw_circle_node(ax, x, y)

        if call_ids:
            ax.text(
                x, y - 0.9,
                "→ " + ", ".join(call_ids),
                ha="center", va="top",
                fontsize=7.5, color="#065f46"
            )

    ax.text(
        0.5, 1.02,
        "Log Template Trie",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=18, weight="bold", color="#111827"
    )
    ax.text(
        0.5, 0.99,
        "Blue circle = internal trie node | Green diamond = terminal (CPG CALL node) | <*> = wildcard",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10, color="#6b7280"
    )

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    pad_x = 2.5
    pad_y = 2.0

    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[Trie] Saved matplotlib PNG -> {output_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    G: nx.MultiDiGraph,
    log_rows: List[Tuple[str, str]],
    trie_image: str = "trie.png",
    max_ddg_depth: int = 5,
    min_static: int = 1,
):
    templates = build_templates_from_cpg(G, max_ddg_depth=max_ddg_depth)
    root = build_trie(templates)
    visualize_trie_matplotlib(root, output_path=trie_image)
    mappings = map_logs(log_rows, root, templates, min_static=min_static)
    return templates, root, mappings


def load_dot_graph(dot_path: str) -> nx.MultiDiGraph:
    graphs = pydot.graph_from_dot_file(dot_path)
    if not graphs:
        raise ValueError(f"Failed to parse DOT file: {dot_path}")

    P = graphs[0]
    G = nx.MultiDiGraph()

    for node in P.get_nodes():
        name = node.get_name()
        if name in (None, "node", "graph", "edge"):
            continue
        nid = str(name).strip('"')
        attrs = {k: v for k, v in node.get_attributes().items()}
        G.add_node(nid, **attrs)

    for edge in P.get_edges():
        src = str(edge.get_source()).strip('"')
        dst = str(edge.get_destination()).strip('"')
        attrs = {k: v for k, v in edge.get_attributes().items()}
        G.add_edge(src, dst, **attrs)

    return G


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python cpg_trie_visualizer_matplotlib.py <export.dot> <trie.png>")
        sys.exit(1)

    dot_path = sys.argv[1]
    image_path = sys.argv[2]

    if not Path(dot_path).exists():
        print(f"ERROR: file not found: {dot_path}")
        sys.exit(1)

    print(f"[CPG] Loading DOT: {dot_path}")
    G = load_dot_graph(dot_path)
    print(f"[CPG] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    log_rows = [
        ("b1", '"received ad request (context_words=[clothing, tops])"'),
    ]

    templates, root, mappings = run_pipeline(
        G,
        log_rows,
        trie_image=image_path,
        max_ddg_depth=5,
        min_static=1,
    )

    print("\nTemplates with Methods:")
    if not templates:
        print("  (none)")
    else:
        for t in templates:
            print(f"  [{t.call_node_id}] {t.method_name}() -> '{t.raw_template}' (static={t.static_count})")

    print("\nLog to Method Mapping:")
    if not mappings:
        print("  (none)")
    else:
        for m in mappings:
            if m.matched:
                print(f"  ✓ Log Message: {m.message}")
                print(f"    Method: {m.method_name}()")
                print(f"    Template: '{m.template}' (score={m.score})")
                print(f"    CPG Call: {m.call_node_id} | CPG Method: {m.method_node_id}")
            else:
                print(f"  ✗ Log Message: {m.message}")
                print(f"    Status: UNMATCHED")

    print("\nDone. Trie image saved as: trie.png")


if __name__ == "__main__":
    main()
