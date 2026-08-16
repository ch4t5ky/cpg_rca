import html as _html
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _clean(raw) -> str:
    s = _html.unescape(str(raw)).strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1]
    return s


def _attr(G: nx.MultiDiGraph, nid: str, key: str) -> str:
    return _clean(G.nodes.get(nid, {}).get(key, ""))


def _label(G: nx.MultiDiGraph, nid: str) -> str:
    return _attr(G, nid, "label").upper()


def _edge_label(data: dict) -> str:
    return _clean(data.get("label", "")).upper()


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Parameter:
    """METHOD_PARAMETER_IN — CPG spec §Method"""

    index: int
    name: str
    type_full_name: str
    evaluation_strategy: str
    is_variadic: bool
    code: str
    node_id: str


@dataclass
class ReturnType:
    """METHOD_RETURN — CPG spec §Method"""

    type_full_name: str
    evaluation_strategy: str
    code: str
    node_id: str


@dataclass
class LocalVar:
    """LOCAL — CPG spec §Ast"""

    name: str
    type_full_name: str
    code: str
    line: str
    node_id: str


@dataclass
class CallSite:
    """
    CALL node inside the method — CPG spec §CallGraph / §Ast.
    We store only the id and callee full name; the callee body is NOT
    expanded here (use MethodConstructor recursively if needed).
    """

    node_id: str
    name: str  # short name
    method_full_name: str  # fully qualified callee name
    code: str
    line: str
    argument_index: str  # position in parent call, if nested


@dataclass
class CfgNode:
    """One node in the execution-ordered CFG subgraph."""

    node_id: str
    label: str  # AST node type: CALL, RETURN, IDENTIFIER, LITERAL …
    code: str
    line: str
    cfg_index: int  # position in execution order (0-based)


@dataclass
class PdgEdge:
    """
    One PDG edge inside the method.
    dep_type: CDG (control) or REACHING_DEF (data flow)
    variable:  the variable name carried by REACHING_DEF edges (else "")
    """

    src: str
    dst: str
    dep_type: str  # "CDG" | "REACHING_DEF"
    variable: str  # only for REACHING_DEF


@dataclass
class MethodGraph:
    """
    Complete graph representation of one method.

    Fields
    ──────
    method_node_id   source node id in the CPG
    name             short method name
    full_name        fully qualified name
    signature        type signature string
    filename         source file
    line_start        first line
    line_end          last line

    parameters       ordered list of METHOD_PARAMETER_IN nodes
    return_type      METHOD_RETURN node
    local_vars       LOCAL variable declarations

    call_sites       all CALL nodes — callee id not expanded
    cfg_nodes        CFG nodes in execution order
    cfg_edges        (src_id, dst_id, condition) list
                       condition: "" | "TRUE" | "FALSE" | "LOOP_TRUE" | "LOOP_FALSE"
    pdg_edges        PDG edges with dep_type and variable

    cfg_order        node_ids in topological order (None if graph is cyclic)
    cfg_order_approx node_ids in DFS reverse-postorder (always available)

    ast_node_ids     full set of AST node ids belonging to this method
    """

    method_node_id: str
    name: str
    full_name: str
    signature: str
    filename: str
    line_start: str
    line_end: str

    parameters: List[Parameter] = field(default_factory=list)
    return_type: Optional[ReturnType] = None
    local_vars: List[LocalVar] = field(default_factory=list)

    call_sites: List[CallSite] = field(default_factory=list)
    cfg_nodes: List[CfgNode] = field(default_factory=list)
    cfg_edges: List[Tuple[str, str, str]] = field(default_factory=list)
    pdg_edges: List[PdgEdge] = field(default_factory=list)

    cfg_order: Optional[List[str]] = None
    cfg_order_approx: List[str] = field(default_factory=list)

    ast_node_ids: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Constructor
# ─────────────────────────────────────────────────────────────────────────────


class MethodConstructor:
    """
    Builds a MethodGraph from a METHOD node id.

    Usage
    ─────
        mc = MethodConstructor(G)
        mg = mc.build("107374182777")
    """

    def __init__(self, G: nx.MultiDiGraph) -> None:
        self._G = G

    # ── public ────────────────────────────────────────────────────────────────

    def build(self, method_node_id: str) -> MethodGraph:
        G = self._G

        mg = MethodGraph(
            method_node_id=method_node_id,
            name=_attr(G, method_node_id, "NAME"),
            full_name=_attr(G, method_node_id, "FULL_NAME"),
            signature=_attr(G, method_node_id, "SIGNATURE"),
            filename=_attr(G, method_node_id, "FILENAME"),
            line_start=_attr(G, method_node_id, "LINE_NUMBER"),
            line_end=_attr(G, method_node_id, "LINE_NUMBER_END"),
        )

        # 1. Collect all AST nodes belonging to this method (BFS on AST edges)
        ast_nodes = self._collect_ast_nodes(method_node_id)
        mg.ast_node_ids = list(ast_nodes)

        # 2. Parameters (METHOD_PARAMETER_IN — direct AST children of METHOD)
        mg.parameters = self._collect_parameters(method_node_id)

        # 3. Return type (METHOD_RETURN — direct AST child of METHOD)
        mg.return_type = self._collect_return_type(method_node_id)

        # 4. Local variables (LOCAL nodes in AST subtree)
        mg.local_vars = self._collect_locals(ast_nodes)

        # 5. Call sites (CALL nodes in AST subtree)
        mg.call_sites = self._collect_call_sites(ast_nodes)

        # 6. CFG subgraph — edges carry "condition" attribute
        cfg_sub = self._build_cfg_subgraph(ast_nodes)
        mg.cfg_edges = [
            (src, dst, data.get("condition", ""))
            for src, dst, data in cfg_sub.edges(data=True)
        ]
        mg.cfg_nodes = self._ordered_cfg_nodes(cfg_sub)

        # 7. Execution order
        mg.cfg_order = self._topo_order(cfg_sub)
        mg.cfg_order_approx = self._dfs_order(cfg_sub)

        # 8. PDG (CDG + REACHING_DEF)
        mg.pdg_edges = self._collect_pdg_edges(ast_nodes)

        return mg

    # ── step 1: AST subtree ───────────────────────────────────────────────────

    def _collect_ast_nodes(self, root: str) -> set:
        visited, queue = set(), [root]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for _, dst, data in self._G.out_edges(cur, data=True):
                if _edge_label(data) == "AST":
                    queue.append(dst)
        return visited

    # ── step 2: parameters ────────────────────────────────────────────────────

    def _collect_parameters(self, method_nid: str) -> List[Parameter]:
        params: List[Parameter] = []
        G = self._G
        for _, dst, data in G.out_edges(method_nid, data=True):
            if _edge_label(data) != "AST":
                continue
            if _label(G, dst) != "METHOD_PARAMETER_IN":
                continue
            idx_raw = _attr(G, dst, "INDEX")
            params.append(
                Parameter(
                    index=int(idx_raw) if idx_raw.isdigit() else 0,
                    name=_attr(G, dst, "NAME"),
                    type_full_name=_attr(G, dst, "TYPE_FULL_NAME"),
                    evaluation_strategy=_attr(G, dst, "EVALUATION_STRATEGY"),
                    is_variadic=_attr(G, dst, "IS_VARIADIC").lower() == "true",
                    code=_attr(G, dst, "CODE"),
                    node_id=dst,
                )
            )
        return sorted(params, key=lambda p: p.index)

    # ── step 3: return type ───────────────────────────────────────────────────

    def _collect_return_type(self, method_nid: str) -> Optional[ReturnType]:
        G = self._G
        for _, dst, data in G.out_edges(method_nid, data=True):
            if _edge_label(data) != "AST":
                continue
            if _label(G, dst) == "METHOD_RETURN":
                return ReturnType(
                    type_full_name=_attr(G, dst, "TYPE_FULL_NAME"),
                    evaluation_strategy=_attr(G, dst, "EVALUATION_STRATEGY"),
                    code=_attr(G, dst, "CODE"),
                    node_id=dst,
                )
        return None

    # ── step 4: local variables ───────────────────────────────────────────────

    def _collect_locals(self, ast_nodes: set) -> List[LocalVar]:
        G = self._G
        locals_: List[LocalVar] = []
        for nid in ast_nodes:
            if _label(G, nid) == "LOCAL":
                locals_.append(
                    LocalVar(
                        name=_attr(G, nid, "NAME"),
                        type_full_name=_attr(G, nid, "TYPE_FULL_NAME"),
                        code=_attr(G, nid, "CODE"),
                        line=_attr(G, nid, "LINE_NUMBER"),
                        node_id=nid,
                    )
                )
        return sorted(locals_, key=lambda l: l.line or "0")

    # ── step 5: call sites ────────────────────────────────────────────────────

    def _collect_call_sites(self, ast_nodes: set) -> List[CallSite]:
        G = self._G
        calls: List[CallSite] = []
        for nid in ast_nodes:
            if _label(G, nid) != "CALL":
                continue
            calls.append(
                CallSite(
                    node_id=nid,
                    name=_attr(G, nid, "NAME"),
                    method_full_name=_attr(G, nid, "METHOD_FULL_NAME"),
                    code=_attr(G, nid, "CODE"),
                    line=_attr(G, nid, "LINE_NUMBER"),
                    argument_index=_attr(G, nid, "ARGUMENT_INDEX"),
                )
            )
        return sorted(calls, key=lambda c: c.line or "0")

    # ── step 6: CFG subgraph ──────────────────────────────────────────────────

    def _build_cfg_subgraph(self, ast_nodes: set) -> nx.DiGraph:
        G = self._G
        cfg = nx.DiGraph()

        for nid in ast_nodes:
            d = G.nodes[nid]
            cfg.add_node(
                nid,
                label=_clean(d.get("label", "")),
                code=_clean(d.get("CODE", ""))[:80],
                line=_clean(d.get("LINE_NUMBER", "")),
            )

        # Derive TRUE/FALSE labels from CONTROL_STRUCTURE AST structure
        branch_conditions = self._collect_branch_conditions(ast_nodes)

        for src, dst, data in G.edges(data=True):
            if src in ast_nodes and dst in ast_nodes:
                if _edge_label(data) == "CFG":
                    cond = branch_conditions.get((src, dst), "")
                    cfg.add_edge(src, dst, condition=cond)

        return cfg

    def _collect_branch_conditions(self, ast_nodes: set) -> Dict[Tuple[str, str], str]:
        """
        Determines TRUE/FALSE condition for CFG edges that originate from
        the condition expression of a CONTROL_STRUCTURE node.

        Algorithm
        ─────────
        For each CONTROL_STRUCTURE (IF / WHILE / FOR / DO) inside ast_nodes:

          1. Find the condition node via the CONDITION edge
             (src=CONTROL_STRUCTURE, dst=condition_expr).

          2. Find the body block via AST children ordered by ORDER:
               IF    → ORDER=2 is THEN-block, ORDER=3 is ELSE-block
               WHILE → ORDER=2 is body-block
               FOR   → ORDER=4 is body-block  (ORDER 1-3 = init/cond/post)
               DO    → ORDER=1 is body-block

          3. Build the set of all AST-descendants of the body block.

          4. For each CFG-successor of the condition node:
               • successor ∈ body_descendants  → "TRUE"  (enters loop/then)
               • successor ∉ body_descendants  → "FALSE" (skips / exits)

        Notes
        ─────
        • Joern emits CFG edges from the *condition expression* node,
          not from the CONTROL_STRUCTURE node itself.
        • The CONDITION edge (label="CONDITION") points from
          CONTROL_STRUCTURE → condition_expr.
        • For IF without ELSE the FALSE successor is the first node
          after the if-block (join point).
        • For loops (WHILE/FOR/DO) the TRUE successor enters the body
          and the FALSE successor exits the loop.
        """
        G = self._G
        result: Dict[Tuple[str, str], str] = {}

        for nid in ast_nodes:
            if _label(G, nid) != "CONTROL_STRUCTURE":
                continue

            cst = _clean(G.nodes[nid].get("CONTROL_STRUCTURE_TYPE", "")).upper()
            if cst not in ("IF", "WHILE", "FOR", "DO"):
                continue

            # ── 1. condition node (via CONDITION edge) ────────────────────
            cond_node: Optional[str] = None
            for _, dst, data in G.out_edges(nid, data=True):
                if _edge_label(data) == "CONDITION" and dst in ast_nodes:
                    cond_node = dst
                    break
            if cond_node is None:
                continue

            # ── 2. body block (via AST children, sorted by ORDER) ─────────
            children: List[Tuple[int, str]] = []
            for _, dst, data in G.out_edges(nid, data=True):
                if _edge_label(data) == "AST" and dst in ast_nodes:
                    order_raw = _clean(G.nodes[dst].get("ORDER", "99"))
                    order = int(order_raw) if order_raw.isdigit() else 99
                    children.append((order, dst))
            children.sort()

            body_order = {"IF": 2, "WHILE": 2, "FOR": 4, "DO": 1}.get(cst, 2)
            body_root: Optional[str] = next(
                (n for o, n in children if o == body_order), None
            )
            # Fallback: second AST child if ORDER-based lookup failed
            if body_root is None and len(children) >= 2:
                body_root = children[1][1]
            if body_root is None:
                continue

            else_root: Optional[str] = None
            if cst == "IF":
                else_root = next((n for o, n in children if o == 3), None)

            # ── 3. AST-descendants of body (and else) block ───────────────
            body_desc = self._ast_descendants(body_root, ast_nodes)
            else_desc: Set[str] = set()
            if else_root:
                else_desc = self._ast_descendants(else_root, ast_nodes)

            # ── 4. label CFG-successors of condition node ─────────────────
            cfg_succs = [
                dst
                for _, dst, data in G.out_edges(cond_node, data=True)
                if _edge_label(data) == "CFG" and dst in ast_nodes
            ]

            if len(cfg_succs) < 2:
                # Single successor — no branch to label
                continue

            for succ in cfg_succs:
                if succ in body_desc or succ == body_root:
                    label = "LOOP_TRUE" if cst in ("WHILE", "FOR", "DO") else "TRUE"
                elif else_root and (succ in else_desc or succ == else_root):
                    label = "FALSE"
                else:
                    label = "LOOP_FALSE" if cst in ("WHILE", "FOR", "DO") else "FALSE"
                result[(cond_node, succ)] = label

        return result

    def _ast_descendants(self, root: str, ast_nodes: set) -> Set[str]:
        """BFS over AST edges, restricted to ast_nodes."""
        visited: Set[str] = set()
        queue = [root]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for _, dst, data in self._G.out_edges(cur, data=True):
                if _edge_label(data) == "AST" and dst in ast_nodes:
                    queue.append(dst)
        return visited

    def _ordered_cfg_nodes(self, cfg: nx.DiGraph) -> List[CfgNode]:
        order = self._dfs_order(cfg)
        result = []
        for idx, nid in enumerate(order):
            d = cfg.nodes[nid]
            result.append(
                CfgNode(
                    node_id=nid,
                    label=d.get("label", ""),
                    code=d.get("code", ""),
                    line=d.get("line", ""),
                    cfg_index=idx,
                )
            )
        return result

    # ── execution order ───────────────────────────────────────────────────────

    def _topo_order(self, cfg: nx.DiGraph) -> Optional[List[str]]:
        """Topological sort — exact order for DAGs. Returns None if cyclic."""
        try:
            return list(nx.topological_sort(cfg))
        except nx.NetworkXUnfeasible:
            return None

    def _dfs_order(self, cfg: nx.DiGraph) -> List[str]:
        """
        DFS reverse-postorder — approximates execution order even for cyclic
        CFGs (loops). Entry = nodes with in_degree 0, or all if none found.
        """
        roots = [n for n in cfg.nodes if cfg.in_degree(n) == 0]
        if not roots:
            roots = list(cfg.nodes)[:1]
        visited, result = set(), []

        def dfs(n):
            if n in visited:
                return
            visited.add(n)
            for succ in cfg.successors(n):
                dfs(succ)
            result.append(n)

        for r in roots:
            dfs(r)
        return result[::-1]

    # ── step 8: PDG edges ─────────────────────────────────────────────────────

    def _collect_pdg_edges(self, ast_nodes: set) -> List[PdgEdge]:
        G = self._G
        edges: List[PdgEdge] = []
        for src, dst, data in G.edges(data=True):
            if src not in ast_nodes or dst not in ast_nodes:
                continue
            lbl = _edge_label(data)
            if lbl in ("CDG", "REACHING_DEF"):
                edges.append(
                    PdgEdge(
                        src=src,
                        dst=dst,
                        dep_type=lbl,
                        variable=_clean(data.get("VARIABLE", "")),
                    )
                )
        return edges


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────


def print_method_graph(mg: MethodGraph) -> None:
    w = 70
    print(f"\n{'═'*w}")
    print(f"  METHOD  {mg.name}")
    print(f"  {mg.full_name}")
    print(f"  {mg.filename}  lines {mg.line_start}–{mg.line_end}")
    print(f"  Signature: {mg.signature}")
    print(f"{'─'*w}")

    print(f"\n  PARAMETERS ({len(mg.parameters)})")
    for p in mg.parameters:
        variadic = " *variadic*" if p.is_variadic else ""
        print(
            f"    [{p.index}] {p.name}: {p.type_full_name}{variadic}  ({p.evaluation_strategy})"
        )

    rt = mg.return_type
    print(f"\n  RETURN TYPE")
    if rt:
        print(f"    {rt.type_full_name}  ({rt.evaluation_strategy})")
    else:
        print(f"    (not found)")

    print(f"\n  LOCAL VARS ({len(mg.local_vars)})")
    for lv in mg.local_vars:
        print(f"    L{lv.line:>4}  {lv.name}: {lv.type_full_name}")

    print(f"\n  CALL SITES ({len(mg.call_sites)})  [id only — not expanded]")
    for cs in mg.call_sites:
        print(f"    L{cs.line:>4}  {cs.code[:55]:55s}  → {cs.node_id}")

    print(f"\n  CFG  ({len(mg.cfg_nodes)} nodes, {len(mg.cfg_edges)} edges)")
    has_topo = mg.cfg_order is not None
    order_src = "topological" if has_topo else "DFS approx (cyclic CFG)"
    print(f"  Execution order: {order_src}")
    order = mg.cfg_order if has_topo else mg.cfg_order_approx
    node_map = {cn.node_id: cn for cn in mg.cfg_nodes}
    # build condition lookup from cfg_edges
    cond_map: Dict[Tuple[str, str], str] = {
        (src, dst): cond for src, dst, cond in mg.cfg_edges if cond
    }
    for idx, nid in enumerate(order[:30]):
        cn = node_map.get(nid)
        if cn:
            # show outgoing conditions for branching nodes
            out_conds = [
                f"→{cond}" for (s, _), cond in cond_map.items() if s == nid and cond
            ]
            cond_str = "  " + " ".join(out_conds) if out_conds else ""
            print(
                f"    {idx:>3}  L{cn.line:>4}  {cn.label:20s}  "
                f"{cn.code[:36]}{cond_str}"
            )
    if len(order) > 30:
        print(f"    … and {len(order)-30} more nodes")

    print(f"\n  PDG  ({len(mg.pdg_edges)} edges)")
    cdg_count = sum(1 for e in mg.pdg_edges if e.dep_type == "CDG")
    rd_count = sum(1 for e in mg.pdg_edges if e.dep_type == "REACHING_DEF")
    print(f"    CDG: {cdg_count}   REACHING_DEF: {rd_count}")
    shown = 0
    for e in mg.pdg_edges:
        if e.dep_type == "REACHING_DEF" and e.variable:
            src_code = node_map.get(e.src)
            dst_code = node_map.get(e.dst)
            sc = src_code.code[:30] if src_code else e.src
            dc = dst_code.code[:30] if dst_code else e.dst
            print(f"    DEF  {e.variable:15s}  {sc:30s} → {dc}")
            shown += 1
            if shown >= 8:
                print(f"    … and {rd_count - shown} more REACHING_DEF edges")
                break

    print(f"{'═'*w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pseudocode / Algorithm generator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PseudoLine:
    indent: int
    text: str
    node_id: str
    label: str


class PseudocodeGenerator:
    _LINE_PRIORITY = [
        "RETURN",
        "CONTROL_STRUCTURE",
        "CALL",
        "LOCAL",
        "JUMP_TARGET",
        "ASSIGNMENT",
    ]

    _SKIP = frozenset(
        {
            "METHOD",
            "METHOD_RETURN",
            "METHOD_PARAMETER_IN",
            "METHOD_PARAMETER_OUT",
            "BLOCK",
            "FILE",
            "NAMESPACE_BLOCK",
            "IDENTIFIER",
            "LITERAL",
            "FIELD_IDENTIFIER",
            "TYPE_REF",
            "METHOD_REF",
            "UNKNOWN",
            "ANNOTATION",
        }
    )

    def _line_priority(self, label: str) -> int:
        try:
            return self._LINE_PRIORITY.index(label.upper())
        except ValueError:
            return len(self._LINE_PRIORITY)

    def generate(self, mg: MethodGraph) -> List[PseudoLine]:
        call_index = {cs.node_id: cs for cs in mg.call_sites}
        local_index = {lv.node_id: lv for lv in mg.local_vars}

        best: Dict[str, CfgNode] = {}
        order = mg.cfg_order if mg.cfg_order else mg.cfg_order_approx
        node_map = {cn.node_id: cn for cn in mg.cfg_nodes}

        for nid in order:
            cn = node_map.get(nid)
            if cn is None or not cn.line:
                continue
            label = cn.label.upper()
            if label in self._SKIP:
                continue
            line = cn.line
            if line not in best:
                best[line] = cn
            else:
                prev = best[line]
                p_new = self._line_priority(label)
                p_old = self._line_priority(prev.label.upper())
                if p_new < p_old:
                    best[line] = cn
                elif p_new == p_old and len(cn.code) > len(prev.code):
                    best[line] = cn

        result: List[PseudoLine] = []

        params = ", ".join(f"{p.name}: {p.type_full_name}" for p in mg.parameters)
        ret = mg.return_type.type_full_name if mg.return_type else ""
        sig = f"FUNCTION {mg.name}({params})"
        if ret and ret not in ("void", "ANY", ""):
            sig += f" → {ret}"
        sig += ":"
        result.append(PseudoLine(0, sig, mg.method_node_id, "METHOD"))

        line_nodes = [
            best[k]
            for k in sorted(best.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        ]

        indent = 1
        control_stack: List[str] = []

        for cn in line_nodes:
            label = cn.label.upper()
            text = self._format(cn, label, call_index, local_index)
            if not text:
                continue

            result.append(PseudoLine(indent, text, cn.node_id, label))

            if label == "CONTROL_STRUCTURE":
                control_stack.append(cn.node_id)
                indent += 1
                continue

            if label == "RETURN":
                if control_stack:
                    control_stack.pop()
                indent = max(1, len(control_stack) + 1)
                continue

            # если это обычная строка после тела if, а return не было,
            # схлопываем только один guard-блок
            if control_stack and label not in ("CONTROL_STRUCTURE",):
                if (
                    text.startswith("log.WithField(")
                    or ":=" in text
                    or text.startswith("product :=")
                    or text.startswith("packagingInfo")
                    or text.startswith("recommendations")
                ):
                    indent = max(1, len(control_stack))
                    if len(control_stack) >= indent:
                        control_stack = control_stack[: indent - 1]

        return result

    def _build_cdg_children(self, mg: MethodGraph) -> Dict[str, List[str]]:
        children: Dict[str, List[str]] = {}
        for e in mg.pdg_edges:
            if e.dep_type == "CDG":
                children.setdefault(e.src, []).append(e.dst)
        return children

    def _controllers_of(
        self,
        node_id: str,
        control_children: Dict[str, List[str]],
        cache: Dict[str, Set[str]],
    ) -> Set[str]:
        if node_id in cache:
            return cache[node_id]

        controllers: Set[str] = set()
        for ctrl, kids in control_children.items():
            if node_id in kids:
                controllers.add(ctrl)
                controllers |= self._controllers_of(ctrl, control_children, cache)

        cache[node_id] = controllers
        return controllers

    # ── node → text ───────────────────────────────────────────────────────────

    def _format(
        self,
        cn: CfgNode,
        label: str,
        call_index: Dict[str, "CallSite"],
        local_index: Dict[str, "LocalVar"],
    ) -> str:
        code = cn.code.strip()

        if label == "CONTROL_STRUCTURE":
            keywords = (
                "if ",
                "for ",
                "while ",
                "switch ",
                "else",
                "try",
                "catch",
                "select",
            )
            if any(code.lower().startswith(k) for k in keywords):
                return code.rstrip("{").rstrip() + ":"
            return f"IF {code}:"

        if label == "CALL":
            return code

        if label == "RETURN":
            if code.lower().startswith("return"):
                return code
            return f"RETURN {code}" if code else "RETURN"

        if label == "LOCAL":
            lv = local_index.get(cn.node_id)
            if lv:
                return f"VAR {lv.name}: {lv.type_full_name}"
            return f"VAR {code}"

        if label == "JUMP_TARGET":
            return f"LABEL {code}"

        return code if code else ""
