import html

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx

@dataclass
class Endpoint:
    node_id:    str
    name:       str
    full_name:  str
    filename:   str
    # call-graph metrics
    in_degree:  int = 0   # always 0 for an endpoint, kept for transparency
    out_degree: int = 0   # fan-out: how many distinct internal methods it calls
    # raw callee names (for inspection)
    callees: List[str] = field(default_factory=list)


class EndpointDetector:
    """
    Detects structural endpoints using call-graph topology only.

    Parameters
    ----------
    G : nx.MultiDiGraph
        CPG graph (loaded from .dot via pydot + networkx).
    """

    def __init__(self, G: nx.MultiDiGraph) -> None:
        self._G = G

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self) -> List[Endpoint]:
        """
        Returns endpoints sorted descending by out_degree (fan-out).
        """
        internal   = self._collect_internal_methods()
        call_graph = self._build_call_graph(internal)
        return self._select_endpoints(internal, call_graph)

    # ── CPG attribute access ──────────────────────────────────────────────────

    def _attr(self, nid: str, key: str) -> str:
        raw = self._G.nodes.get(nid, {}).get(key, "")
        s   = html.unescape(str(raw)).strip()
        if len(s) >= 2 and s[0] == s[-1] == '"':
            s = s[1:-1]
        return s

    # ── filter 1: build the set of internal METHOD nodes ─────────────────────
    #
    # CPG-native gates — no naming heuristics:
    #
    #   a) label must be METHOD
    #   b) IS_EXTERNAL != "true"        (Joern CPG schema attribute)
    #   c) NAME must not start with "<" (Joern CPG schema convention for
    #                                    compiler-generated functions)

    def _is_internal(self, nid: str) -> bool:
        data = self._G.nodes[nid]

        # (a) label
        if data.get("label") != '"METHOD"':
            return False

        # (b) IS_EXTERNAL flag — set by Joern for resolved-but-not-defined symbols
        if data.get("IS_EXTERNAL") == '"true"':
            return False

        # (c) compiler-generated / anonymous — CPG schema: NAME begins with "<"
        name = data.get("NAME", "").strip('"')
        if name.startswith("<"):
            return False

        return True

    def _collect_internal_methods(self) -> Dict[str, dict]:
        return {
            nid: data
            for nid, data in self._G.nodes(data=True)
            if self._is_internal(nid)
        }

    # ── filter 2: build directed call graph over internal methods ─────────────
    #
    # Edge (u → v) exists when:
    #   1. u   CONTAINS  call_node       (AST containment edge)
    #   2. call_node is a CALL node
    #   3. call_node.METHOD_FULL_NAME resolves to some v in internal

    def _build_call_graph(
        self, internal: Dict[str, dict]
    ) -> nx.DiGraph:
        """Returns a simple DiGraph (no multi-edges) over internal node ids."""

        # index: FULL_NAME → node_id  (for fast callee resolution)
        fn_index: Dict[str, str] = {
            self._attr(nid, "FULL_NAME"): nid
            for nid in internal
        }

        cg = nx.DiGraph()
        cg.add_nodes_from(internal.keys())

        for method_nid in internal:
            for _, call_nid, edata in self._G.out_edges(method_nid, data=True):
                el = html.unescape(str(edata.get("label", ""))).strip().strip('"').upper()
                if el != "CONTAINS":
                    continue
                if self._G.nodes[call_nid].get("label") != '"CALL"':
                    continue

                callee_fn = self._attr(call_nid, "METHOD_FULL_NAME")
                if callee_fn in fn_index:
                    callee_nid = fn_index[callee_fn]
                    if callee_nid != method_nid:           # skip self-calls
                        cg.add_edge(method_nid, callee_nid)

        return cg

    # ── selection: in_degree = 0  AND  out_degree ≥ 1 ────────────────────────

    def _select_endpoints(
        self,
        internal: Dict[str, dict],
        cg: nx.DiGraph,
    ) -> List[Endpoint]:

        result: List[Endpoint] = []
        for nid in cg.nodes:
            in_d  = cg.in_degree(nid)
            out_d = cg.out_degree(nid)

            # Condition (1):  in_degree = 0  ∧  out_degree ≥ 1
            if in_d != 0 or out_d < 1:
                continue

            callees = [
                self._attr(v, "NAME")
                for v in cg.successors(nid)
            ]

            result.append(Endpoint(
                node_id    = nid,
                name       = self._attr(nid, "NAME"),
                full_name  = self._attr(nid, "FULL_NAME"),
                filename   = self._attr(nid, "FILENAME"),
                in_degree  = in_d,
                out_degree = out_d,
                callees    = sorted(callees),
            ))

        return sorted(result, key=lambda e: -e.out_degree)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_endpoints(endpoints: List[Endpoint], max_fn: int = 60) -> None:
    print(f"\n{'─'*72}")
    print(f"  Structural Endpoints detected: {len(endpoints)}")
    print(f"  Criterion: in_degree=0  ∧  out_degree≥1  in internal call graph")
    print(f"{'─'*72}")
    header = f"  {'rank':>4}  {'out_deg':>7}  {'name':28s}  full_name"
    print(header)
    print(f"  {'─'*4}  {'─'*7}  {'─'*28}  {'─'*max_fn}")
    for rank, ep in enumerate(endpoints, 1):
        print(
            f"  {rank:>4}  {ep.out_degree:>7}  {ep.name:28s}  "
            f"{ep.full_name[:max_fn]}"
        )
        print(f"         {'':>7}  callees: {', '.join(ep.callees[:6])}"
              + (" …" if len(ep.callees) > 6 else ""))
    print(f"{'─'*72}\n")