import html

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Dict, List, Tuple

import networkx as nx

# Used only with skip_non_request=True. Case-sensitive globs on "/" + FILENAME.
# Test and generated files are not reported and not counted as callers, so a
# test does not hide the handler it calls (a Go TestGetQuote "calls" GetQuote).
NON_REQUEST_FILES: Tuple[str, ...] = (
    "*_test.go", "*_test.py", "*/test_*.py", "*Test.java", "*Tests.java",
    "*/test/*", "*/tests/*", "*.pb.go", "*_pb2.py", "*_pb2_grpc.py",
)
# Start-up methods: hidden by NAME only, but still counted as callers.
NON_REQUEST_NAMES: Tuple[str, ...] = ("main", ":program", "init", "__init__")


@dataclass
class Entrypoint:
    node_id: str
    name: str
    full_name: str
    filename: str


class EntrypointDetector:
    """
    Detects structural entrypoints using call-graph topology only.

    Parameters
    ----------
    G : nx.MultiDiGraph
        CPG graph (loaded from .dot via pydot + networkx).
    skip_non_request : bool
        Apply NON_REQUEST_FILES and NON_REQUEST_NAMES. Default: off.
    """

    def __init__(self, G: nx.MultiDiGraph, skip_non_request: bool = False) -> None:
        self._G = G
        self._skip = skip_non_request

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self) -> List[Entrypoint]:
        internal = self._collect_internal_methods()
        call_graph = self._build_call_graph(internal)
        return self._select_entrypoints(internal, call_graph)

    # ── CPG attribute access ──────────────────────────────────────────────────

    def _attr(self, nid: str, key: str) -> str:
        raw = self._G.nodes.get(nid, {}).get(key, "")
        s = html.unescape(str(raw)).strip()
        if len(s) >= 2 and s[0] == s[-1] == '"':
            s = s[1:-1]
        return s

    def _is_non_request_file(self, nid: str) -> bool:
        path = "/" + self._attr(nid, "FILENAME")
        return any(fnmatchcase(path, p) for p in NON_REQUEST_FILES)

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

    def _build_call_graph(self, internal: Dict[str, dict]) -> nx.DiGraph:
        """Returns a simple DiGraph (no multi-edges) over internal node ids."""

        # index: FULL_NAME → node_id  (for fast callee resolution)
        fn_index: Dict[str, str] = {
            self._attr(nid, "FULL_NAME"): nid for nid in internal
        }

        cg = nx.DiGraph()
        cg.add_nodes_from(internal.keys())

        for method_nid in internal:
            for _, call_nid, edata in self._G.out_edges(method_nid, data=True):
                el = (
                    html.unescape(str(edata.get("label", "")))
                    .strip()
                    .strip('"')
                    .upper()
                )
                if el != "CONTAINS":
                    continue
                if self._G.nodes[call_nid].get("label") != '"CALL"':
                    continue

                callee_fn = self._attr(call_nid, "METHOD_FULL_NAME")
                if callee_fn in fn_index:
                    callee_nid = fn_index[callee_fn]
                    if callee_nid != method_nid:  # skip self-calls
                        cg.add_edge(method_nid, callee_nid)

        return cg

    # ── selection: in_degree = 0  AND  out_degree ≥ 1 ────────────────────────

    def _select_entrypoints(
        self,
        internal: Dict[str, dict],
        cg: nx.DiGraph,
    ) -> List[Entrypoint]:

        ignored = {nid for nid in cg.nodes if self._skip and self._is_non_request_file(nid)}

        result: List[Entrypoint] = []
        for nid in cg.nodes:
            if nid in ignored:
                continue
            in_d = sum(1 for caller in cg.predecessors(nid) if caller not in ignored)
            out_d = cg.out_degree(nid)

            # Condition (1):  in_degree = 0  ∧  out_degree ≥ 1
            if in_d != 0 or out_d < 1:
                continue

            if self._skip and self._attr(nid, "NAME") in NON_REQUEST_NAMES:
                continue

            result.append(
                Entrypoint(
                    node_id=nid,
                    name=self._attr(nid, "NAME"),
                    full_name=self._attr(nid, "FULL_NAME"),
                    filename=self._attr(nid, "FILENAME"),
                )
            )

        return result
