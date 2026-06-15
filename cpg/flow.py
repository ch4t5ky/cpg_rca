# cpg/flow.py
from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Dict, List

import networkx as nx

from cpg.method import MethodConstructor, MethodGraph


@dataclass
class MethodEntry:
    method_graph: MethodGraph
    depth:        int
    call_index:   int


@dataclass
class EndpointFlowResult:
    endpoint_node_id:   str
    endpoint_name:      str
    endpoint_full_name: str
    sequence:           List[MethodEntry] = field(default_factory=list)
    cycle_warnings:     List[str]         = field(default_factory=list)


class EndpointFlow:
    """
    Restores all methods reachable from a METHOD node_id in true CFG
    execution order. Each callee is expanded inline every time its CALL
    node appears in the parent's CFG — no global deduplication.
    Only direct/indirect recursion (active call stack) is guarded.

    Parameters
    ----------
    G         : nx.MultiDiGraph
    max_depth : int   (-1 = unlimited)

    Usage
    ─────
        flow   = EndpointFlow(G, max_depth=10)
        result = flow.build("107374182770")

        for e in result.sequence:
            print(f"[{e.call_index:>3}] depth={e.depth}  {e.method_graph.name}")
    """

    def __init__(self, G: nx.MultiDiGraph, max_depth: int = -1) -> None:
        self._G         = G
        self._max_depth = max_depth
        self._mc        = MethodConstructor(G)
        self._fn_index  = self._build_fn_index()

    # ── public ────────────────────────────────────────────────────────────────

    def build(self, endpoint_node_id: str) -> EndpointFlowResult:
        result = EndpointFlowResult(
            endpoint_node_id   = endpoint_node_id,
            endpoint_name      = self._attr(endpoint_node_id, "NAME"),
            endpoint_full_name = self._attr(endpoint_node_id, "FULL_NAME"),
        )
        counter: List[int] = [0]
        self._expand(
            full_name = result.endpoint_full_name,
            depth     = 0,
            counter   = counter,
            in_stack  = set(),
            result    = result,
        )
        return result

    # ── inline recursive expansion ────────────────────────────────────────────

    def _expand(
        self,
        full_name: str,
        depth:     int,
        counter:   List[int],
        in_stack:  set,
        result:    EndpointFlowResult,
    ) -> None:
        if self._max_depth >= 0 and depth > self._max_depth:
            return

        # Guard only against active recursion (direct/indirect cycles).
        # Do NOT guard against re-visiting a method that completed earlier —
        # that would suppress legitimate repeated calls (e.g. renderHTTPError
        # called 5 times in viewCartHandler).
        if full_name in in_stack:
            result.cycle_warnings.append(full_name)
            return

        nid = self._fn_index.get(full_name)
        if nid is None:
            return  # external or unresolved

        mg = self._mc.build(nid)

        # Register this invocation at the current counter position
        result.sequence.append(MethodEntry(
            method_graph = mg,
            depth        = depth,
            call_index   = counter[0],
        ))
        counter[0] += 1

        in_stack.add(full_name)

        # Walk CFG order and expand each callee inline at its call site.
        # Same callee appearing N times in the CFG → expanded N times.
        order     = mg.cfg_order if mg.cfg_order else mg.cfg_order_approx
        order_idx = {n: i for i, n in enumerate(order)}

        cfg_calls = sorted(
            [cs for cs in mg.call_sites if cs.node_id in order_idx],
            key=lambda cs: order_idx[cs.node_id],
        )

        for cs in cfg_calls:
            callee = cs.method_full_name
            if not callee or callee == full_name:
                continue
            if callee not in self._fn_index:
                continue  # external — not expanded

            self._expand(
                full_name = callee,
                depth     = depth + 1,
                counter   = counter,
                in_stack  = in_stack,
                result    = result,
            )

        in_stack.discard(full_name)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_fn_index(self) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for nid, data in self._G.nodes(data=True):
            if str(data.get("label", "")).strip().strip('"').upper() != "METHOD":
                continue
            if str(data.get("IS_EXTERNAL", "")).strip().strip('"').lower() == "true":
                continue
            if str(data.get("NAME", "")).strip().strip('"').startswith("<"):
                continue
            fn = self._clean(data.get("FULL_NAME", ""))
            if fn:
                index[fn] = nid
        return index

    def _attr(self, nid: str, key: str) -> str:
        return self._clean(self._G.nodes.get(nid, {}).get(key, ""))

    @staticmethod
    def _clean(raw) -> str:
        s = html.unescape(str(raw)).strip()
        if len(s) >= 2 and s[0] == s[-1] == '"':
            s = s[1:-1]
        return s


# ── pretty printer ────────────────────────────────────────────────────────────

def print_endpoint_flow(result: EndpointFlowResult, show_cfg: bool = False) -> None:
    w = 72
    print(f"\n{'═' * w}")
    print(f"  ENDPOINT FLOW  {result.endpoint_name}")
    print(f"  {result.endpoint_full_name}")
    print(f"  Total invocations : {len(result.sequence)}")
    if result.cycle_warnings:
        print(f"  ⚠ Cycles skipped : {', '.join(set(result.cycle_warnings))}")
    print(f"{'─' * w}")

    for e in result.sequence:
        mg     = e.method_graph
        indent = "  " * e.depth
        topo   = "topo" if mg.cfg_order else "dfs-approx"
        print(f"  [{e.call_index:>3}] {indent}{mg.name}")
        print(f"        {indent}full : {mg.full_name}")
        print(f"        {indent}cfg  : {len(mg.cfg_nodes)} nodes  ({topo})")
        if show_cfg:
            order    = mg.cfg_order if mg.cfg_order else mg.cfg_order_approx
            node_map = {cn.node_id: cn for cn in mg.cfg_nodes}
            for idx, nid in enumerate(order[:10]):
                cn = node_map.get(nid)
                if cn:
                    print(f"        {indent}  {idx:>3} L{cn.line:>4}  {cn.label:18s}  {cn.code[:38]}")
        print()

    print(f"{'═' * w}\n")