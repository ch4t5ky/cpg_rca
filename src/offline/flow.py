"""Build semantic control-flow graphs directly from Code Property Graph methods.

No execution paths are enumerated.  For each reachable method, the public
result contains one SemanticFlowGraph whose nodes are source-level operations
and whose edges are possible control-flow transitions.
"""
from __future__ import annotations

import html
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from src.offline.method import CfgNode, MethodConstructor, MethodGraph

__all__ = [
    "SemanticUnitKind", "VariableKind", "VariableRef", "DataDependency",
    "SemanticUnit", "SemanticEdge", "SemanticFlowGraph", "MethodEntry",
    "ExternalCall", "EntrypointFlowResult", "EntrypointFlow",
    "print_entrypoint_flow",
]


# ── helpers ───────────────────────────────────────────────────────────────

def _clean(raw: object) -> str:
    value = html.unescape(str(raw)).strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    return value


def _node_attr(graph: nx.MultiDiGraph, node_id: str, *keys: str) -> str:
    data = graph.nodes.get(node_id, {})
    for key in keys:
        value = _clean(data.get(key, ""))
        if value:
            return value
    return ""


def _edge_label(data: dict) -> str:
    return _clean(data.get("label", "")).upper()


# ── semantic model ───────────────────────────────────────────────────────


class SemanticUnitKind(StrEnum):
    DECLARATION = "declaration"
    ASSIGNMENT = "assignment"
    CALL = "call"
    LOG_CALL = "log_call"
    CONDITION = "condition"
    LOOP = "loop"
    RETURN = "return"
    JUMP = "jump"
    UNKNOWN = "unknown"


class VariableKind(StrEnum):
    LOCAL = "local"
    PARAMETER = "parameter"
    FIELD = "field"
    RECEIVER = "receiver"
    GLOBAL = "global"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VariableRef:
    name: str
    declaration_node_id: Optional[str] = None
    type_full_name: str = ""
    kind: VariableKind = VariableKind.UNKNOWN


@dataclass(frozen=True)
class DataDependency:
    variable: VariableRef
    definition_node_id: str
    use_node_id: str


@dataclass
class SemanticUnit:
    """One source-level operation represented by a semantic AST anchor."""

    node_id: str
    kind: SemanticUnitKind
    code: str
    line: str
    raw_cfg_node_ids: List[str] = field(default_factory=list)
    defines: List[VariableRef] = field(default_factory=list)
    uses: List[VariableRef] = field(default_factory=list)
    dependencies: List[DataDependency] = field(default_factory=list)
    call_node_ids: List[str] = field(default_factory=list)
    callee_full_names: List[str] = field(default_factory=list)
    internal_callee_full_names: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SemanticEdge:
    source_id: str
    target_id: str
    condition: str = ""


@dataclass
class SemanticFlowGraph:
    """One directed cyclic semantic graph for a method."""

    method_node_id: str
    method_full_name: str
    start_node_id: str
    nodes: Dict[str, SemanticUnit] = field(default_factory=dict)
    edges: Dict[Tuple[str, str, str], SemanticEdge] = field(default_factory=dict)
    return_node_ids: Set[str] = field(default_factory=set)

    input_parameters: list[tuple[str, str]] = field(default_factory=list)
    output_parameters: list[tuple[str, str]] = field(default_factory=list)

    def add_edge(self, source_id: str, target_id: str, condition: str = "") -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        edge = SemanticEdge(source_id, target_id, condition)
        self.edges[(source_id, target_id, condition)] = edge

    @property
    def edge_list(self) -> List[SemanticEdge]:
        return list(self.edges.values())


# ── flow result ───────────────────────────────────────────────────────────


@dataclass
class MethodEntry:
    method_graph: MethodGraph
    depth: int
    call_index: int
    caller_full_name: str = ""
    via_call_node_id: Optional[str] = None


@dataclass
class ExternalCall:
    caller_full_name: str
    callee_full_name: str
    call_code: str
    line: str
    depth: int
    call_index: int


@dataclass
class EntrypointFlowResult:
    entrypoint_node_id: str
    entrypoint_name: str
    entrypoint_full_name: str
    sequence: List[MethodEntry] = field(default_factory=list)
    external_calls: List[ExternalCall] = field(default_factory=list)
    cycle_warnings: List[str] = field(default_factory=list)
    max_depth_reached: bool = False
    semantic_graphs: Dict[str, SemanticFlowGraph] = field(default_factory=dict)

    @property
    def total_invocations(self) -> int:
        return len(self.sequence)

    @property
    def unique_methods(self) -> Set[str]:
        return {entry.method_graph.full_name for entry in self.sequence}

    def summary(self) -> str:
        node_count = sum(len(graph.nodes) for graph in self.semantic_graphs.values())
        edge_count = sum(len(graph.edges) for graph in self.semantic_graphs.values())
        lines = [
            f"entrypoint : {self.entrypoint_name}",
            f"full name : {self.entrypoint_full_name}",
            f"reachable methods : {len(self.semantic_graphs)}",
            f"semantic nodes : {node_count}",
            f"semantic edges : {edge_count}",
            f"external calls : {len(self.external_calls)}",
        ]
        if self.cycle_warnings:
            lines.append(f"recursive calls skipped : {', '.join(sorted(set(self.cycle_warnings)))}")
        if self.max_depth_reached:
            lines.append("max_depth reached — callees were not expanded further")
        return "\n".join(lines)


# ── direct CFG → semantic graph ──────────────────────────────────────────


class EntrypointFlow:
    """Build semantic method graphs directly from CFG edges.

    No MethodPath, PathSegment or path enumeration is used.  The optional
    max_paths and max_loop_unroll arguments are retained only for backward
    compatible construction calls and have no effect on graph construction.
    """

    _SEMANTIC_ANCHOR_LABELS = frozenset({
        "CALL", "LOCAL", "RETURN", "METHOD_RETURN", "CONTROL_STRUCTURE",
        "JUMP_TARGET",
    })
    _LOOP_TYPES = frozenset({"WHILE", "FOR", "DO"})

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        max_depth: int = -1,
        max_loop_unroll: int = 1,
        max_paths: int = 256,
    ) -> None:
        self._G = graph
        self._max_depth = max_depth
        self._max_loop_unroll = max_loop_unroll
        self._max_paths = max_paths
        self._mc = MethodConstructor(graph)
        self._fn_index = self._build_fn_index()
        self._name_index = self._build_name_index()

    # ── public ──────────────────────────────────────────────────────────

    def build(self, entrypoint_node_id: str) -> EntrypointFlowResult:
        full_name = _node_attr(self._G, entrypoint_node_id, "FULLNAME", "FULL_NAME")
        result = EntrypointFlowResult(
            entrypoint_node_id=entrypoint_node_id,
            entrypoint_name=_node_attr(self._G, entrypoint_node_id, "NAME"),
            entrypoint_full_name=full_name,
        )
        if full_name:
            self._expand_method(full_name, 0, [0], set(), set(), result, "", None)
        return result

    def build_by_name(self, name: str) -> Optional[EntrypointFlowResult]:
        candidates = self._name_index.get(name, [])
        return self.build(candidates[0]) if candidates else None

    def all_entrypoints(self) -> List[str]:
        incoming: Dict[str, int] = {name: 0 for name in self._fn_index}
        for full_name, node_id in self._fn_index.items():
            method = self._mc.build(node_id)
            for call in method.call_sites:
                callee = call.method_full_name
                if callee in incoming and callee != full_name:
                    incoming[callee] += 1
        return [self._fn_index[name] for name, degree in incoming.items() if degree == 0]

    # ── interprocedural discovery ───────────────────────────────────────

    def _expand_method(
        self,
        full_name: str,
        depth: int,
        counter: List[int],
        in_stack: Set[str],
        expanded: Set[str],
        result: EntrypointFlowResult,
        caller_full_name: str,
        via_call_node_id: Optional[str],
    ) -> None:
        if self._max_depth >= 0 and depth > self._max_depth:
            result.max_depth_reached = True
            return
        if full_name in in_stack:
            result.cycle_warnings.append(full_name)
            return
        if full_name in expanded:
            return

        method_node_id = self._fn_index.get(full_name)
        if method_node_id is None:
            return

        method = self._mc.build(method_node_id)
        semantic_graph = self._build_semantic_graph(method)
        result.semantic_graphs[full_name] = semantic_graph
        result.sequence.append(MethodEntry(
            method_graph=method,
            depth=depth,
            call_index=counter[0],
            caller_full_name=caller_full_name,
            via_call_node_id=via_call_node_id,
        ))
        counter[0] += 1
        expanded.add(full_name)
        in_stack.add(full_name)

        for call in method.call_sites:
            callee = call.method_full_name
            if not callee or callee == full_name:
                continue
            if callee in self._fn_index:
                self._expand_method(
                    callee, depth + 1, counter, in_stack, expanded, result,
                    full_name, call.node_id,
                )
            else:
                result.external_calls.append(ExternalCall(
                    caller_full_name=full_name,
                    callee_full_name=callee,
                    call_code=call.code,
                    line=call.line,
                    depth=depth,
                    call_index=counter[0],
                ))
                counter[0] += 1

        in_stack.discard(full_name)

    # ── one method graph ─────────────────────────────────────────────────

    def _build_semantic_graph(self, method: MethodGraph) -> SemanticFlowGraph:
        """Map CFG nodes/edges directly to semantic nodes/edges.

        Complexity is linear in the method representation: AST edges + CFG
        nodes/edges + PDG edges.  No paths are generated.
        """
        ast_nodes = set(method.ast_node_ids)
        parent: Dict[str, str] = {}
        for source in ast_nodes:
            for _, target, data in self._G.out_edges(source, data=True):
                if target in ast_nodes and _edge_label(data) == "AST":
                    parent[target] = source

        anchor_cache: Dict[str, str] = {}

        def anchor(node_id: str) -> str:
            if node_id in anchor_cache:
                return anchor_cache[node_id]
            current = node_id
            candidate = ""
            seen: Set[str] = set()
            while current and current not in seen:
                seen.add(current)
                label = _node_attr(self._G, current, "label").upper()
                if label in self._SEMANTIC_ANCHOR_LABELS:
                    candidate = current
                parent_id = parent.get(current)
                if not parent_id:
                    break
                parent_label = _node_attr(self._G, parent_id, "label").upper()
                if parent_label in {"METHOD", "BLOCK", "FILE", "NAMESPACE_BLOCK"}:
                    break
                current = parent_id
            anchor_cache[node_id] = candidate or node_id
            return anchor_cache[node_id]

        def unit_kind(anchor_id: str) -> SemanticUnitKind:
            label = _node_attr(self._G, anchor_id, "label").upper()
            if label == "LOCAL":
                return SemanticUnitKind.DECLARATION
            if label in {"RETURN", "METHOD_RETURN"}:
                return SemanticUnitKind.RETURN
            if label == "JUMP_TARGET":
                return SemanticUnitKind.JUMP
            if label == "CONTROL_STRUCTURE":
                control_type = _node_attr(self._G, anchor_id, "CONTROL_STRUCTURE_TYPE").upper()
                return SemanticUnitKind.LOOP if control_type in self._LOOP_TYPES else SemanticUnitKind.CONDITION
            if label == "CALL":
                name = _node_attr(self._G, anchor_id, "NAME")
                if name.startswith("<operator>.assignment") or name.endswith(".assignment"):
                    return SemanticUnitKind.ASSIGNMENT
                return SemanticUnitKind.CALL
            return SemanticUnitKind.UNKNOWN

        # Variable identities.
        references: Dict[str, VariableRef] = {}
        for local in method.local_vars:
            references.setdefault(local.name, VariableRef(
                name=local.name,
                declaration_node_id=local.node_id,
                type_full_name=local.type_full_name,
                kind=VariableKind.LOCAL,
            ))
        for parameter in method.parameters:
            references.setdefault(parameter.name, VariableRef(
                name=parameter.name,
                declaration_node_id=parameter.node_id,
                type_full_name=parameter.type_full_name,
                kind=VariableKind.PARAMETER,
            ))

        def variable_ref(name: str) -> VariableRef:
            return references.get(name, VariableRef(name=name))

        def unique(values: Iterable[VariableRef]) -> List[VariableRef]:
            output: List[VariableRef] = []
            seen: Set[Tuple[str, Optional[str]]] = set()
            for value in values:
                key = (value.name, value.declaration_node_id)
                if key not in seen:
                    seen.add(key)
                    output.append(value)
            return output

        # Index PDG once, by semantic anchor.
        defs_by_anchor: Dict[str, List[VariableRef]] = defaultdict(list)
        uses_by_anchor: Dict[str, List[VariableRef]] = defaultdict(list)
        deps_by_anchor: Dict[str, List[DataDependency]] = defaultdict(list)
        for pdg_edge in method.pdg_edges:
            if pdg_edge.dep_type != "REACHING_DEF" or not pdg_edge.variable:
                continue
            variable = variable_ref(pdg_edge.variable)
            defs_by_anchor[anchor(pdg_edge.src)].append(variable)
            uses_by_anchor[anchor(pdg_edge.dst)].append(variable)
            deps_by_anchor[anchor(pdg_edge.dst)].append(DataDependency(
                variable=variable,
                definition_node_id=pdg_edge.src,
                use_node_id=pdg_edge.dst,
            ))

        call_sites_by_anchor: Dict[str, List[object]] = defaultdict(list)
        for call in method.call_sites:
            call_sites_by_anchor[anchor(call.node_id)].append(call)

        raw_nodes_by_anchor: Dict[str, List[CfgNode]] = defaultdict(list)
        for cfg_node in method.cfg_nodes:
            raw_nodes_by_anchor[anchor(cfg_node.node_id)].append(cfg_node)

        graph = SemanticFlowGraph(
            method_node_id=method.method_node_id,
            method_full_name=method.full_name,
            start_node_id=f"start@{method.full_name}",
            input_parameters=self._extract_input_parameters(method),
            output_parameters=self._extract_output_parameters(method),
        )

        def make_unit(anchor_id: str, raw_nodes: List[CfgNode]) -> SemanticUnit:
            data = self._G.nodes.get(anchor_id, {})
            code = _clean(data.get("CODE", ""))
            if not code:
                code = max((node.code for node in raw_nodes), key=len, default="")
            line = _clean(data.get("LINE_NUMBER", "")) or (raw_nodes[0].line if raw_nodes else "")
            calls = call_sites_by_anchor.get(anchor_id, [])
            callees = [call.method_full_name for call in calls if call.method_full_name]
            internal = [callee for callee in callees if callee in self._fn_index]
            return SemanticUnit(
                node_id=anchor_id,
                kind=unit_kind(anchor_id),
                code=code,
                line=line,
                raw_cfg_node_ids=[node.node_id for node in raw_nodes],
                defines=unique(defs_by_anchor.get(anchor_id, [])),
                uses=unique(uses_by_anchor.get(anchor_id, [])),
                dependencies=deps_by_anchor.get(anchor_id, []),
                call_node_ids=[call.node_id for call in calls],
                callee_full_names=list(dict.fromkeys(callees)),
                internal_callee_full_names=list(dict.fromkeys(internal)),
            )

        # Register every non-empty semantic anchor once.
        transparent_anchors: Set[str] = set()
        for anchor_id, raw_nodes in raw_nodes_by_anchor.items():
            unit = make_unit(anchor_id, raw_nodes)
            # Technical nodes not represented by an executable semantic anchor
            # are transparent: edges pass through them below.
            if unit.kind is SemanticUnitKind.UNKNOWN:
                transparent_anchors.add(anchor_id)
                continue
            graph.nodes[anchor_id] = unit

            if unit.kind == SemanticUnitKind.RETURN:
                graph.return_node_ids.add(anchor_id)

        raw_cfg = nx.DiGraph()
        for src, dst, condition in method.cfg_edges:
            raw_cfg.add_edge(src, dst, condition=condition)
        for node in method.cfg_nodes:
            raw_cfg.add_node(node.node_id)

        # Each raw node has either a represented unit or is transparent.
        represented_for_raw: Dict[str, Optional[str]] = {}
        for raw_node_id in raw_cfg.nodes:
            anchor_id = anchor(raw_node_id)
            represented_for_raw[raw_node_id] = anchor_id if anchor_id in graph.nodes else None

        def normalize_condition(source_id: str, target_id: str, raw_condition: str) -> str:
            source = graph.nodes.get(source_id)
            target = graph.nodes.get(target_id)
            if source and source.kind == SemanticUnitKind.LOOP:
                if raw_condition == "LOOP_TRUE":
                    return "body"
                if raw_condition == "LOOP_FALSE":
                    return f"exit: not ({source.code})" if source.code else "exit"
            if target and target.kind == SemanticUnitKind.LOOP and raw_condition in {"", "LOOP_BODY", "LOOP_TRUE"}:
                return "next iteration"
            if raw_condition == "LOOP_BODY":
                return "next iteration"
            return raw_condition

        # Contract transparent raw CFG nodes. For every represented source,
        # walk only through transparent nodes until the next represented unit.
        for source_raw, source_unit_id in represented_for_raw.items():
            if source_unit_id is None:
                continue
            queue = deque()
            for successor in raw_cfg.successors(source_raw):
                raw_condition = raw_cfg.edges[source_raw, successor].get("condition", "")
                queue.append((successor, raw_condition))
            visited: Set[Tuple[str, str]] = set()
            while queue:
                current_raw, inherited_condition = queue.popleft()
                state = (current_raw, inherited_condition)
                if state in visited:
                    continue
                visited.add(state)
                target_unit_id = represented_for_raw.get(current_raw)
                if target_unit_id is not None:
                    if target_unit_id != source_unit_id:
                        graph.add_edge(
                            source_unit_id,
                            target_unit_id,
                            normalize_condition(source_unit_id, target_unit_id, inherited_condition),
                        )
                    continue
                for successor in raw_cfg.successors(current_raw):
                    own_condition = raw_cfg.edges[current_raw, successor].get("condition", "")
                    queue.append((successor, inherited_condition or own_condition))

        # Connect synthetic start to all represented nodes reachable from raw roots.
        if method.method_node_id in raw_cfg:
            root_raw_node_id = method.method_node_id
        else:
            order = method.cfg_order or method.cfg_order_approx
            root_raw_node_id = order[0] if order else None

        if root_raw_node_id is not None:
            queue = deque([root_raw_node_id])
            seen: Set[str] = set()

            while queue:
                current = queue.popleft()

                if current in seen:
                    continue
                seen.add(current)

                target_id = represented_for_raw.get(current)

                if target_id is not None:
                    graph.add_edge(
                        graph.start_node_id,
                        target_id,
                    )
                    break

                queue.extend(raw_cfg.successors(current))

        self._merge_returns(graph)
        self._prune_unreachable(graph)

        return graph

    # ── indexes ──────────────────────────────────────────────────────────

    def _build_fn_index(self) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for node_id, data in self._G.nodes(data=True):
            if _clean(data.get("label", "")).upper() != "METHOD":
                continue
            external = _clean(data.get("IS_EXTERNAL", data.get("ISEXTERNAL", ""))).lower()
            if external == "true":
                continue
            name = _clean(data.get("NAME", ""))
            if name.startswith("<"):
                continue
            full_name = _clean(data.get("FULL_NAME", data.get("FULLNAME", "")))
            if full_name:
                index[full_name] = node_id
        return index

    def _extract_input_parameters(
        self,
        method: Any,
    ) -> list[tuple[str, str]]:
        return [
            (parameter.name or "_", parameter.type_full_name or "unknown")
            for parameter in method.parameters
        ]


    def _extract_output_parameters(
        self,
        method: MethodGraph,
    ) -> list[tuple[str, str]]:
        return_type = method.return_type

        if return_type is None:
            return []

        return [
            (
                "",
                return_type.type_full_name or "unknown",
            )
        ]
    
    def _prune_unreachable(
        self,
        graph: SemanticFlowGraph,
    ) -> None:
        adjacency: Dict[str, Set[str]] = defaultdict(set)

        for edge in graph.edge_list:
            adjacency[edge.source_id].add(edge.target_id)

        reachable: Set[str] = {graph.start_node_id}
        queue = deque([graph.start_node_id])

        while queue:
            source_id = queue.popleft()

            for target_id in adjacency.get(source_id, set()):
                if target_id not in reachable:
                    reachable.add(target_id)
                    queue.append(target_id)

        graph.nodes = {
            node_id: unit
            for node_id, unit in graph.nodes.items()
            if node_id in reachable
        }

        graph.edges = {
            key: edge
            for key, edge in graph.edges.items()
            if edge.source_id == graph.start_node_id
            or (
                edge.source_id in graph.nodes
                and edge.target_id in graph.nodes
            )
        }

        graph.return_node_ids &= set(graph.nodes)

    def _merge_returns(
        self,
        graph: SemanticFlowGraph,
    ) -> None:
        canonical_return_id = f"return@{graph.method_full_name}"

        return_ids = {
            node_id
            for node_id, unit in graph.nodes.items()
            if unit.kind is SemanticUnitKind.RETURN
        }

        if not return_ids:
            return

        graph.nodes[canonical_return_id] = SemanticUnit(
            node_id=canonical_return_id,
            kind=SemanticUnitKind.RETURN,
            code="RETURN",
            line="",
        )

        merged_edges: Dict[
            Tuple[str, str, str],
            SemanticEdge,
        ] = {}

        for edge in graph.edge_list:
            # return не должен вести дальше на технический RET
            if edge.source_id in return_ids:
                continue

            target_id = (
                canonical_return_id
                if edge.target_id in return_ids
                else edge.target_id
            )

            if edge.source_id == target_id:
                continue

            merged = SemanticEdge(
                source_id=edge.source_id,
                target_id=target_id,
                condition=edge.condition,
            )

            merged_edges[
                (merged.source_id, merged.target_id, merged.condition)
            ] = merged

        graph.edges = merged_edges

        # Удаляем все исходные RETURN / RET nodes.
        for node_id in return_ids:
            if node_id != canonical_return_id:
                graph.nodes.pop(node_id, None)

        graph.return_node_ids = {canonical_return_id}

    def _build_name_index(self) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = defaultdict(list)
        for node_id in self._fn_index.values():
            name = _node_attr(self._G, node_id, "NAME")
            if name:
                index[name].append(node_id)
        return dict(index)


# ── output ────────────────────────────────────────────────────────────────


def print_entrypoint_flow(
    result: EntrypointFlowResult,
    show_semantic: bool = True,
    show_external: bool = False,
) -> None:
    print("\n" + "═" * 76)
    print(f" entrypoint FLOW {result.entrypoint_name}")
    print(f" {result.entrypoint_full_name}")
    print(result.summary())
    print("─" * 76)

    for entry in result.sequence:
        method = entry.method_graph
        indent = " " * entry.depth
        graph = result.semantic_graphs.get(method.full_name)
        print(f" [{entry.call_index:>3}] {indent}{method.name} ({method.full_name})")
        if show_semantic and graph:
            for unit in graph.nodes.values():
                print(f" {indent}  {unit.kind.value.upper():12s} L{unit.line:>4} {unit.code}")
                if unit.defines:
                    print(f" {indent}    DEF: " + ", ".join(v.name for v in unit.defines))
                if unit.uses:
                    print(f" {indent}    USE: " + ", ".join(v.name for v in unit.uses))
                if unit.internal_callee_full_names:
                    print(f" {indent}    INTERNAL: " + ", ".join(unit.internal_callee_full_names))
            print(f" {indent}  EDGES: {len(graph.edges)}")
        print()

    if show_external and result.external_calls:
        print(" external calls:")
        for call in result.external_calls:
            print(f"  {call.caller_full_name} → {call.callee_full_name} L{call.line} {call.call_code}")
    print("═" * 76)

