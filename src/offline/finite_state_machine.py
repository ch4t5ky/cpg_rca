from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.offline.flow import (
    EntrypointFlowResult,
    SemanticFlowGraph,
    SemanticUnit,
    SemanticUnitKind,
)
from src.offline.log import LogTemplateWithMethod


@dataclass(frozen=True)
class ExecutionSegment:
    """A state: code between a preceding boundary and the next observable one."""

    id: str
    entrypoint_full_name: str
    method_full_name: str
    direct_methods: Tuple[str, ...]
    external_calls: Tuple[str, ...]
    conditions: Tuple[str, ...]
    previous_log_call_node_id: Optional[str]
    next_log_call_node_id: Optional[str]
    is_start: bool
    is_terminal: bool
    kind: str  # START_SEGMENT | BETWEEN_LOGS | RETURN_SEGMENT | INCOMPLETE_SEGMENT

    @property
    def label(self) -> str:
        methods = self.direct_methods or self.external_calls
        if not methods:
            return self.kind
        short = [value.rsplit(".", 1)[-1] for value in methods[:3]]
        return ", ".join(short) + (", ..." if len(methods) > 3 else "")


@dataclass(frozen=True)
class LogTransition:
    """One logger call connecting two observable intervals."""

    id: str
    source_segment_id: str
    target_segment_id: str
    template: str
    log_call_node_id: str
    method_full_name: str
    method_node_id: str
    conditions: Tuple[str, ...] = ()
    static_score: int = 0

    @property
    def source(self) -> str:
        return self.source_segment_id

    @property
    def target(self) -> str:
        return self.target_segment_id


LogEdge = LogTransition


@dataclass
class StaticLogFSM:
    entrypoint_node_id: str
    entrypoint_name: str
    entrypoint_full_name: str
    states: Dict[str, ExecutionSegment] = field(default_factory=dict)
    transitions: List[LogTransition] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def edges(self) -> List[LogTransition]:
        return self.transitions

    @property
    def start_states(self) -> set[str]:
        return {state_id for state_id, state in self.states.items() if state.is_start}

    @property
    def terminals(self) -> set[str]:
        return {
            state_id for state_id, state in self.states.items() if state.is_terminal
        }

    @property
    def edgeset(self) -> set[Tuple[str, str]]:
        return {
            (edge.source_segment_id, edge.target_segment_id)
            for edge in self.transitions
        }

    def outgoing(self, state_id: str) -> List[LogTransition]:
        return [edge for edge in self.transitions if edge.source_segment_id == state_id]

    def incoming(self, state_id: str) -> List[LogTransition]:
        return [edge for edge in self.transitions if edge.target_segment_id == state_id]

    def format_state(self, state_id: str) -> str:
        state = self.states.get(state_id)
        return state_id if state is None else f"{state.kind}: {state.label}"

    def summary(self) -> str:
        return (
            f"entrypoint={self.entrypoint_full_name}, "
            f"segments={len(self.states)}, transitions={len(self.transitions)}, "
            f"starts={len(self.start_states)}, terminals={len(self.terminals)}, "
            f"warnings={len(self.warnings)}"
        )


def _ordered_unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _is_technical_call(callee_full_name: str) -> bool:
    """Exclude CPG implementation details from labels, never from CFG flow."""
    value = (callee_full_name or "").strip()
    return (
        not value
        or value.startswith("<operator>.")
        or value.startswith("ANY.")
        or ".<ReturnType>.<unknown>" in value
        or value.endswith(".<init>")
        or value in {"append", "len", "make"}
    )


@dataclass
class _SegmentFacts:
    direct_methods: List[str] = field(default_factory=list)
    external_calls: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)

    def add_condition(self, value: str) -> bool:
        value = (value or "").strip()
        if value and value not in self.conditions:
            self.conditions.append(value)
            return True
        return False

    def add_direct_method(self, full_name: str) -> bool:
        full_name = (full_name or "").strip()
        if full_name and full_name not in self.direct_methods:
            self.direct_methods.append(full_name)
            return True
        return False

    def add_unit(self, unit: SemanticUnit, known_methods: Set[str]) -> bool:
        changed = False
        for callee in unit.callee_full_names:
            if _is_technical_call(callee):
                continue
            target = (
                self.direct_methods if callee in known_methods else self.external_calls
            )
            if callee not in target:
                target.append(callee)
                changed = True
        return changed

    def merge(self, other: "_SegmentFacts") -> bool:
        changed = False
        for own, incoming in (
            (self.direct_methods, other.direct_methods),
            (self.external_calls, other.external_calls),
            (self.conditions, other.conditions),
        ):
            for value in incoming:
                if value not in own:
                    own.append(value)
                    changed = True
        return changed

    def copy(self) -> "_SegmentFacts":
        return _SegmentFacts(
            direct_methods=list(self.direct_methods),
            external_calls=list(self.external_calls),
            conditions=list(self.conditions),
        )


@dataclass(frozen=True)
class _LogMarker:
    template: str
    call_node_id: str
    method_node_id: str
    static_score: int


@dataclass
class _Outcome:
    """One interval outcome, aggregated over all paths reaching this boundary."""

    next_marker: Optional[_LogMarker]
    kind: str
    facts: _SegmentFacts = field(default_factory=_SegmentFacts)



class LogFlowExtractor:
    """Build a boundary-minimized FSM from the semantic graphs of a flow.

    Interprocedural contract
    ────────────────────────
    Each SemanticFlowGraph is analysed in its own context. A logger CALL is a
    global observable boundary identified by its physical CPG CALL node id, so
    segments and transitions connect across method boundaries naturally.

    * Downward calls use per-method summaries: ``_within(method, seeds)``
      walks one graph, crossing into callees through their own summaries, and
      reports callee-internal logger CALLs as outcomes plus an abstract RETURN
      outcome when the callee can finish. Summaries are memoized per
      (method, seed set); re-entry (recursion) is cut off and reported.
    * A callee that finishes continues the caller *at the local call site
      successors only* — never at another call site's continuation.
    * When the walk of a non-root method reaches its return, the abstract
      RETURN is resolved at every known call site of that method (union over
      callers). This is a sound over-approximation: every emitted sequence is
      physically possible, but segments that share the same observable
      boundary (prev log, next log, kind) are merged per project semantics.
    * Top-level (root) returns become RETURN_SEGMENT terminal states.
    * Abstract returns that cannot be resolved (no call site in the flow,
      recursion cut off, depth limit) become INCOMPLETE_SEGMENT terminals and
      are reported in the diagnostics — the normal flow is never dropped.
    """

    _ABSTRACT_RETURN = "RETURN"  # internal kind: method may return to caller

    def __init__(
        self,
        templates: Iterable[LogTemplateWithMethod],
        max_call_depth: int = 8,
    ) -> None:
        self.templates_by_call: Dict[str, LogTemplateWithMethod] = {}
        for template in templates:
            call_id = str(template.call_node_id)
            previous = self.templates_by_call.get(call_id)
            if previous is None or template.static_count > previous.static_count:
                self.templates_by_call[call_id] = template
        self.max_call_depth = max_call_depth

    # ── public ──────────────────────────────────────────────────────────

    def extract(self, flow: EntrypointFlowResult) -> StaticLogFSM:
        fsm = StaticLogFSM(
            entrypoint_node_id=flow.entrypoint_node_id,
            entrypoint_name=flow.entrypoint_name,
            entrypoint_full_name=flow.entrypoint_full_name,
        )
        for note in getattr(flow, "entrypoint_resolution", []):
            fsm.warnings.append(f"entrypoint: {note}")

        diagnostics: Dict[str, object] = {
            "entrypoint_full_name": flow.entrypoint_full_name,
            "catalog_templates": len(self.templates_by_call),
        }
        fsm.diagnostics = diagnostics

        if not self.templates_by_call:
            fsm.warnings.append("Log template catalog is empty.")
            return fsm

        graphs = flow.semantic_graphs
        root_name = flow.entrypoint_full_name
        graph = graphs.get(root_name)
        if graph is None:
            fsm.warnings.append(
                f"Entrypoint semantic graph is missing for '{root_name}'. "
                f"Available methods: {sorted(graphs)}"
            )
            diagnostics["analyzed_methods"] = []
            return fsm

        matched_ids = {
            str(call_id)
            for g in graphs.values()
            for unit in g.nodes.values()
            for call_id in unit.call_node_ids
            if str(call_id) in self.templates_by_call
        }

        explorer = _Explorer(self, flow, set(graphs))
        outcomes_by_source = explorer.run(root_name)

        diagnostics["analyzed_methods"] = sorted(explorer.touched_methods)
        diagnostics["matched_call_ids"] = sorted(
            matched_ids & explorer.all_call_ids_in_touched
        )
        reachable = {
            outcome.next_marker.call_node_id
            for outcomes in outcomes_by_source.values()
            for outcome in outcomes.values()
            if outcome.next_marker is not None
        }
        diagnostics["reachable_log_boundaries"] = sorted(reachable)
        if explorer.limited:
            diagnostics["interprocedural_limits"] = list(explorer.limited)
            fsm.warnings.append(
                "Interprocedural expansion limited: " + "; ".join(explorer.limited)
            )

        self._assemble(fsm, flow.entrypoint_full_name, outcomes_by_source)

        if not fsm.transitions:
            fsm.warnings.append(
                self._explain_empty(flow, matched_ids, explorer, reachable)
            )
        return fsm

    def _explain_empty(
        self,
        flow: EntrypointFlowResult,
        matched_ids: Set[str],
        explorer: "_Explorer",
        reachable: Set[str],
    ) -> str:
        root_name = flow.entrypoint_full_name
        if not matched_ids:
            return (
                f"No logger CALL IDs from the catalog "
                f"({len(self.templates_by_call)} templates) match any semantic "
                f"unit call_node_id in the {len(flow.semantic_graphs)} flow methods."
            )
        analyzed = matched_ids & explorer.all_call_ids_in_touched
        if not analyzed:
            return (
                f"Catalog logger CALL IDs match only in methods that were not "
                f"analyzed from entrypoint '{root_name}'. "
                f"Matched elsewhere: {sorted(matched_ids)}. "
                f"Analyzed methods: {sorted(explorer.touched_methods)}."
            )
        if not reachable:
            return (
                f"Matched logger CALL IDs {sorted(analyzed)} exist in the analyzed "
                f"methods but are unreachable from START of '{root_name}'."
            )
        return (
            f"Segments were built around reachable logger CALL IDs "
            f"{sorted(reachable)} but no transitions were emitted."
        )

    def _marker_for_unit(self, unit: Optional[SemanticUnit]) -> Optional[_LogMarker]:
        if unit is None:
            return None
        candidates = [
            self.templates_by_call[str(call_id)]
            for call_id in unit.call_node_ids
            if str(call_id) in self.templates_by_call
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda item: item.static_count)
        return _LogMarker(
            template=best.raw_template,
            call_node_id=str(best.call_node_id),
            method_node_id=str(best.method_node_id),
            static_score=best.static_count,
        )

    # ── FSM assembly (state = observable boundary) ──────────────────────

    def _assemble(
        self,
        fsm: StaticLogFSM,
        entrypoint_full_name: str,
        outcomes_by_source: Dict[Optional[str], Dict[Tuple[str, str], "_Outcome"]],
    ) -> None:
        state_ids: Dict[Tuple[Optional[str], str, str], str] = {}
        state_counter = 0

        def resolve_state(source_log_id: Optional[str], outcome: _Outcome) -> str:
            # Observable boundaries only. Facts/guards are deliberately absent:
            # they are annotations, not state identity.
            target = outcome.next_marker.call_node_id if outcome.next_marker else ""
            key = (source_log_id, target, outcome.kind)
            nonlocal state_counter
            state_id = state_ids.get(key)
            if state_id is None:
                state_id = f"segment:{state_counter}"
                state_counter += 1
                state_ids[key] = state_id
                fsm.states[state_id] = ExecutionSegment(
                    id=state_id,
                    entrypoint_full_name=entrypoint_full_name,
                    method_full_name=entrypoint_full_name,
                    direct_methods=_ordered_unique(outcome.facts.direct_methods),
                    external_calls=_ordered_unique(outcome.facts.external_calls),
                    conditions=_ordered_unique(outcome.facts.conditions),
                    previous_log_call_node_id=source_log_id,
                    next_log_call_node_id=target or None,
                    is_start=source_log_id is None,
                    is_terminal=outcome.kind
                    in {"RETURN_SEGMENT", "INCOMPLETE_SEGMENT"},
                    kind=outcome.kind,
                )
            else:
                # Same observable state reached via a different CFG route:
                # retain all descriptive facts without creating another node.
                old = fsm.states[state_id]
                merged_direct = _ordered_unique(
                    (*old.direct_methods, *outcome.facts.direct_methods)
                )
                merged_external = _ordered_unique(
                    (*old.external_calls, *outcome.facts.external_calls)
                )
                merged_conditions = _ordered_unique(
                    (*old.conditions, *outcome.facts.conditions)
                )
                if (merged_direct, merged_external, merged_conditions) != (
                    old.direct_methods,
                    old.external_calls,
                    old.conditions,
                ):
                    fsm.states[state_id] = ExecutionSegment(
                        id=old.id,
                        entrypoint_full_name=old.entrypoint_full_name,
                        method_full_name=old.method_full_name,
                        direct_methods=merged_direct,
                        external_calls=merged_external,
                        conditions=merged_conditions,
                        previous_log_call_node_id=old.previous_log_call_node_id,
                        next_log_call_node_id=old.next_log_call_node_id,
                        is_start=old.is_start,
                        is_terminal=old.is_terminal,
                        kind=old.kind,
                    )
            return state_id

        states_by_source: Dict[Optional[str], List[str]] = defaultdict(list)
        for source_log_id, outcomes in outcomes_by_source.items():
            for outcome in outcomes.values():
                states_by_source[source_log_id].append(
                    resolve_state(source_log_id, outcome)
                )

        seen_transitions: Set[Tuple[str, str, str]] = set()
        transition_counter = 0
        for source_log_id, source_state_ids in states_by_source.items():
            if source_log_id is None:
                continue
            source_template = self.templates_by_call.get(source_log_id)
            if source_template is None:
                continue
            targets = states_by_source.get(source_log_id, [])
            # A transition is emitted from every interval that *ends* at this
            # physical logger call to every interval that begins after it.
            incoming = [
                state_id
                for state_id, state in fsm.states.items()
                if state.next_log_call_node_id == source_log_id
            ]
            for from_state_id in incoming:
                for to_state_id in targets:
                    key = (from_state_id, to_state_id, source_log_id)
                    if key in seen_transitions:
                        continue
                    seen_transitions.add(key)
                    fsm.transitions.append(
                        LogTransition(
                            id=f"log_transition:{transition_counter}",
                            source_segment_id=from_state_id,
                            target_segment_id=to_state_id,
                            template=source_template.raw_template,
                            log_call_node_id=source_log_id,
                            method_full_name=entrypoint_full_name,
                            method_node_id=str(source_template.method_node_id),
                            conditions=(),
                            static_score=source_template.static_count,
                        )
                    )
                    transition_counter += 1


class _Explorer:
    """Bounded interprocedural exploration of the flow semantic graphs."""

    def __init__(
        self,
        extractor: LogFlowExtractor,
        flow: EntrypointFlowResult,
        known_methods: Set[str],
    ) -> None:
        self._x = extractor
        self._flow = flow
        self._known = known_methods
        self._graphs = flow.semantic_graphs
        self._successors: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        self._within_memo: Dict[Tuple[str, str], Dict[Tuple[str, str], _Outcome]] = {}
        self._within_progress: Set[Tuple[str, str]] = set()
        self._resolve_memo: Dict[Tuple[str, str], Dict[Tuple[str, str], _Outcome]] = {}
        self._resolve_progress: Set[Tuple[str, str]] = set()
        self._return_sites: Optional[Dict[str, List[Tuple[str, str]]]] = None
        self._resolve_method_stack: List[str] = []
        self.limited: List[str] = []
        self.touched_methods: Set[str] = set()
        self.all_call_ids_in_touched: Set[str] = set()

    # ── driver ──────────────────────────────────────────────────────────

    def run(
        self, root_name: str
    ) -> Dict[Optional[str], Dict[Tuple[str, str], "_Outcome"]]:
        root_graph = self._graphs[root_name]
        outcomes_by_source: Dict[Optional[str], Dict[Tuple[str, str], _Outcome]] = {
            None: self._resolve_source(
                root_name,
                "",
                root_graph.start_node_id,
                depth=0,
                is_root=True,
                source_is_log=False,
            )
        }
        for method_name, graph in self._graphs.items():
            for node_id, unit in graph.nodes.items():
                marker = self._x._marker_for_unit(unit)
                if marker is None:
                    continue
                outcomes_by_source[marker.call_node_id] = self._resolve_source(
                    method_name,
                    marker.call_node_id,
                    node_id,
                    depth=0,
                    is_root=method_name == root_name,
                    source_is_log=True,
                )
        return outcomes_by_source

    def _resolve_source(
        self,
        method_name: str,
        source_key: str,
        source_node_id: str,
        depth: int,
        is_root: bool,
        source_is_log: bool,
    ) -> Dict[Tuple[str, str], "_Outcome"]:
        """Outcomes for one boundary source; abstract returns are resolved."""
        memo_key = (method_name, source_key)
        if memo_key in self._resolve_memo:
            return self._resolve_memo[memo_key]
        if memo_key in self._resolve_progress:
            self.limited.append(
                f"cycle while resolving returns of '{method_name}' "
                f"(source '{source_key or 'START'}')"
            )
            return self._terminal_outcome("INCOMPLETE_SEGMENT")
        self._resolve_progress.add(memo_key)
        self._resolve_method_stack.append(method_name)
        try:
            within = self._within(
                method_name, source_node_id, depth, source_is_log
            )
            resolved: Dict[Tuple[str, str], _Outcome] = {}
            for outcome in within.values():
                if outcome.kind != LogFlowExtractor._ABSTRACT_RETURN:
                    self._merge_outcome(resolved, outcome)
                    continue
                facts = outcome.facts.copy()
                if is_root:
                    self._merge_outcome(
                        resolved,
                        _Outcome(next_marker=None, kind="RETURN_SEGMENT", facts=facts),
                    )
                    continue
                sites = self._return_sites_map().get(method_name, [])
                if not sites:
                    self._merge_outcome(
                        resolved,
                        _Outcome(
                            next_marker=None, kind="INCOMPLETE_SEGMENT", facts=facts
                        ),
                    )
                    self.limited.append(
                        f"no call site of '{method_name}' inside the flow; "
                        f"its return was modelled as an incomplete exit"
                    )
                    continue
                if depth >= self._x.max_call_depth:
                    self._merge_outcome(
                        resolved,
                        _Outcome(
                            next_marker=None, kind="INCOMPLETE_SEGMENT", facts=facts
                        ),
                    )
                    self.limited.append(
                        f"call depth limit {self._x.max_call_depth} reached at "
                        f"returns of '{method_name}'"
                    )
                    continue
                for caller_name, call_node_id in sites:
                    caller_successors = self._successors_map(caller_name).get(
                        call_node_id, []
                    )
                    for target_id, condition in caller_successors:
                        continuation = self._resolve_source(
                            caller_name,
                            f"after:{method_name}:{call_node_id}:{target_id}",
                            target_id,
                            depth + 1,
                            is_root=False,
                            source_is_log=False,
                        )
                        for sub in continuation.values():
                            merged = _Outcome(
                                next_marker=sub.next_marker,
                                kind=sub.kind,
                                facts=facts.copy(),
                            )
                            merged.facts.add_condition(condition)
                            merged.facts.merge(sub.facts)
                            self._merge_outcome(resolved, merged)
            self._resolve_memo[memo_key] = resolved
            return resolved
        finally:
            self._resolve_progress.discard(memo_key)
            self._resolve_method_stack.pop()

    # ── within one method (downward summaries) ──────────────────────────

    def _within(
        self,
        method_name: str,
        source_node_id: str,
        depth: int,
        source_is_log: bool,
    ) -> Dict[Tuple[str, str], _Outcome]:
        graph = self._graphs[method_name]
        memo_key = (method_name, f"{source_node_id}|{'log' if source_is_log else 'start'}")
        if memo_key in self._within_memo:
            return self._within_memo[memo_key]
        if memo_key in self._within_progress:
            # Downward recursion: approximate the recursive call as a call
            # that simply returns.
            self.limited.append(f"recursive call into '{method_name}' was cut off")
            return {
                ("", LogFlowExtractor._ABSTRACT_RETURN): _Outcome(
                    next_marker=None, kind=LogFlowExtractor._ABSTRACT_RETURN
                )
            }
        self._within_progress.add(memo_key)
        self.touched_methods.add(method_name)
        try:
            successors = self._successors_map(method_name)
            outcomes = self._walk(
                method_name, graph, successors, source_node_id, depth, source_is_log
            )
            self._within_memo[memo_key] = outcomes
            return outcomes
        finally:
            self._within_progress.discard(memo_key)

    def _walk(
        self,
        method_name: str,
        graph: SemanticFlowGraph,
        successors: Dict[str, List[Tuple[str, str]]],
        source_node_id: str,
        depth: int,
        source_is_log: bool,
    ) -> Dict[Tuple[str, str], "_Outcome"]:
        """Fixpoint walk over one graph from one boundary source.

        ``facts_at_node`` is merged monotonically and a node is reconsidered
        only when that metadata changes. Therefore loops are finite and no CFG
        execution paths are enumerated.
        """
        facts_at_node: Dict[str, _SegmentFacts] = {}
        queue: deque[str] = deque()

        initial_targets = (
            [target for target, _condition in successors.get(source_node_id, [])]
            if source_is_log
            else [source_node_id]
        )
        for target_id in initial_targets:
            facts = _SegmentFacts()
            facts_at_node[target_id] = facts
            queue.append(target_id)

        outcomes: Dict[Tuple[str, str], _Outcome] = {}

        def add_outcome(
            marker: Optional[_LogMarker], kind: str, facts: _SegmentFacts
        ) -> None:
            next_id = marker.call_node_id if marker else ""
            key = (next_id, kind)
            outcome = outcomes.get(key)
            if outcome is None:
                outcomes[key] = _Outcome(
                    next_marker=marker, kind=kind, facts=facts.copy()
                )
            else:
                outcome.facts.merge(facts)

        def propagate(local: _SegmentFacts, outgoing: List[Tuple[str, str]]) -> None:
            for target_id, condition in outgoing:
                propagated = local.copy()
                propagated.add_condition(condition)
                existing = facts_at_node.get(target_id)
                if existing is None:
                    facts_at_node[target_id] = propagated
                    queue.append(target_id)
                elif existing.merge(propagated):
                    queue.append(target_id)

        processed: Dict[
            str, Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
        ] = {}
        while queue:
            node_id = queue.popleft()
            facts = facts_at_node[node_id]
            fingerprint = (
                tuple(facts.direct_methods),
                tuple(facts.external_calls),
                tuple(facts.conditions),
            )
            if processed.get(node_id) == fingerprint:
                continue
            processed[node_id] = fingerprint

            unit = graph.nodes.get(node_id)
            marker = self._x._marker_for_unit(unit)
            # Every logger CALL is a boundary. (A log source is seeded from
            # its successors, so it can only reappear here through a loop
            # back-edge — which is a new occurrence of that same call.)
            if marker is not None:
                add_outcome(marker, "START_SEGMENT", facts)
                continue

            local = facts.copy()
            if unit is not None:
                local.add_unit(unit, self._known)

            if unit is not None and unit.kind is SemanticUnitKind.RETURN:
                add_outcome(None, LogFlowExtractor._ABSTRACT_RETURN, local)
                continue

            outgoing = successors.get(node_id, [])
            if not outgoing:
                kind = (
                    LogFlowExtractor._ABSTRACT_RETURN
                    if node_id in graph.return_node_ids
                    else "INCOMPLETE_SEGMENT"
                )
                add_outcome(None, kind, local)
                continue

            internal = (
                [
                    callee
                    for callee in unit.internal_callee_full_names
                    if callee in self._graphs
                ]
                if unit is not None
                else []
            )
            if internal and depth < self._x.max_call_depth:
                for callee in internal:
                    if callee in self._resolve_method_stack:
                        # Recursive re-entry: approximate the call as one that
                        # simply returns, and report the cut-off explicitly.
                        self.limited.append(
                            f"recursive call into '{callee}' was cut off"
                        )
                        merged = local.copy()
                        merged.add_direct_method(callee)
                        propagate(merged, outgoing)
                        continue
                    callee_graph = self._graphs[callee]
                    sub = self._within(
                        callee,
                        callee_graph.start_node_id,
                        depth + 1,
                        source_is_log=False,
                    )
                    for sub_outcome in sub.values():
                        merged = local.copy()
                        merged.add_direct_method(callee)
                        merged.merge(sub_outcome.facts)
                        if sub_outcome.next_marker is not None:
                            add_outcome(
                                sub_outcome.next_marker, "START_SEGMENT", merged
                            )
                        else:
                            # Callee finished (or was cut off): continue the
                            # caller strictly at this call site's successors.
                            propagate(merged, outgoing)
                continue

            propagate(local, outgoing)

        if not outcomes:
            add_outcome(None, "INCOMPLETE_SEGMENT", _SegmentFacts())
        return outcomes

    # ── helpers ─────────────────────────────────────────────────────────

    def _terminal_outcome(self, kind: str) -> Dict[Tuple[str, str], "_Outcome"]:
        return {
            ("", kind): _Outcome(next_marker=None, kind=kind, facts=_SegmentFacts())
        }

    @staticmethod
    def _merge_outcome(
        target: Dict[Tuple[str, str], "_Outcome"], outcome: "_Outcome"
    ) -> None:
        key = (
            outcome.next_marker.call_node_id if outcome.next_marker else "",
            outcome.kind,
        )
        existing = target.get(key)
        if existing is None:
            target[key] = _Outcome(
                next_marker=outcome.next_marker,
                kind=outcome.kind,
                facts=outcome.facts.copy(),
            )
        else:
            existing.facts.merge(outcome.facts)

    def _successors_map(
        self, method_name: str
    ) -> Dict[str, List[Tuple[str, str]]]:
        cached = self._successors.get(method_name)
        if cached is None:
            by_source: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
            for edge in self._graphs[method_name].edge_list:
                by_source[edge.source_id].append((edge.target_id, edge.condition))
            cached = dict(by_source)
            self._successors[method_name] = cached
            graph = self._graphs[method_name]
            self.all_call_ids_in_touched.update(
                str(call_id)
                for unit in graph.nodes.values()
                for call_id in unit.call_node_ids
            )
        return cached

    def _return_sites_map(self) -> Dict[str, List[Tuple[str, str]]]:
        if self._return_sites is not None:
            return self._return_sites
        sites: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for method_name, graph in self._graphs.items():
            for node_id, unit in graph.nodes.items():
                for callee in unit.internal_callee_full_names:
                    if callee in self._graphs and callee != method_name:
                        sites[callee].append((method_name, node_id))
        self._return_sites = dict(sites)
        return self._return_sites


def printlogfsm(fsm: StaticLogFSM) -> None:
    print(fsm.summary())
    for state_id, state in fsm.states.items():
        print(
            f"STATE {state_id} {state.kind} method={state.method_full_name} "
            f"methods={list(state.direct_methods)} external={list(state.external_calls)}"
        )
    for transition in fsm.transitions:
        print(
            f"{transition.source_segment_id} --[{transition.template}]--> "
            f"{transition.target_segment_id}"
        )
