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
    """Build a boundary-minimized FSM directly from SemanticFlowGraph."""

    def __init__(self, templates: Iterable[LogTemplateWithMethod]) -> None:
        self.templates_by_call: Dict[str, LogTemplateWithMethod] = {}
        for template in templates:
            call_id = str(template.call_node_id)
            previous = self.templates_by_call.get(call_id)
            if previous is None or template.static_count > previous.static_count:
                self.templates_by_call[call_id] = template

    def extract(self, flow: EntrypointFlowResult) -> StaticLogFSM:
        fsm = StaticLogFSM(
            entrypoint_node_id=flow.entrypoint_node_id,
            entrypoint_name=flow.entrypoint_name,
            entrypoint_full_name=flow.entrypoint_full_name,
        )
        if not self.templates_by_call:
            fsm.warnings.append("Log template catalog is empty.")
            return fsm

        graph = flow.semantic_graphs.get(flow.entrypoint_full_name)
        if graph is None:
            fsm.warnings.append("Entrypoint has no semantic graph.")
            return fsm

        self._build_from_graph(
            flow.entrypoint_full_name, graph, set(flow.semantic_graphs), fsm
        )
        if not fsm.transitions:
            fsm.warnings.append("No logger CALL nodes matched the template catalog.")
        return fsm

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

    def _build_from_graph(
        self,
        entrypoint_full_name: str,
        graph: SemanticFlowGraph,
        known_methods: Set[str],
        fsm: StaticLogFSM,
    ) -> None:
        successors: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for edge in graph.edge_list:
            successors[edge.source_id].append((edge.target_id, edge.condition))

        # Each source is analysed once: method START or a physical logger CALL.
        source_nodes: Dict[Optional[str], str] = {None: graph.start_node_id}
        for node_id, unit in graph.nodes.items():
            marker = self._marker_for_unit(unit)
            if marker is not None:
                source_nodes[marker.call_node_id] = node_id

        outcomes_by_source: Dict[Optional[str], Dict[Tuple[str, str], _Outcome]] = {}
        for source_log_id, source_node_id in source_nodes.items():
            outcomes_by_source[source_log_id] = self._discover_outcomes(
                graph=graph,
                successors=successors,
                source_node_id=source_node_id,
                source_is_log=source_log_id is not None,
                known_methods=known_methods,
            )

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

    def _discover_outcomes(
        self,
        graph: SemanticFlowGraph,
        successors: Dict[str, List[Tuple[str, str]]],
        source_node_id: str,
        source_is_log: bool,
        known_methods: Set[str],
    ) -> Dict[Tuple[str, str], _Outcome]:
        """Data-flow walk from one source boundary to next boundaries.

        ``facts_at_node`` is merged monotonically and a node is reconsidered
        only when that metadata changes. Therefore loops are finite and no CFG
        execution paths are enumerated.
        """
        facts_at_node: Dict[str, _SegmentFacts] = {}
        queue: deque[str] = deque()

        initial_targets = (
            successors.get(source_node_id, [])
            if source_is_log
            else [(source_node_id, "")]
        )
        for target_id, condition in initial_targets:
            facts = _SegmentFacts()
            facts.add_condition(condition)
            current = facts_at_node.get(target_id)
            if current is None:
                facts_at_node[target_id] = facts
                queue.append(target_id)
            elif current.merge(facts):
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
            marker = self._marker_for_unit(unit)
            # The source log was skipped above; every other log is a boundary.
            if marker is not None:
                add_outcome(marker, "START_SEGMENT", facts)
                continue

            local = facts.copy()
            if unit is not None:
                local.add_unit(unit, known_methods)
            if unit is not None and unit.kind is SemanticUnitKind.RETURN:
                add_outcome(None, "RETURN_SEGMENT", local)
                continue

            outgoing = successors.get(node_id, [])
            if not outgoing:
                kind = (
                    "RETURN_SEGMENT"
                    if node_id in graph.return_node_ids
                    else "INCOMPLETE_SEGMENT"
                )
                add_outcome(None, kind, local)
                continue

            for target_id, condition in outgoing:
                propagated = local.copy()
                propagated.add_condition(condition)
                existing = facts_at_node.get(target_id)
                if existing is None:
                    facts_at_node[target_id] = propagated
                    queue.append(target_id)
                elif existing.merge(propagated):
                    queue.append(target_id)

        if not outcomes:
            add_outcome(None, "INCOMPLETE_SEGMENT", _SegmentFacts())
        return outcomes


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
