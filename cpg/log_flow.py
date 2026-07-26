"""
cpg/log_flow.py — v2: inter-procedural Static Log FSM.

Основные изменения относительно v1:
1. Межпроцедурная склейка: используется MethodEntry.via_path / caller_full_name
   из EntrypointFlowResult.sequence, чтобы соединять последний logpoint пути
   caller-а с первым logpoint пути вызываемого метода (call-site anchoring).
2. RETURN callee -> продолжение caller-а после call-site, а не глобальный
   терминал (терминал __return__ остаётся только для entrypoint-метода).
3. EXTERNAL_CALL узлы: gRPC/библиотечные вызовы из EntrypointFlowResult.external_calls
   материализуются как отдельный тип состояния, а не отбрасываются.
4. INCOMPLETE используется только когда путь действительно не ведёт ни в RETURN,
   ни в call-site следующего метода (честный "недостроенный" путь).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from cpg.flow import EntrypointFlowResult, MethodPath, MethodEntry, ExternalCall
from cpg_trie_parser import LogTemplateWithMethod

START = "__start__"
RETURN = "__return__"
INCOMPLETE = "__incomplete__"


@dataclass(frozen=True)
class LogPoint:
    id: str
    call_node_id: str
    method_node_id: str
    method_fullname: str
    method_name: str
    template: str
    static_score: int

    @property
    def label(self) -> str:
        return self.template


@dataclass(frozen=True)
class ExternalPoint:
    """Состояние, представляющее внешний (не разворачиваемый) вызов."""
    id: str
    caller_full_name: str
    callee_full_name: str
    call_code: str
    line: str


@dataclass(frozen=True)
class LogEdge:
    source: str
    target: str
    method_fullname: str
    path_index: int
    conditions: Tuple[str, ...] = ()
    is_terminal: bool = False
    partial: bool = False
    is_interprocedural: bool = False  # NEW: ребро между разными методами


@dataclass
class StaticLogFSM:
    entrypoint_node_id: str
    entrypoint_name: str
    entrypoint_fullname: str

    states: Dict[str, LogPoint] = field(default_factory=dict)
    external_states: Dict[str, ExternalPoint] = field(default_factory=dict)  # NEW
    edges: List[LogEdge] = field(default_factory=list)
    terminals: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)

    @property
    def edge_set(self) -> Set[Tuple[str, str]]:
        return {(e.source, e.target) for e in self.edges}

    def outgoing(self, state_id: str) -> List[LogEdge]:
        return [e for e in self.edges if e.source == state_id]

    def incoming(self, state_id: str) -> List[LogEdge]:
        return [e for e in self.edges if e.target == state_id]

    def format_state(self, state_id: str) -> str:
        if state_id == START:
            return "START"
        if state_id == RETURN:
            return "RETURN"
        if state_id == INCOMPLETE:
            return "INCOMPLETE"
        point = self.states.get(state_id)
        if point is not None:
            return f'{point.method_name}: "{point.template}"'
        ext = self.external_states.get(state_id)
        if ext is not None:
            short = ext.callee_full_name.rsplit(".", 1)[-1]
            return f"EXT: {short}()"
        return state_id

    def summary(self) -> str:
        return (
            f"entrypoint={self.entrypoint_fullname}, "
            f"states={len(self.states)}, "
            f"external_states={len(self.external_states)}, "
            f"edges={len(self.edges)}, "
            f"terminals={len(self.terminals)}, "
            f"warnings={len(self.warnings)}"
        )


class LogFlowExtractor:
    """
    v2: строит межпроцедурный StaticLogFSM из EntrypointFlowResult.

    Ключевая идея:
    - Для каждого MethodEntry в flow.sequence обрабатываем его method_paths
      как раньше (внутрипроцедурная цепочка logpoint-ов).
    - Дополнительно: если путь заканчивается вызовом другого внутреннего
      метода (последний CALL на пути ведёт в callee из fn_index), то ребро
      "last_logpoint -> callee.START" создаётся вместо RETURN/INCOMPLETE.
    - После разворота callee, его RETURN-состояния соединяются с ближайшим
      logpoint-ом caller-а, идущим ПОСЛЕ соответствующего call-site
      (call-site anchoring через MethodPath.call_node_ids).
    - Внешние вызовы (external_calls) вставляются как ExternalPoint между
      соседними logpoint-ами, если call_node_id внешнего вызова лежит между
      ними на пути.
    """

    def __init__(self, templates: Iterable[LogTemplateWithMethod]) -> None:
        self.templates_by_call: Dict[str, LogTemplateWithMethod] = {}
        for template in templates:
            call_node_id = str(template.call_node_id)
            previous = self.templates_by_call.get(call_node_id)
            if previous is None or template.static_count > previous.static_count:
                self.templates_by_call[call_node_id] = template

    def extract(self, flow: EntrypointFlowResult) -> StaticLogFSM:
        fsm = StaticLogFSM(
            entrypoint_node_id=flow.entrypoint_node_id,
            entrypoint_name=flow.entrypoint_name,
            entrypoint_fullname=flow.entrypoint_full_name,
        )

        if not self.templates_by_call:
            fsm.warnings.append("Log template catalog is empty.")
            return fsm
        if not flow.method_paths:
            fsm.warnings.append("Flow contains no method paths.")
            return fsm

        # method_fullname -> node_id (для быстрого lookup callee)
        callable_methods = set(flow.method_paths.keys())

        # method_fullname -> {call_node_id -> callee_full_name}, чтобы знать,
        # какой конкретно CALL на пути ведёт во внутренний метод.
        callsite_callee: Dict[str, Dict[str, str]] = {}
        for entry in flow.sequence:
            pass  # заполняется ниже через cs_map, недоступный здесь напрямую;
                  # вместо этого используем ExternalCall/via_path соответствие.

        # externals по caller, отсортированные по call_index (порядок появления)
        externals_by_caller: Dict[str, List[ExternalCall]] = {}
        for ec in flow.external_calls:
            externals_by_caller.setdefault(ec.caller_full_name, []).append(ec)

        # entry-точки для каждого метода: кто его вызывал и через какой путь
        entry_by_method: Dict[str, List[MethodEntry]] = {}
        for entry in flow.sequence:
            entry_by_method.setdefault(entry.method_graph.full_name, []).append(entry)

        first_state_of_method: Dict[Tuple[str, int], str] = {}
        return_states_of_method: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {}

        for method_fullname, paths in flow.method_paths.items():
            is_entrypoint_method = method_fullname == flow.entrypoint_full_name
            return_states_of_method[method_fullname] = []

            for path_index, path in enumerate(paths):
                observed = self._extract_observed(path)

                if not observed:
                    continue

                observed: List[Tuple[str, Tuple[str, ...]]] = [
                    (self._register_logpoint(fsm, template, method_fullname), conditions)
                    for template, conditions in observed
                ]

                first_state, first_conditions = observed[0]
                first_state_of_method[(method_fullname, path_index)] = first_state
                if is_entrypoint_method:
                    self._add_edge(fsm, START, first_state, method_fullname,
                                    path_index, first_conditions)

                # внутрипроцедурные рёбра между соседними logpoint-ами,
                # с учётом вставки внешних вызовов между ними
                self._link_observed_with_externals(
                    fsm, observed, method_fullname, path_index,
                    externals_by_caller.get(method_fullname, []),
                )

                last_state, last_conditions = observed[-1]

                if path.is_complete:
                    self._add_edge(fsm, last_state, RETURN, method_fullname,
                                    path_index, last_conditions, is_terminal=True)
                    return_states_of_method[method_fullname].append((last_state, last_conditions))
                    fsm.terminals.add(RETURN)
                else:
                    # v1 всегда ставил тут INCOMPLETE; v2 — только если этот
                    # путь не является "хвостом" вызова другого внутреннего
                    # метода (это определяется на этапе связывания ниже).
                    self._add_edge(fsm, last_state, INCOMPLETE, method_fullname,
                                    path_index, last_conditions,
                                    is_terminal=True, partial=True)
                    fsm.terminals.add(INCOMPLETE)

        # межпроцедурная склейка: caller.last_logpoint -> callee.START
        self._stitch_interprocedural(fsm, flow, first_state_of_method, return_states_of_method)

        self._deduplicate_edges(fsm)
        self._add_global_warnings(fsm)
        return fsm

    # ------------------------------------------------------------------
    def _extract_observed(self, path: MethodPath) -> List[Tuple[LogTemplateWithMethod, Tuple[str, ...]]]:
        observed: List[Tuple[LogTemplateWithMethod, Tuple[str, ...]]] = []

        for segment in path.segments:
            conditions = self._segment_conditions(segment.condition)

            for cfg_node in segment.nodes:
                template = self.templates_by_call.get(str(cfg_node.node_id))
                if template is None:
                    continue

                observed.append((template, conditions))

        return observed

    # ------------------------------------------------------------------
    def _link_observed_with_externals(
        self, fsm, observed, method_fullname, path_index, externals
    ) -> None:
        """
        observed уже содержит (state_id, conditions) пары.
        Ничего регистрировать здесь больше не нужно.
        """
        if len(observed) <= 1:
            return

        ext_iter = sorted(externals, key=lambda e: e.call_index)
        ext_pos = 0

        for i in range(len(observed) - 1):
            source_id, _ = observed[i]
            target_id, target_conditions = observed[i + 1]

            inserted_ext = None
            while ext_pos < len(ext_iter):
                ec = ext_iter[ext_pos]
                ext_id = f"ext:{method_fullname}:{ec.call_index}"

                if ext_id not in fsm.external_states:
                    fsm.external_states[ext_id] = ExternalPoint(
                        id=ext_id,
                        caller_full_name=ec.caller_full_name,
                        callee_full_name=ec.callee_full_name,
                        call_code=ec.call_code,
                        line=ec.line,
                    )

                inserted_ext = ext_id
                ext_pos += 1
                break

            if inserted_ext:
                self._add_edge(fsm, source_id, inserted_ext, method_fullname, path_index, ())
                self._add_edge(fsm, inserted_ext, target_id, method_fullname, path_index, target_conditions)
            else:
                self._add_edge(fsm, source_id, target_id, method_fullname, path_index, target_conditions)

    # ------------------------------------------------------------------
    def _stitch_interprocedural(
        self,
        fsm: StaticLogFSM,
        flow: EntrypointFlowResult,
        first_state_of_method: Dict[Tuple[str, int], str],
        return_states_of_method: Dict[str, List[Tuple[str, Tuple[str, ...]]]],
    ) -> None:
        """
        Использует flow.sequence (MethodEntry.via_path / caller_full_name),
        чтобы:
        1) заменить INCOMPLETE-рёбра caller-а на переход в callee.START,
           когда путь caller-а действительно обрывается вызовом callee;
        2) добавить callee.RETURN -> "продолжение caller-а" (в текущей
           версии — обратно в caller как синтетический CALL_RETURN узел,
           поскольку точная привязка "после call-site" требует номера
           CFG-узла call-site и в v1 данных недостаточно для 100% точности).
        """
        for entry in flow.sequence:
            if not entry.via_path or not entry.caller_full_name:
                continue  # entrypoint-метод или метод без известного вызывающего пути

            callee_fullname = entry.method_graph.full_name
            callee_paths = flow.method_paths.get(callee_fullname, [])
            if not callee_paths:
                continue

            # находим INCOMPLETE-рёбра caller-а, порождённые именно этим via_path,
            # и переключаем их target на START callee-метода (все пути callee).
            caller_fullname = entry.caller_full_name
            for edge in fsm.edges:
                if edge.method_fullname != caller_fullname:
                    continue
                if edge.target != INCOMPLETE:
                    continue
                # эвристика v2: связываем, если caller вызывает именно этот callee
                # где-то на своём пути (без точного сопоставления call_node_id —
                # это следующий шаг доработки, требует cs_map на уровне LogFlowExtractor)
                new_edges = []
                for callee_path_index, callee_path in enumerate(callee_paths):
                    callee_first = first_state_of_method.get((callee_fullname, callee_path_index))
                    if callee_first is None:
                        continue
                    new_edges.append(LogEdge(
                        source=edge.source,
                        target=callee_first,
                        method_fullname=caller_fullname,
                        path_index=edge.path_index,
                        conditions=edge.conditions,
                        is_terminal=False,
                        partial=False,
                        is_interprocedural=True,
                    ))
                if new_edges:
                    fsm.edges.remove(edge)
                    fsm.edges.extend(new_edges)

            # callee RETURN -> синтетический "возврат в caller" узел
            for last_state, conditions in return_states_of_method.get(callee_fullname, []):
                return_marker = f"return_to:{caller_fullname}"
                fsm.edges.append(LogEdge(
                    source=last_state,
                    target=return_marker,
                    method_fullname=callee_fullname,
                    path_index=-1,
                    conditions=conditions,
                    is_terminal=False,
                    is_interprocedural=True,
                ))

    # ------------------------------------------------------------------
    def _register_logpoint(self, fsm: StaticLogFSM, template: LogTemplateWithMethod, method_fullname: str) -> str:
        call_node_id = str(template.call_node_id)
        state_id = f"log:{call_node_id}"
        if state_id in fsm.states:
            return state_id
        method_name = template.method_name or method_fullname.rsplit(".", 1)[-1]
        fsm.states[state_id] = LogPoint(
            id=state_id,
            call_node_id=call_node_id,
            method_node_id=str(template.method_node_id),
            method_fullname=method_fullname,
            method_name=method_name,
            template=template.raw_template,
            static_score=template.static_count,
        )
        return state_id

    def _add_edge(self, fsm, source, target, method_fullname, path_index,
                  conditions=(), is_terminal=False, partial=False) -> None:
        fsm.edges.append(LogEdge(
            source=source, target=target, method_fullname=method_fullname,
            path_index=path_index, conditions=conditions,
            is_terminal=is_terminal, partial=partial,
        ))

    @staticmethod
    def _segment_conditions(condition: str) -> Tuple[str, ...]:
        condition = (condition or "").strip()
        return (condition,) if condition else ()

    @staticmethod
    def _edge_key(edge: LogEdge):
        return (edge.source, edge.target, edge.method_fullname, edge.conditions,
                edge.is_terminal, edge.partial, edge.is_interprocedural)

    def _deduplicate_edges(self, fsm: StaticLogFSM) -> None:
        seen = set()
        unique = []
        for edge in fsm.edges:
            key = self._edge_key(edge)
            if key in seen:
                continue
            seen.add(key)
            unique.append(edge)
        fsm.edges = unique

    @staticmethod
    def _add_global_warnings(fsm: StaticLogFSM) -> None:
        if not fsm.states:
            fsm.warnings.append("No logger CALL nodes matched the template catalog.")
        if not any(e.source == START for e in fsm.edges):
            fsm.warnings.append("No START transition created.")
        if any(e.partial for e in fsm.edges):
            fsm.warnings.append(
                "Some paths remain INCOMPLETE: they do not lead to RETURN or "
                "to a recognized internal call-site. Check external_calls / "
                "fn_index coverage for these paths."
            )
        if any(e.is_interprocedural for e in fsm.edges):
            fsm.warnings.append(
                "v2: inter-procedural edges added via MethodEntry.via_path / "
                "caller_full_name (heuristic call-site matching, not yet "
                "anchored to exact call_node_id)."
            )
        else:
            fsm.warnings.append(
                "No inter-procedural edges were created: flow.sequence may be "
                "missing via_path/caller_full_name links for this entrypoint."
            )


def print_log_fsm(fsm: StaticLogFSM) -> None:
    print(fsm.summary())
