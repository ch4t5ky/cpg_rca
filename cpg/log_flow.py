# cpg/log_flow.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

from cpg.flow import EntrypointFlowResult, MethodPath
from cpg_trie_parser import LogTemplateWithMethod


START = "__start__"
RETURN = "__return__"
INCOMPLETE = "__incomplete__"


@dataclass(frozen=True)
class LogPoint:
    """
    Статическая наблюдаемая точка: конкретный CALL logger-а в CPG.

    id привязан к call_node_id, а не к template: один и тот же template
    может использоваться в нескольких местах исходного кода.
    """

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
class LogEdge:
    """
    Допустимый статический переход между наблюдаемыми состояниями.

    source/target:
      - log:<CALL_NODE_ID>
      - __start__
      - __return__
      - __incomplete__
    """

    source: str
    target: str
    method_fullname: str
    path_index: int
    conditions: Tuple[str, ...] = ()
    is_terminal: bool = False
    partial: bool = False


@dataclass
class StaticLogFSM:
    """
    Проекция EntrypointFlowResult на logpoint-ы.

    Ограничение первой версии:
    - рёбра извлекаются только внутри каждого MethodPath;
    - порядок caller -> callee -> caller не материализуется;
    - связи между logpoint-ами разных методов пока не строятся.
    """

    entrypoint_node_id: str
    entrypoint_name: str
    entrypoint_fullname: str

    states: Dict[str, LogPoint] = field(default_factory=dict)
    edges: List[LogEdge] = field(default_factory=list)
    terminals: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)

    @property
    def edge_set(self) -> Set[Tuple[str, str]]:
        return {(edge.source, edge.target) for edge in self.edges}

    def outgoing(self, state_id: str) -> List[LogEdge]:
        return [edge for edge in self.edges if edge.source == state_id]

    def incoming(self, state_id: str) -> List[LogEdge]:
        return [edge for edge in self.edges if edge.target == state_id]

    def format_state(self, state_id: str) -> str:
        if state_id == START:
            return "START"
        if state_id == RETURN:
            return "RETURN"
        if state_id == INCOMPLETE:
            return "INCOMPLETE"

        point = self.states.get(state_id)
        if point is None:
            return state_id

        return f'{point.method_name}: "{point.template}"'

    def summary(self) -> str:
        return (
            f"entrypoint={self.entrypoint_fullname}, "
            f"states={len(self.states)}, "
            f"edges={len(self.edges)}, "
            f"terminals={len(self.terminals)}, "
            f"warnings={len(self.warnings)}"
        )

    def to_text(self) -> str:
        lines = [
            "=" * 80,
            "STATIC LOG FSM",
            "=" * 80,
            f"Entrypoint: {self.entrypoint_name}",
            f"Full name:  {self.entrypoint_fullname}",
            f"States:     {len(self.states)}",
            f"Edges:      {len(self.edges)}",
            "",
        ]

        if self.warnings:
            lines.append("WARNINGS")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        lines.append("STATES")
        if not self.states:
            lines.append("  <no logpoints found>")
        else:
            for state_id, point in sorted(
                self.states.items(),
                key=lambda item: (
                    item[1].method_fullname,
                    item[1].template,
                    item[0],
                ),
            ):
                lines.append(
                    f"  {state_id}"
                    f"\n"
                    f"    method:   {point.method_fullname}"
                    f"\n"
                    f"    template: {point.template}"
                    f"\n"
                    f"    call:     {point.call_node_id}"
                )

        lines.append("")
        lines.append("TRANSITIONS")

        if not self.edges:
            lines.append("  <no transitions found>")
        else:
            for edge in self.edges:
                source = self.format_state(edge.source)
                target = self.format_state(edge.target)

                meta: List[str] = [
                    f"method={edge.method_fullname}",
                    f"path={edge.path_index}",
                ]

                if edge.conditions:
                    meta.append("conditions=" + ",".join(edge.conditions))

                if edge.is_terminal:
                    meta.append("terminal=true")

                if edge.partial:
                    meta.append("partial=true")

                lines.append(
                    f"  {source}"
                    f"\n"
                    f"    -> {target}"
                    f"\n"
                    f"    [{'; '.join(meta)}]"
                )

        lines.append("=" * 80)
        return "\n".join(lines)


class LogFlowExtractor:
    """
    Извлекает внутриметодный StaticLogFSM из готового EntrypointFlowResult.

    Использование:

        from cpg.flow import EndpointFlow
        from cpg.log_flow import LogFlowExtractor
        from cpg_trie_parser import buildtemplatesfromcpg

        flow = EndpointFlow(graph)
        flow_result = flow.build(entrypoint_node_id)

        templates = buildtemplatesfromcpg(graph)
        extractor = LogFlowExtractor(templates)

        fsm = extractor.extract(flow_result)
        print(fsm.to_text())
    """

    def __init__(
        self,
        templates: Iterable[LogTemplateWithMethod],
    ) -> None:
        self.templates_by_call: Dict[str, LogTemplateWithMethod] = {}

        for template in templates:
            call_node_id = str(template.call_node_id)

            previous = self.templates_by_call.get(call_node_id)
            if previous is None:
                self.templates_by_call[call_node_id] = template
                continue

            if template.staticcount > previous.staticcount:
                self.templates_by_call[call_node_id] = template

    def extract(self, flow: EntrypointFlowResult) -> StaticLogFSM:
        """
        Строит автомат из путей, уже вычисленных EndpointFlow.

        Для каждого MethodPath:
        - находит logpoint-ы среди path.allnodes;
        - соединяет соседние logpoint-ы;
        - создаёт START -> first_logpoint только для entrypoint method;
        - создаёт last_logpoint -> RETURN, если path.iscomplete;
        - создаёт last_logpoint -> INCOMPLETE для оборванного пути.

        Один и тот же статический переход может быть обнаружен в нескольких
        CFG-путях. В итоговом FSM он дедуплицируется.
        """
        fsm = StaticLogFSM(
            entrypoint_node_id=flow.entrypoint_node_id,
            entrypoint_name=flow.entrypoint_name,
            entrypoint_fullname=flow.entrypoint_full_name,
        )

        if not self.templates_by_call:
            fsm.warnings.append(
                "Log template catalog is empty; no logpoint states can be extracted."
            )
            return fsm

        if not flow.method_paths:
            fsm.warnings.append(
                "Flow contains no method paths; no logpoint states can be extracted."
            )
            return fsm

        for method_fullname, paths in flow.method_paths.items():
            is_entrypoint_method = (
                method_fullname == flow.entrypoint_full_name
            )

            for path_index, path in enumerate(paths):
                self._extract_path(
                    fsm=fsm,
                    method_fullname=method_fullname,
                    path_index=path_index,
                    path=path,
                    is_entrypoint_method=is_entrypoint_method,
                )

        self._deduplicate_edges(fsm)
        self._add_global_warnings(fsm)

        return fsm

    def _extract_path(
        self,
        fsm: StaticLogFSM,
        method_fullname: str,
        path_index: int,
        path: MethodPath,
        is_entrypoint_method: bool,
    ) -> None:
        """
        Обрабатывает один CFG-путь одного метода.

        PathSegment.condition относится к сегменту, поэтому condition
        присоединяется к logpoint, встретившемуся внутри этого сегмента.
        """
        observed: List[Tuple[str, Tuple[str, ...]]] = []

        for segment in path.segments:
            segment_conditions = self._segment_conditions(segment.condition)

            for cfg_node in segment.nodes:
                template = self.templates_by_call.get(str(cfg_node.node_id))
                if template is None:
                    continue

                state_id = self._register_logpoint(
                    fsm=fsm,
                    template=template,
                    method_fullname=method_fullname,
                )

                observed.append((state_id, segment_conditions))

        if not observed:
            return

        first_state, first_conditions = observed[0]

        if is_entrypoint_method:
            self._add_edge(
                fsm=fsm,
                source=START,
                target=first_state,
                method_fullname=method_fullname,
                path_index=path_index,
                conditions=first_conditions,
            )

        for index in range(len(observed) - 1):
            source_id, source_conditions = observed[index]
            target_id, target_conditions = observed[index + 1]

            conditions = self._merge_conditions(
                source_conditions,
                target_conditions,
            )

            self._add_edge(
                fsm=fsm,
                source=source_id,
                target=target_id,
                method_fullname=method_fullname,
                path_index=path_index,
                conditions=conditions,
            )

        last_state, last_conditions = observed[-1]

        if path.is_complete:
            self._add_edge(
                fsm=fsm,
                source=last_state,
                target=RETURN,
                method_fullname=method_fullname,
                path_index=path_index,
                conditions=last_conditions,
                is_terminal=True,
            )
            fsm.terminals.add(RETURN)
        else:
            self._add_edge(
                fsm=fsm,
                source=last_state,
                target=INCOMPLETE,
                method_fullname=method_fullname,
                path_index=path_index,
                conditions=last_conditions,
                is_terminal=True,
                partial=True,
            )
            fsm.terminals.add(INCOMPLETE)

    def _register_logpoint(
        self,
        fsm: StaticLogFSM,
        template: LogTemplateWithMethod,
        method_fullname: str,
    ) -> str:
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

    def _add_edge(
        self,
        fsm: StaticLogFSM,
        source: str,
        target: str,
        method_fullname: str,
        path_index: int,
        conditions: Tuple[str, ...] = (),
        is_terminal: bool = False,
        partial: bool = False,
    ) -> None:
        fsm.edges.append(
            LogEdge(
                source=source,
                target=target,
                method_fullname=method_fullname,
                path_index=path_index,
                conditions=conditions,
                is_terminal=is_terminal,
                partial=partial,
            )
        )

    @staticmethod
    def _segment_conditions(condition: str) -> Tuple[str, ...]:
        """
        Flow хранит TRUE/FALSE/LOOPTRUE/LOOPFALSE в PathSegment.condition.

        Пока не интерпретируем условное выражение из исходного кода:
        сохраняем техническую метку, которую позднее можно обогатить
        текстом CONTROLSTRUCTURE condition.
        """
        condition = (condition or "").strip()

        if not condition:
            return ()

        return (condition,)

    @staticmethod
    def _merge_conditions(
        left: Tuple[str, ...],
        right: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        merged: List[str] = []

        for item in (*left, *right):
            if item and item not in merged:
                merged.append(item)

        return tuple(merged)

    @staticmethod
    def _edge_key(edge: LogEdge) -> Tuple[
        str,
        str,
        str,
        Tuple[str, ...],
        bool,
        bool,
    ]:
        return (
            edge.source,
            edge.target,
            edge.method_fullname,
            edge.conditions,
            edge.is_terminal,
            edge.partial,
        )

    def _deduplicate_edges(self, fsm: StaticLogFSM) -> None:
        """
        Одинаковые рёбра появляются при нескольких CFG-path-ах.

        path_index намеренно не входит в ключ дедупликации:
        он отражает путь-источник, но не изменяет семантику автомата.
        """
        seen: Set[
            Tuple[
                str,
                str,
                str,
                Tuple[str, ...],
                bool,
                bool,
            ]
        ] = set()

        unique: List[LogEdge] = []

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
            fsm.warnings.append(
                "No logger CALL nodes from the template catalog were found "
                "on the extracted CFG paths."
            )

        if not any(edge.source == START for edge in fsm.edges):
            fsm.warnings.append(
                "No START transition was created. The entrypoint method may "
                "not contain a reachable logger call, or its CFG paths are absent."
            )

        if any(edge.partial for edge in fsm.edges):
            fsm.warnings.append(
                "The FSM contains INCOMPLETE terminal transitions. "
                "These paths ended without a RETURN in the extracted CFG."
            )

        fsm.warnings.append(
            "Version 1 is intra-procedural: it does not materialize "
            "caller -> callee -> caller log ordering."
        )


def print_log_fsm(fsm: StaticLogFSM) -> None:
    print(fsm.to_text())