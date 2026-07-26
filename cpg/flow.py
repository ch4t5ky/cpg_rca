"""
EntrypointFlow — разворачивает полную последовательность вызовов от METHOD-узла
в CFG-порядке выполнения с учётом всех путей (if/else, циклы, ранний return).

Алгоритм
────────
1. _build_fn_index   : собирает FULL_NAME → node_id для всех внутренних
                       METHOD-узлов CPG (совместимость Joern 0.x / 1.x).
2. build(node_id)    : создаёт EntrypointFlowResult и запускает _expand от корня.
3. build_by_name     : то же, но поиск по короткому NAME.
4. all_entrypoints()   : возвращает методы с indegree=0 в call-graph.
5. _expand           : рекурсивный обход цепочки вызовов.
   • _build_cfg_paths(mg) строит все пути через CFG метода:
     - линейный участок   → один PathSegment
     - ветвление (if)     → расходимся; условие читается из mg.cfg_edges
     - RETURN             → путь терминален, не продолжается
     - back-edge (цикл)   → разворачивается max_loop_unroll раз
   • Для каждого пути разворачиваются только вызовы на этом пути.
   • Активная рекурсия (in_stack) → cycle_warnings.
   • Внешние вызовы → external_calls.
"""
from __future__ import annotations

import html
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from cpg.method import MethodConstructor, MethodGraph, CfgNode

__all__ = [
    "PathSegment",
    "MethodPath",
    "MethodEntry",
    "ExternalCall",
    "EntrypointFlowResult",
    "EntrypointFlow",
    "print_entrypoint_flow",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _clean(raw) -> str:
    s = html.unescape(str(raw)).strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1]
    return s


def _node_attr(G: nx.MultiDiGraph, nid: str, *keys: str) -> str:
    data = G.nodes.get(nid, {})
    for key in keys:
        val = _clean(data.get(key, ""))
        if val:
            return val
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Path data-classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PathSegment:
    """
    Один отрезок пути: линейная последовательность CFG-узлов.

    condition   : "" для линейного участка,
                  "TRUE" / "FALSE" для ветки после IF/WHILE/FOR,
                  "LOOP_BODY" для тела цикла (back-edge).
    is_terminal : True если сегмент заканчивается RETURN-узлом.
    """
    nodes:       List[CfgNode]
    condition:   str  = ""
    is_terminal: bool = False


@dataclass
class MethodPath:
    """
    Полный путь выполнения через один метод.

    segments     : чередование linear → branch → join → linear → ...
    is_complete  : True если путь дошёл до RETURN или конца метода.
    loop_unrolls : сколько раз был развёрнут back-edge.
    """
    segments:     List[PathSegment]
    is_complete:  bool = False
    loop_unrolls: int  = 0

    @property
    def all_nodes(self) -> List[CfgNode]:
        result = []
        for seg in self.segments:
            result.extend(seg.nodes)
        return result

    @property
    def call_node_ids(self) -> List[str]:
        """node_id-ы CALL-узлов в порядке появления на пути."""
        return [cn.node_id for cn in self.all_nodes if cn.label.upper() == "CALL"]

    def summary(self) -> str:
        lines = [
            f"entrypoint       : {self.entrypoint_name}",
            f"full name      : {self.entrypoint_full_name}",
            f"invocations    : {self.total_invocations}",
            f"unique methods : {len(self.unique_methods)}",
            f"external calls : {len(self.external_calls)}",
        ]
        if self.method_paths:
            total = sum(len(p) for p in self.method_paths.values())  # сумма, не произведение
            lines.append(f"total paths    : {total} (sum across methods)")
        if self.cycle_warnings:
            lines.append(f"cycles skipped : {', '.join(set(self.cycle_warnings))}")
        if self.max_depth_reached:
            lines.append("⚠ max_depth reached — chain truncated")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Flow data-classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MethodEntry:
    """Одно вхождение метода в развёрнутой последовательности вызовов."""
    method_graph:     MethodGraph
    depth:            int
    call_index:       int
    caller_full_name: str                  = ""
    via_path:         Optional[MethodPath] = None   # путь, на котором вызов


@dataclass
class ExternalCall:
    """Внешний (библиотечный) вызов — не разворачивается."""
    caller_full_name: str
    callee_full_name: str
    call_code:        str
    line:             str
    depth:            int
    call_index:       int


@dataclass
class EntrypointFlowResult:
    """Полная развёрнутая цепочка вызовов одного entrypoint-а."""
    entrypoint_node_id:   str
    entrypoint_name:      str
    entrypoint_full_name: str
    sequence:           List[MethodEntry]           = field(default_factory=list)
    external_calls:     List[ExternalCall]          = field(default_factory=list)
    cycle_warnings:     List[str]                   = field(default_factory=list)
    max_depth_reached:  bool                        = False
    # Все пути по каждому методу: full_name → List[MethodPath]
    method_paths:       Dict[str, List[MethodPath]] = field(default_factory=dict)

    @property
    def total_invocations(self) -> int:
        return len(self.sequence)

    @property
    def unique_methods(self) -> Set[str]:
        return {e.method_graph.full_name for e in self.sequence}

    def summary(self) -> str:
        lines = [
            f"entrypoint       : {self.entrypoint_name}",
            f"full name      : {self.entrypoint_full_name}",
            f"invocations    : {self.total_invocations}",
            f"unique methods : {len(self.unique_methods)}",
            f"external calls : {len(self.external_calls)}",
        ]
        if self.method_paths:
            total = sum(len(p) for p in self.method_paths.values())
            print(f"  Total paths    : {total} (sum across methods)")
            lines.append(f"total paths    : {total} (multiplicative across chain)")
        if self.cycle_warnings:
            lines.append(f"cycles skipped : {', '.join(set(self.cycle_warnings))}")
        if self.max_depth_reached:
            lines.append("⚠ max_depth reached — chain truncated")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# EntrypointFlow
# ─────────────────────────────────────────────────────────────────────────────


class EntrypointFlow:
    """
    Строит полную развёрнутую последовательность вызовов от METHOD-узла
    в порядке выполнения CFG, с учётом всех путей через каждый метод.

    Parameters
    ----------
    G               : nx.MultiDiGraph  — CPG-граф (загружен из .dot)
    max_depth       : максимальная глубина inline-разворачивания (-1 = без лимита)
    max_loop_unroll : сколько раз разворачивать тело цикла (back-edge)
    max_paths       : лимит путей на метод (защита от комбинаторного взрыва)

    Условия TRUE/FALSE на CFG-рёбрах читаются из MethodGraph.cfg_edges.
    Они не приходят напрямую из label CFG-ребра в Joern .dot: MethodConstructor
    выводит их из CONTROL_STRUCTURE + CONDITION + AST/BLOCK структуры метода.

    Usage
    ─────
        flow = EntrypointFlow
    (G, max_depth=10)

        result = flow.build("107374182770")
        print(result.summary())

        result = flow.build_by_name("viewCartHandler")
        print_entrypoint_flow(result, show_paths=True)

        for nid in flow.all_entrypoints():
            r = flow.build(nid)
            print(r.summary())
    """

    def __init__(
        self,
        G:               nx.MultiDiGraph,
        max_depth:       int = -1,
        max_loop_unroll: int = 1,
        max_paths:       int = 256,
    ) -> None:
        self._G               = G
        self._max_depth       = max_depth
        self._max_loop_unroll = max_loop_unroll
        self._max_paths       = max_paths
        self._mc              = MethodConstructor(G)
        self._fn_index        = self._build_fn_index()
        self._name_index      = self._build_name_index()

    # ── public ────────────────────────────────────────────────────────────────

    def build(self, entrypoint_node_id: str) -> EntrypointFlowResult:
        """Строит EntrypointFlowResult по node_id METHOD-узла."""
        full_name = _node_attr(self._G, entrypoint_node_id, "FULLNAME", "FULL_NAME")
        name      = _node_attr(self._G, entrypoint_node_id, "NAME")
        result = EntrypointFlowResult(
            entrypoint_node_id   = entrypoint_node_id,
            entrypoint_name      = name,
            entrypoint_full_name = full_name,
        )
        self._expand(
            full_name        = full_name,
            depth            = 0,
            counter          = [0],
            in_stack         = set(),
            result           = result,
            caller_full_name = "",
            via_path         = None,
        )
        return result

    def build_by_name(self, name: str) -> Optional[EntrypointFlowResult]:
        """Строит EntrypointFlowResult по короткому имени метода."""
        candidates = self._name_index.get(name)
        if not candidates:
            return None
        return self.build(candidates[0])

    # ── core: recursive inline expansion ─────────────────────────────────────

    def _expand(
        self,
        full_name:        str,
        depth:            int,
        counter:          List[int],
        in_stack:         Set[str],
        result:           EntrypointFlowResult,
        caller_full_name: str,
        via_path:         Optional[MethodPath],
    ) -> None:
        if self._max_depth >= 0 and depth > self._max_depth:
            result.max_depth_reached = True
            return

        if full_name in in_stack:
            result.cycle_warnings.append(full_name)
            return

        nid = self._fn_index.get(full_name)
        if nid is None:
            return

        mg = self._mc.build(nid)

        # Строим все CFG-пути метода (однократно, кэшируем в result)
        if full_name not in result.method_paths:
            result.method_paths[full_name] = self._build_cfg_paths(mg)
        paths = result.method_paths[full_name]

        # Фиксируем вхождение метода в sequence
        result.sequence.append(MethodEntry(
            method_graph     = mg,
            depth            = depth,
            call_index       = counter[0],
            caller_full_name = caller_full_name,
            via_path         = via_path,
        ))
        counter[0] += 1

        in_stack.add(full_name)
        cs_map = {cs.node_id: cs for cs in mg.call_sites}
        # Для каждого пути разворачиваем только вызовы, лежащие на нём
        seen_callees: Set[str] = set()

        for path in paths:
            for call_nid in path.call_node_ids:
                cs = cs_map.get(call_nid)
                if cs is None:
                    continue
                callee = cs.method_full_name
                if not callee or callee == full_name:
                    continue

                if callee in seen_callees:      # ← ключевое
                    continue
                seen_callees.add(callee)

                if callee in self._fn_index:
                    self._expand(
                        full_name        = callee,
                        depth            = depth + 1,
                        counter          = counter,
                        in_stack         = in_stack,
                        result           = result,
                        caller_full_name = full_name,
                        via_path         = path,   # via_path = первый путь где встретился
                    )
                else:
                    result.external_calls.append(ExternalCall(
                        caller_full_name = full_name,
                        callee_full_name = callee,
                        call_code        = cs.code,
                        line             = cs.line,
                        depth            = depth,
                        call_index       = counter[0],
                    ))
                    counter[0] += 1

        in_stack.discard(full_name)

    # ── CFG path building ─────────────────────────────────────────────────────

    def _build_cfg_paths(self, mg: MethodGraph) -> List[MethodPath]:
        """
        Строит все пути выполнения через CFG метода.

        Условия на рёбрах читаются из mg.cfg_edges (src, dst, condition).
        MethodConstructor выводит TRUE/FALSE/LOOP_TRUE/LOOP_FALSE из AST-структуры
        CONTROL_STRUCTURE-узлов и их CONDITION/AST-потомков.
        """
        # nx.DiGraph из mg.cfg_edges с атрибутом condition на рёбрах
        cfg = nx.DiGraph()
        node_map: Dict[str, CfgNode] = {cn.node_id: cn for cn in mg.cfg_nodes}

        for src, dst, cond in mg.cfg_edges:
            cfg.add_edge(src, dst, condition=cond)
        for nid in node_map:
            if nid not in cfg:
                cfg.add_node(nid)

        roots = [n for n in cfg.nodes if cfg.in_degree(n) == 0]
        if not roots:
            order = mg.cfg_order_approx
            roots = [order[0]] if order else []
        if not roots:
            return []

        collected: List[MethodPath] = []
        self._cfg_walk(
            cfg            = cfg,
            node_map       = node_map,
            mg             = mg,
            node           = roots[0],
            seg_nodes      = [],
            seg_condition  = "",
            segments       = [],
            collected      = collected,
            visited_counts = {},
            loop_unrolls   = 0,
        )
        return collected

    def _cfg_walk(
        self,
        cfg:            nx.DiGraph,
        node_map:       Dict[str, CfgNode],
        mg:             MethodGraph,
        node:           str,
        seg_nodes:      List[CfgNode],
        seg_condition:  str,
        segments:       List[PathSegment],
        collected:      List[MethodPath],
        visited_counts: Dict[str, int],
        loop_unrolls:   int,
    ) -> None:
        stack = [(node, seg_nodes, seg_condition, segments, visited_counts, loop_unrolls)]

        while stack:
            if len(collected) >= self._max_paths:
                break

            node, seg_nodes, seg_condition, segments, visited_counts, loop_unrolls = stack.pop()

            cn = node_map.get(node)
            if cn is None:
                self._cfg_close(seg_nodes, seg_condition, segments, collected,
                                is_terminal=False, loop_unrolls=loop_unrolls)
                continue
            
            count = visited_counts.get(node, 0)
            if count > self._max_loop_unroll:
                # текущий узел — внутри тела цикла
                # ищем ближайший предок с LOOP_FALSE ребром
                loop_exit = self._find_loop_exit(cfg, node, visited_counts)

                if loop_exit is not None:
                    stack.append((loop_exit, seg_nodes, seg_condition,
                                segments, visited_counts, loop_unrolls))
                else:
                    self._cfg_close(seg_nodes, seg_condition, segments, collected,
                                    is_terminal=False, loop_unrolls=loop_unrolls)
                continue

            # ключевое: обновляем visited_counts ТОЛЬКО для текущей ветки
            visited_counts = {**visited_counts, node: count + 1}
            if count > 0:
                loop_unrolls += 1

            if cn.label.upper() == "RETURN":
                self._cfg_close(seg_nodes + [cn], seg_condition, segments, collected,
                                is_terminal=True, loop_unrolls=loop_unrolls)
                continue

            seg_nodes = seg_nodes + [cn]
            successors = list(cfg.successors(node))

            if len(successors) == 0:
                self._cfg_close(seg_nodes, seg_condition, segments, collected,
                                is_terminal=False, loop_unrolls=loop_unrolls)

            elif len(successors) == 1:
                stack.append((successors[0], seg_nodes, seg_condition,
                            segments, visited_counts, loop_unrolls))

            else:
                new_seg = PathSegment(
                    nodes=seg_nodes,
                    condition=seg_condition,
                    is_terminal=False,
                )
                ordered = sorted(
                    successors,
                    key=lambda s: (
                        0 if cfg.edges[node, s].get("condition", "") in ("TRUE", "LOOP_TRUE") else
                        1 if cfg.edges[node, s].get("condition", "") in ("FALSE", "LOOP_FALSE") else 2,
                        node_map.get(s).cfg_index if node_map.get(s) else 10**9,
                    )
                )
                for succ in ordered:
                    cond = cfg.edges[node, succ].get("condition", "")
                    if visited_counts.get(succ, 0) > 0:
                        cond = cond or "LOOP_BODY"
                    # каждая ветка получает КОПИЮ visited_counts — не делит состояние с другими
                    stack.append((succ, [], cond,
                                segments + [new_seg],
                                dict(visited_counts),   # <— вот и всё исправление
                                loop_unrolls))
                    
    def _find_loop_exit(
        self,
        cfg:            nx.DiGraph,
        node:           str,
        visited_counts: Dict[str, int],
    ) -> Optional[str]:
        """
        Идём по предкам вверх, ищем узел с исходящим LOOP_FALSE ребром.
        Это condition-узел цикла — он уже посещён (visited_counts > 0).
        """
        # BFS по предкам
        queue = list(cfg.predecessors(node))
        seen  = {node}
        while queue:
            pred = queue.pop()
            if pred in seen:
                continue
            seen.add(pred)
            for succ in cfg.successors(pred):
                cond = cfg.edges[pred, succ].get("condition", "")
                if cond == "LOOP_FALSE":
                    return succ   # узел ПОСЛЕ цикла
            queue.extend(cfg.predecessors(pred))
        return None
                
    def _cfg_close(
        self,
        seg_nodes:     List[CfgNode],
        seg_condition: str,
        segments:      List[PathSegment],
        collected:     List[MethodPath],
        is_terminal:   bool,
        loop_unrolls:  int,
    ) -> None:
        if len(collected) >= self._max_paths:
            return
        final = list(segments)
        if seg_nodes:
            final.append(PathSegment(
                nodes       = list(seg_nodes),
                condition   = seg_condition,
                is_terminal = is_terminal,
            ))
        collected.append(MethodPath(
            segments     = final,
            is_complete  = is_terminal,
            loop_unrolls = loop_unrolls,
        ))

    def _find_join(
        self,
        cfg:         nx.DiGraph,
        mg:          MethodGraph,
        branch_node: str,
        successors:  List[str],
    ) -> Optional[str]:
        """
        Immediate post-dominator через пересечение множеств достижимости.
        Возвращает ближайший общий узел по cfg_order_approx.
        """
        reachable_sets = [
            nx.descendants(cfg, s) | {s}
            for s in successors
        ]
        common = reachable_sets[0]
        for rs in reachable_sets[1:]:
            common = common & rs
        common.discard(branch_node)

        if not common:
            return None

        order = mg.cfg_order or mg.cfg_order_approx
        for nid in order:
            if nid in common:
                return nid
        return None

    def _trim_before(
        self,
        segments:  List[PathSegment],
        join_node: str,
    ) -> List[PathSegment]:
        """Срезает сегменты до join_node (не включая его)."""
        result = []
        for seg in segments:
            cut_nodes = []
            hit = False
            for cn in seg.nodes:
                if cn.node_id == join_node:
                    hit = True
                    break
                cut_nodes.append(cn)
            if cut_nodes:
                result.append(PathSegment(
                    nodes       = cut_nodes,
                    condition   = seg.condition,
                    is_terminal = seg.is_terminal,
                ))
            if hit:
                break
        return result

    # ── index builders ────────────────────────────────────────────────────────

    def _build_fn_index(self) -> Dict[str, str]:
        """FULL_NAME → node_id для всех внутренних METHOD-узлов."""
        index: Dict[str, str] = {}
        for nid, data in self._G.nodes(data=True):
            if _clean(data.get("label", "")).upper() != "METHOD":
                continue
            ext = _clean(data.get("ISEXTERNAL", data.get("IS_EXTERNAL", ""))).lower()
            if ext == "true":
                continue
            if _clean(data.get("NAME", "")).startswith("<"):
                continue
            fn = _clean(data.get("FULLNAME", data.get("FULL_NAME", "")))
            if fn:
                index[fn] = nid
        return index

    def _build_name_index(self) -> Dict[str, List[str]]:
        """NAME → [node_id, ...]"""
        index: Dict[str, List[str]] = defaultdict(list)
        for nid in self._fn_index.values():
            name = _clean(self._G.nodes.get(nid, {}).get("NAME", ""))
            if name:
                index[name].append(nid)
        return dict(index)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────


def print_entrypoint_flow(
    result:         EntrypointFlowResult,
    show_cfg:       bool = False,
    show_paths:     bool = False,
    show_external:  bool = False,
    show_locals:    bool = False,
    max_cfg_nodes:  int  = 10,
    max_path_nodes: int  = 8,
) -> None:
    """
    Вывод EntrypointFlowResult в stdout.

    show_cfg      : первые max_cfg_nodes узлов CFG для каждого метода
    show_paths    : все пути каждого метода с CFG-узлами
    show_external : EXT-вызовы рядом с методом
    show_locals   : локальные переменные метода
    max_path_nodes: лимит строк узлов на один путь (0 = без лимита)
    """
    w = 72
    print(f"\n{'═' * w}")
    print(f"  entrypoint FLOW  {result.entrypoint_name}")
    print(f"  {result.entrypoint_full_name}")
    print(f"  Invocations    : {result.total_invocations}")
    print(f"  Unique methods : {len(result.unique_methods)}")
    print(f"  External calls : {len(result.external_calls)}")
    if result.method_paths:
        total = 1
        for paths in result.method_paths.values():
            if paths:
                total *= len(paths)
        print(f"  Total paths    : {total} (multiplicative across chain)")
    if result.cycle_warnings:
        print(f"  ⚠ Cycles       : {', '.join(set(result.cycle_warnings))}")
    if result.max_depth_reached:
        print("  ⚠ max_depth reached — chain truncated")
    print(f"{'-' * w}")

    ext_by_caller: Dict[str, List[ExternalCall]] = defaultdict(list)
    if show_external:
        for ec in result.external_calls:
            ext_by_caller[ec.caller_full_name].append(ec)

    for e in result.sequence:
        mg     = e.method_graph
        indent = "  " * e.depth
        topo   = "topo" if mg.cfg_order else "dfs-approx"

        print(f"  [{e.call_index:>3}] {indent}{mg.name}")
        print(f"        {indent}full : {mg.full_name}")
        print(f"        {indent}file : {mg.filename}  L{mg.line_start}–{mg.line_end}")
        print(f"        {indent}cfg  : {len(mg.cfg_nodes)} nodes  ({topo})")

        if show_locals and mg.local_vars:
            for lv in mg.local_vars[:6]:
                print(f"        {indent}  VAR  {lv.name}: {lv.type_full_name}")

        if show_cfg:
            order    = mg.cfg_order if mg.cfg_order else mg.cfg_order_approx
            node_map = {cn.node_id: cn for cn in mg.cfg_nodes}
            shown    = 0
            for idx, nid in enumerate(order):
                cn = node_map.get(nid)
                if cn:
                    print(f"        {indent}  {idx:>3} L{cn.line:>4}  "
                          f"{cn.label:18s}  {cn.code[:38]}")
                    shown += 1
                    if shown >= max_cfg_nodes:
                        rest = len(order) - shown
                        if rest > 0:
                            print(f"        {indent}  … {rest} more nodes")
                        break

        if show_paths and mg.full_name in result.method_paths:
            paths = result.method_paths[mg.full_name]
            print(f"        {indent}  PATHS ({len(paths)}):")
            for i, path in enumerate(paths):
                status = "✓" if path.is_complete else "○"
                total  = sum(len(s.nodes) for s in path.segments)
                loops  = f" loops×{path.loop_unrolls}" if path.loop_unrolls else ""
                print(f"        {indent}    PATH {i+1:>2} {status}{loops} "
                      f"({total} nodes, {len(path.segments)} segments)")
                shown = 0
                stop  = False
                for seg in path.segments:
                    if stop:
                        break
                    cond = f"[{seg.condition}] " if seg.condition else ""
                    term = " ⟵ RETURN" if seg.is_terminal else ""
                    print(f"        {indent}      ── {cond}segment "
                          f"({len(seg.nodes)} nodes){term}")
                    for cn in seg.nodes:
                        if max_path_nodes > 0 and shown >= max_path_nodes:
                            print(f"        {indent}         … {total - shown} more nodes")
                            stop = True
                            break
                        print(f"        {indent}         L{cn.line:>4}  "
                              f"{cn.label:18s}  {cn.code[:34]}")
                        shown += 1

        if show_external:
            for ec in ext_by_caller.get(mg.full_name, []):
                short    = ec.callee_full_name.rsplit(".", 1)[-1][:50]
                line_tag = f"L{ec.line} " if ec.line else ""
                print(f"        {indent}  EXT  {line_tag}{short}")

        print()

    print(f"{'═' * w}\n")

def print_paths_summary(result: EntrypointFlowResult, method_full_name: str) -> None:
    paths = result.method_paths.get(method_full_name, [])
    if not paths:
        print(f"No paths for {method_full_name}")
        return

    # Сортировка: сначала завершённые, потом по длине, потом лексически
    def sort_key(p):
        conditions = [s.condition for s in p.segments if s.condition]
        return (0 if p.is_complete else 1, len(conditions), conditions)

    sorted_paths = sorted(paths, key=sort_key)

    print(f"\n{method_full_name}  ({len(paths)} paths)")
    print(f"  {'#':>3}  {'S':1}  {'nodes':>5}  path")
    print(f"  {'─'*3}  {'─':1}  {'─'*5}  {'─'*50}")
    for i, p in enumerate(sorted_paths, 1):
        conditions = [s.condition for s in p.segments if s.condition]
        status     = "✓" if p.is_complete else "○"
        nodes      = sum(len(s.nodes) for s in p.segments)
        loops      = f" ↺{p.loop_unrolls}" if p.loop_unrolls else ""
        cond_str   = " → ".join(conditions) if conditions else "linear"
        print(f"  {i:>3}  {status}  {nodes:>5}{loops}  {cond_str}")

    # Какие методы посещены на каждом пути
    print(f"\n  Calls per path:")
    for i, p in enumerate(sorted_paths, 1):
        calls = [cn.code[:40] for cn in p.all_nodes if cn.label.upper() == "CALL"]
        if calls:
            print(f"  {i:>3}  {', '.join(calls)}")