import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import pydot

import src.offline.log as cpg_parser
from src.offline.entrypoint import EntrypointDetector
from src.offline.flow import EntrypointFlow
from src.offline.finite_state_machine import LogFlowExtractor

logger = logging.getLogger("builder")


# --------------------------------------------------------------------------- #
# Загрузка CPG
# --------------------------------------------------------------------------- #

def load_cpg(dot_path: Path) -> nx.MultiDiGraph:
    """Читает export.dot и строит networkx.MultiDiGraph."""
    dot_text = dot_path.read_text(encoding="utf-8")
    graphs = pydot.graph_from_dot_data(dot_text)
    if not graphs:
        raise ValueError(f"Не удалось распарсить DOT-файл: {dot_path}")

    p = graphs[0]
    graph = nx.MultiDiGraph()

    for node in p.get_nodes():
        name = node.get_name()
        if name in (None, "node", "graph", "edge"):
            continue
        node_id = str(name).strip('"')
        attrs = dict(node.get_attributes())
        graph.add_node(node_id, **attrs)

    for edge in p.get_edges():
        src = str(edge.get_source()).strip('"')
        dst = str(edge.get_destination()).strip('"')
        attrs = dict(edge.get_attributes())
        graph.add_edge(src, dst, **attrs)

    logger.info("CPG загружен: %d узлов, %d рёбер", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _json_default(value: Any):
    if hasattr(value, "value") and hasattr(type(value), "__members__"):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize type {type(value).__name__}")

# --------------------------------------------------------------------------- #
# Log Trie
# --------------------------------------------------------------------------- #

def serialize_trie_node(node) -> dict:
    """Рекурсивно сериализует TrieNode (children: Dict[str, TrieNode], terminals: List[LogTemplate])."""
    return {
        "terminals": [asdict(t) for t in node.terminals],
        "children": {
            token: serialize_trie_node(child) for token, child in node.children.items()
        },
    }

def build_log_trie(graph: nx.MultiDiGraph, output_dir: Path, max_ddg_depth: int = 5):
    """Строит лог-шаблоны и Trie, сохраняет визуализацию (trie.png) и данные (trie.json)."""
    templates = cpg_parser.build_templates_from_cpg(graph, max_ddg_depth=max_ddg_depth)
    root = cpg_parser.build_trie(templates)

    trie_png_path = output_dir / "trie.png"
    cpg_parser.visualize_trie_matplotlib(root, output_path=str(trie_png_path))

    trie_json_path = output_dir / "trie.json"
    trie_payload = {
        "templates_total": len(templates),
        "templates": [asdict(t) for t in templates],
        "trie": serialize_trie_node(root),
    }
    trie_json_path.write_text(
        json.dumps(trie_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    logger.info("Log Trie: %d шаблонов -> %s, %s", len(templates), trie_png_path, trie_json_path)
    return templates, root



# --------------------------------------------------------------------------- #
# Сериализация FLOW / FSM
# --------------------------------------------------------------------------- #

def serialize_flow(flow_result) -> dict:
    return {
        "entrypoint_node_id": flow_result.entrypoint_node_id,
        "entrypoint_name": flow_result.entrypoint_name,
        "entrypoint_full_name": flow_result.entrypoint_full_name,
        "summary": flow_result.summary(),
        "cycle_warnings": flow_result.cycle_warnings,
        "max_depth_reached": flow_result.max_depth_reached,
        "external_calls": [asdict(call) for call in flow_result.external_calls],
        "methods": {
            f"{method_name}": {
                "method_node_id": semantic_graph.method_node_id,
                "start_node_id": semantic_graph.start_node_id,
                "return_node_ids": list(semantic_graph.return_node_ids),
                "input_parameters": semantic_graph.input_parameters,
                "output_parameters": semantic_graph.output_parameters,
                "nodes": [
                    {
                        "node_id": unit.node_id,
                        "kind": unit.kind,
                        "code": unit.code,
                        "line": unit.line,
                        "raw_cfg_node_ids": unit.raw_cfg_node_ids,
                        "defines": [asdict(var) for var in unit.defines],
                        "uses": [asdict(var) for var in unit.uses],
                        "call_node_ids": unit.call_node_ids,
                        "callee_full_names": unit.callee_full_names,
                        "internal_callee_full_names": unit.internal_callee_full_names,
                    }
                    for unit in semantic_graph.nodes.values()
                ],
                "edges": [asdict(edge) for edge in semantic_graph.edge_list],
            }
            for method_name, semantic_graph in flow_result.semantic_graphs.items()
        },
    }

def serialize_fsm(fsm) -> dict:
    return {
        "entrypoint_node_id": fsm.entrypoint_node_id,
        "entrypoint_name": fsm.entrypoint_name,
        "entrypoint_full_name": fsm.entrypoint_full_name,
        "summary": fsm.summary(),
        "warnings": fsm.warnings,
        "start_states": sorted(fsm.start_states),
        "terminal_states": sorted(fsm.terminals),
        "states": [asdict(state) for state in fsm.states.values()],
        "transitions": [asdict(edge) for edge in fsm.transitions],
    }

def _safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("<", "_")
        .replace(">", "_")
    )

# --------------------------------------------------------------------------- #
# FLOW + FSM для всех entrypoint'ов сервиса
# --------------------------------------------------------------------------- #

def build_entrypoint_artifacts(
    graph: nx.MultiDiGraph,
    templates,
    output_dir: Path,
    flow_max_depth: int = 5,
    flow_max_paths: int = 50
) -> list[dict]:
    """Строит FLOW и FSM для каждого entrypoint'а сервиса, сохраняет JSON (и PNG опционально)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = EntrypointDetector(graph)
    all_entrypoints = detector.detect()
    logger.info("Найдено %d entrypoint(s)", len(all_entrypoints))

    flow_analyzer = EntrypointFlow(graph, max_depth=flow_max_depth, max_paths=flow_max_paths)
    fsm_extractor = LogFlowExtractor(templates)

    index: list[dict] = []

    for i, entrypoint in enumerate(all_entrypoints, start=1):
        artifact_name = _safe_name(entrypoint.name)
        logger.info("[%d/%d] entrypoint: %s", i, len(all_entrypoints), entrypoint.name)

        try:
            flow_result = flow_analyzer.build(entrypoint.node_id)
            fsm = fsm_extractor.extract(flow_result)

            flow_path = output_dir / f"{artifact_name}.flow.json"
            fsm_path = output_dir / f"{artifact_name}.fsm.json"

            flow_path.write_text(
                json.dumps(serialize_flow(flow_result), ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            fsm_path.write_text(
                json.dumps(serialize_fsm(fsm), ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

            semantic_image_path = None
            fsm_image_path = None

            row = {
                "entrypoint": entrypoint.name,
                "entrypoint_full_name": entrypoint.full_name,
                "status": "ok",
                "reachable_methods": len(flow_result.semantic_graphs),
                "semantic_nodes": sum(len(sg.nodes) for sg in flow_result.semantic_graphs.values()),
                "semantic_edges": sum(len(sg.edges) for sg in flow_result.semantic_graphs.values()),
                "fsm_states": len(fsm.states),
                "fsm_transitions": len(fsm.transitions),
                "fsm_warnings": fsm.warnings,
                "flow_file": flow_path.name,
                "fsm_file": fsm_path.name,
                "semantic_image": str(semantic_image_path) if semantic_image_path else None,
                "fsm_image": str(fsm_image_path) if fsm_image_path else None,
            }
            index.append(row)
            logger.info(
                "  ok: methods=%d semantic(nodes=%d,edges=%d) fsm(states=%d,transitions=%d)",
                row["reachable_methods"], row["semantic_nodes"], row["semantic_edges"],
                row["fsm_states"], row["fsm_transitions"],
            )

        except Exception as error:  # noqa: BLE001 - продолжаем обработку остальных entrypoint'ов
            row = {
                "entrypoint": entrypoint.name,
                "entrypoint_full_name": getattr(entrypoint, "full_name", None),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            index.append(row)
            logger.exception("  ERROR при обработке entrypoint %s", entrypoint.name)

    (output_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index


# --------------------------------------------------------------------------- #
# Обработка одного сервиса / всех сервисов в папке
# --------------------------------------------------------------------------- #

def discover_services(services_dir: Path) -> list[Path]:
    """Возвращает список путей к export.dot внутри каждой подпапки-сервиса."""
    dot_files = sorted(services_dir.glob("*/export.dot"))
    if not dot_files:
        raise FileNotFoundError(
            f"В {services_dir} не найдено ни одного 'export.dot' (ожидается <service>/export.dot)"
        )
    return dot_files


def process_service(
    service_name: str,
    dot_path: Path,
    output_root: Path,
    max_ddg_depth: int = 5,
    flow_max_depth: int = 5,
    flow_max_paths: int = 50
) -> dict:
    """Полный пайплайн обработки одного сервиса: CPG -> Log Trie + FLOW + FSM."""
    logger.info("=== Сервис: %s ===", service_name)
    output_dir = output_root / service_name
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_cpg(dot_path)
    templates, _root = build_log_trie(graph, output_dir, max_ddg_depth=max_ddg_depth)

    index = build_entrypoint_artifacts(
        graph=graph,
        templates=templates,
        output_dir=output_dir,
        flow_max_depth=flow_max_depth,
        flow_max_paths=flow_max_paths,
    )

    summary = {
        "service": service_name,
        "dot_path": str(dot_path),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "templates": len(templates),
        "entrypoints_total": len(index),
        "entrypoints_ok": sum(1 for row in index if row["status"] == "ok"),
        "entrypoints_error": sum(1 for row in index if row["status"] == "error"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Сервис %s готов: templates=%d entrypoints=%d (ok=%d, error=%d)",
        service_name, summary["templates"], summary["entrypoints_total"],
        summary["entrypoints_ok"], summary["entrypoints_error"],
    )
    return summary


def process_services_dir(
    services_dir: Path,
    output_root: Path,
    only_service: str | None = None,
    max_ddg_depth: int = 5,
    flow_max_depth: int = 5,
    flow_max_paths: int = 50
) -> list[dict]:
    """Обходит все сервисы в services_dir и обрабатывает каждый export.dot."""
    dot_files = discover_services(services_dir)

    if only_service:
        dot_files = [p for p in dot_files if p.parent.name == only_service]
        if not dot_files:
            raise FileNotFoundError(f"Сервис '{only_service}' не найден в {services_dir}")

    summaries = []
    for dot_path in dot_files:
        service_name = dot_path.parent.name
        try:
            summary = process_service(
                service_name=service_name,
                dot_path=dot_path,
                output_root=output_root,
                max_ddg_depth=max_ddg_depth,
                flow_max_depth=flow_max_depth,
                flow_max_paths=flow_max_paths
            )
        except Exception as error:  # noqa: BLE001 - не прерываем обработку остальных сервисов
            logger.exception("Сбой при обработке сервиса %s", service_name)
            summary = {"service": service_name, "dot_path": str(dot_path), "status": "error",
                       "error": f"{type(error).__name__}: {error}"}
        summaries.append(summary)

    (output_root / "services_index.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summaries


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Строит FLOW, FSM и Log Trie из готовых CPG (export.dot) по сервисам."
    )
    parser.add_argument(
        "--services-dir", type=Path, default=Path("services"),
        help="Папка с сервисами вида <service>/export.dot (по умолчанию: services)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Папка для результатов (по умолчанию: output)",
    )
    parser.add_argument(
        "--service", type=str, default=None,
        help="Обработать только один сервис по имени (иначе — все найденные)",
    )
    parser.add_argument("--max-ddg-depth", type=int, default=5, help="Глубина DDG для лог-шаблонов")
    parser.add_argument("--flow-max-depth", type=int, default=5, help="Максимальная глубина FLOW-обхода")
    parser.add_argument("--flow-max-paths", type=int, default=50, help="Максимум путей в FLOW-обходе")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод (DEBUG)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = process_services_dir(
        services_dir=args.services_dir,
        output_root=args.output_dir,
        only_service=args.service,
        max_ddg_depth=args.max_ddg_depth,
        flow_max_depth=args.flow_max_depth,
        flow_max_paths=args.flow_max_paths
    )

    ok = sum(1 for s in summaries if s.get("entrypoints_error", 0) == 0 and "error" not in s)
    logger.info("Готово: %d/%d сервисов обработано без ошибок. Результаты -> %s",
                ok, len(summaries), args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
