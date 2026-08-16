"""
runtime_analyzer.py
===================

End-to-end multi-service online RCA pipeline.

Pipeline:
    1. Load builder.py FLOW and FSM artifacts for every service under output/.
    2. Read a chronological multi-service runtime log CSV.
    3. Route each event to the explicit-hypothesis FSM matcher of its service.
    4. On terminal FSM paths, emit LLM-ready reports with a constrained FLOW slice.
    5. Emit reports for structurally unexplained events and optional incomplete paths.

Expected artifact layout:
    output/
      frontend/
        <entrypoint>.flow.json
        <entrypoint>.fsm.json
      cartservice/
        ...

Expected CSV layout by default:
    timestamp,service,message

Optional columns:
    bucket        - arbitrary correlation/display label
    call_node_id  - resolved CPG CALL id; enables exact transition matching

Usage:
    python runtime_analyzer.py --artifacts-dir output --logs runtime.csv

    python runtime_analyzer.py \\
        --artifacts-dir output \\
        --logs runtime.csv \\
        --reports-dir output/runtime_reports \\
        --threshold 0.60 \\
        --time-gap-sec 30 \\
        --max-hypotheses-per-chain 16 \\
        --max-call-depth 3 \\
        --emit-incomplete
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.offline.finite_state_machine import ExecutionSegment, LogTransition, StaticLogFSM
from src.online.stateful_chain_store import ExplicitHypothesisStore
from src.online.stream_processor import RuntimeStreamProcessor, ServiceCatalog

logger = logging.getLogger("runtime_analyzer")


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #

def tuple_or_empty(value: Any) -> tuple:
    return tuple(value) if value is not None else ()


def load_fsm(fsm_path: Path) -> StaticLogFSM:
    """Restore StaticLogFSM from builder.py <entrypoint>.fsm.json artifact."""
    payload = json.loads(fsm_path.read_text(encoding="utf-8"))
    states: dict[str, ExecutionSegment] = {}
    for raw in payload.get("states", []):
        state = ExecutionSegment(
            id=str(raw["id"]),
            entrypoint_full_name=str(raw.get("entrypoint_full_name", "")),
            method_full_name=str(raw.get("method_full_name", "")),
            direct_methods=tuple_or_empty(raw.get("direct_methods")),
            external_calls=tuple_or_empty(raw.get("external_calls")),
            conditions=tuple_or_empty(raw.get("conditions")),
            previous_log_call_node_id=raw.get("previous_log_call_node_id"),
            next_log_call_node_id=raw.get("next_log_call_node_id"),
            is_start=bool(raw.get("is_start", False)),
            is_terminal=bool(raw.get("is_terminal", False)),
            kind=str(raw.get("kind", "INCOMPLETE_SEGMENT")),
        )
        states[state.id] = state

    transitions = [
        LogTransition(
            id=str(raw["id"]),
            source_segment_id=str(raw["source_segment_id"]),
            target_segment_id=str(raw["target_segment_id"]),
            template=str(raw.get("template", "")),
            log_call_node_id=str(raw.get("log_call_node_id", "")),
            method_full_name=str(raw.get("method_full_name", "")),
            method_node_id=str(raw.get("method_node_id", "")),
            conditions=tuple_or_empty(raw.get("conditions")),
            static_score=int(raw.get("static_score", 0)),
        )
        for raw in payload.get("transitions", [])
    ]

    return StaticLogFSM(
        entrypoint_node_id=str(payload.get("entrypoint_node_id", "")),
        entrypoint_name=str(payload.get("entrypoint_name", fsm_path.stem.removesuffix(".fsm"))),
        entrypoint_full_name=str(payload.get("entrypoint_full_name", "")),
        states=states,
        transitions=transitions,
        warnings=list(payload.get("warnings", [])),
    )


def load_service_catalog(
    service_dir: Path,
    threshold: float,
    time_gap_sec: int,
    max_active_chains: int,
    max_hypotheses_per_chain: int,
    max_total_active_hypotheses: int,
) -> ServiceCatalog | None:
    """Load paired FLOW/FSM artifacts for one service directory."""
    fsms: dict[str, StaticLogFSM] = {}
    flows: dict[str, dict[str, Any]] = {}

    for fsm_path in sorted(service_dir.glob("*.fsm.json")):
        fsm = load_fsm(fsm_path)
        if fsm.transitions:
            fsms[fsm.entrypoint_name] = fsm
        else:
            logger.debug("Skipping FSM without transitions: %s", fsm_path)

    for flow_path in sorted(service_dir.glob("*.flow.json")):
        flow = json.loads(flow_path.read_text(encoding="utf-8"))
        entrypoint = str(flow.get("entrypoint_name") or flow_path.stem.removesuffix(".flow"))
        flows[entrypoint] = flow

    # An FSM can match logs without a FLOW report, but such a completed path
    # cannot become an LLM execution report; require paired artifacts here.
    paired = {name: fsm for name, fsm in fsms.items() if name in flows}
    paired_flows = {name: flows[name] for name in paired}
    if not paired:
        return None

    store = ExplicitHypothesisStore(
        fsms=paired,
        threshold=threshold,
        time_gap_sec=time_gap_sec,
        max_active_chains=max_active_chains,
        max_hypotheses_per_chain=max_hypotheses_per_chain,
        max_total_active_hypotheses=max_total_active_hypotheses,
    )
    return ServiceCatalog(fsms=paired, flows=paired_flows, store=store)


def load_catalogs(
    artifacts_dir: Path,
    threshold: float,
    time_gap_sec: int,
    max_active_chains: int,
    max_hypotheses_per_chain: int,
    max_total_active_hypotheses: int,
) -> dict[str, ServiceCatalog]:
    """Load every service directory that contains paired FLOW and FSM artifacts."""
    catalogs: dict[str, ServiceCatalog] = {}
    for service_dir in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        catalog = load_service_catalog(
            service_dir,
            threshold,
            time_gap_sec,
            max_active_chains,
            max_hypotheses_per_chain,
            max_total_active_hypotheses,
        )
        if catalog is not None:
            catalogs[service_dir.name] = catalog
            logger.info("Loaded service=%s entrypoints=%d", service_dir.name, len(catalog.fsms))
    if not catalogs:
        raise FileNotFoundError(f"No paired FLOW/FSM catalogs found under {artifacts_dir}")
    return catalogs


# --------------------------------------------------------------------------- #
# Report export
# --------------------------------------------------------------------------- #

def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "unnamed")
    return normalized.strip("._") or "unnamed"


def write_reports(reports: list[dict[str, Any]], reports_dir: Path) -> list[dict[str, Any]]:
    """Write individual report JSON files and a compact index."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []

    for sequence, report in enumerate(reports):
        service = safe_name(str(report.get("service", "unknown_service")))
        report_type = safe_name(str(report.get("report_type", "report")))
        chain_id = report.get("chain", {}).get("chain_id", report.get("chain_id", "event"))
        hypothesis_id = report.get("hypothesis", {}).get("hypothesis_id", report.get("hypothesis_id", sequence))
        filename = f"{service}_chain_{chain_id}_hypothesis_{hypothesis_id}_{report_type}.json"
        path = reports_dir / report_type / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append({
            "report_type": report.get("report_type"),
            "service": report.get("service"),
            "chain_id": chain_id,
            "hypothesis_id": hypothesis_id,
            "path": str(path),
        })

    (reports_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index


def write_store_audit(catalogs: dict[str, ServiceCatalog], reports_dir: Path) -> None:
    """Export chain and explicit-hypothesis tables for reproducibility."""
    audit_dir = reports_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for service, catalog in catalogs.items():
        catalog.store.chains_frame().to_csv(audit_dir / f"{safe_name(service)}_chains.csv", index=False)
        catalog.store.hypotheses_frame().to_csv(audit_dir / f"{safe_name(service)}_hypotheses.csv", index=False)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a multi-service runtime log stream against builder.py artifacts.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("output"),
                        help="Root artifact directory produced by builder.py (default: output)")
    parser.add_argument("--logs", type=Path, required=True,
                        help="Runtime log CSV with timestamp,service,message columns")
    parser.add_argument("--reports-dir", type=Path, default=Path("output/runtime_reports"))
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--service-column", default="service")
    parser.add_argument("--message-column", default="message")
    parser.add_argument("--bucket-column", default="bucket")
    parser.add_argument("--call-node-column", default=None,
                        help="Optional resolved CPG CALL node id column")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--time-gap-sec", type=int, default=30)
    parser.add_argument("--max-active-chains", type=int, default=512)
    parser.add_argument("--max-hypotheses-per-chain", type=int, default=32)
    parser.add_argument("--max-total-active-hypotheses", type=int, default=2048)
    parser.add_argument("--max-call-depth", type=int, default=3)
    parser.add_argument("--emit-incomplete", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    catalogs = load_catalogs(
        artifacts_dir=args.artifacts_dir,
        threshold=args.threshold,
        time_gap_sec=args.time_gap_sec,
        max_active_chains=args.max_active_chains,
        max_hypotheses_per_chain=args.max_hypotheses_per_chain,
        max_total_active_hypotheses=args.max_total_active_hypotheses,
    )
    logs = pd.read_csv(args.logs)
    processor = RuntimeStreamProcessor(catalogs, max_call_depth=args.max_call_depth)
    reports = processor.process_frame(
        logs,
        timestamp_column=args.timestamp_column,
        service_column=args.service_column,
        message_column=args.message_column,
        bucket_column=args.bucket_column,
        call_node_column=args.call_node_column,
    )
    if args.emit_incomplete:
        reports.extend(processor.finalize())

    index = write_reports(reports, args.reports_dir)
    write_store_audit(catalogs, args.reports_dir)
    logger.info("Done: reports=%d output=%s", len(index), args.reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
