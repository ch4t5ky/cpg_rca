"""
runtime_analyzer.py
===================

Trie-first multi-service runtime analyzer with incremental report export.

For every service under --artifacts-dir, the analyzer loads:
    trie.json                 -> runtime message to CPG CALL candidates
    <entrypoint>.fsm.json     -> structural transition validation
    <entrypoint>.flow.json    -> completed-path FLOW slice for LLM reports

Reports are written immediately as they are produced. Large input streams do
not need to finish before output/runtime_reports starts receiving JSON files.

Usage:
    python3.12 runtime_analyzer.py \\
      --artifacts-dir output \\
      --logs dataset/logs.csv \\
      --reports-dir output/runtime_reports \\
      --service-column container_name \\
      --progress-every 1000 \\
      --checkpoint-every 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.offline.finite_state_machine import ExecutionSegment, LogTransition, StaticLogFSM
from src.online.stateful_chain_store import ExplicitHypothesisStore
from src.online.stream_processor import RuntimeEvent, RuntimeStreamProcessor, ServiceCatalog
from src.online.trie_matcher import ServiceTrieMatcher

logger = logging.getLogger("runtime_analyzer")

_TIMESTAMP_ALIASES = ("timestamp", "time", "ts", "unix_timestamp", "event_time")
_SERVICE_ALIASES = (
    "service", "microservice", "container", "container_name", "containername",
    "component", "application", "app", "source",
)
_MESSAGE_ALIASES = ("message", "log", "log_message", "logmessage", "content", "text")
_BUCKET_ALIASES = ("bucket", "stream", "source_bucket", "log_bucket")
_CALL_NODE_ALIASES = ("call_node_id", "callnodeid", "cpg_call_id", "cpg_call_node_id")


# --------------------------------------------------------------------------- #
# Offline artifact loading
# --------------------------------------------------------------------------- #

def tuple_or_empty(value: Any) -> tuple:
    """Restore tuple fields converted to JSON lists by dataclasses.asdict()."""
    return tuple(value) if value is not None else ()


def load_fsm(path: Path) -> StaticLogFSM:
    """Restore one StaticLogFSM from a builder.py <entrypoint>.fsm.json artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))

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
        entrypoint_name=str(payload.get("entrypoint_name", path.stem.removesuffix(".fsm"))),
        entrypoint_full_name=str(payload.get("entrypoint_full_name", "")),
        states=states,
        transitions=transitions,
        warnings=list(payload.get("warnings", [])),
    )


def load_service_catalog(service_dir: Path, args: argparse.Namespace) -> ServiceCatalog | None:
    """Load all paired Trie/FSM/FLOW artifacts for one service directory."""
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

    paired_fsms = {name: fsm for name, fsm in fsms.items() if name in flows}
    if not paired_fsms:
        return None

    trie_path = service_dir / "trie.json"
    trie_matcher = ServiceTrieMatcher(trie_path) if trie_path.is_file() else None
    if trie_matcher is None:
        logger.warning("Service=%s has no trie.json; lexical FSM fallback will be used", service_dir.name)

    store = ExplicitHypothesisStore(
        fsms=paired_fsms,
        threshold=args.threshold,
        time_gap_sec=args.time_gap_sec,
        max_active_chains=args.max_active_chains,
        max_hypotheses_per_chain=args.max_hypotheses_per_chain,
        max_total_active_hypotheses=args.max_total_active_hypotheses,
    )

    return ServiceCatalog(
        fsms=paired_fsms,
        flows={name: flows[name] for name in paired_fsms},
        store=store,
        trie_matcher=trie_matcher,
    )


def load_catalogs(artifacts_dir: Path, args: argparse.Namespace) -> dict[str, ServiceCatalog]:
    """Load every service with paired FLOW/FSM artifacts under artifacts_dir."""
    catalogs: dict[str, ServiceCatalog] = {}
    for service_dir in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        catalog = load_service_catalog(service_dir, args)
        if catalog is not None:
            catalogs[service_dir.name] = catalog
            logger.info(
                "Loaded service=%s entrypoints=%d trie=%s",
                service_dir.name,
                len(catalog.fsms),
                bool(catalog.trie_matcher),
            )

    if not catalogs:
        raise FileNotFoundError(f"No paired FLOW/FSM catalogs found under {artifacts_dir}")
    return catalogs


# --------------------------------------------------------------------------- #
# Runtime CSV schema resolution
# --------------------------------------------------------------------------- #

def normalize_header(value: object) -> str:
    """Normalize arbitrary CSV headers for case-insensitive alias matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def resolve_column(
    columns: Iterable[object],
    requested: str | None,
    aliases: Iterable[str],
    logical_name: str,
    required: bool = True,
) -> str | None:
    """Resolve a physical CSV header from an explicit name or known aliases."""
    normalized = {normalize_header(column): str(column) for column in columns}
    for candidate in ([requested] if requested else []) + list(aliases):
        if candidate:
            resolved = normalized.get(normalize_header(candidate))
            if resolved is not None:
                return resolved

    if not required:
        return None

    raise ValueError(
        f"Cannot resolve required '{logical_name}' column. "
        f"Actual CSV columns: {list(map(str, columns))}. "
        f"Pass --{logical_name}-column <actual-column-name>."
    )


def prepare_logs(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Normalize an arbitrary runtime CSV into canonical stream columns."""
    timestamp_column = resolve_column(raw.columns, args.timestamp_column, _TIMESTAMP_ALIASES, "timestamp")
    service_column = resolve_column(raw.columns, args.service_column, _SERVICE_ALIASES, "service")
    message_column = resolve_column(raw.columns, args.message_column, _MESSAGE_ALIASES, "message")
    bucket_column = resolve_column(raw.columns, args.bucket_column, _BUCKET_ALIASES, "bucket", required=False)
    call_node_column = resolve_column(raw.columns, args.call_node_column, _CALL_NODE_ALIASES, "call-node", required=False)

    logs = pd.DataFrame({
        "timestamp": pd.to_numeric(raw[timestamp_column], errors="coerce"),
        "service": raw[service_column].astype(str).str.strip(),
        "message": raw[message_column].astype(str),
    })
    logs = logs[logs["timestamp"].notna()].copy()
    logs["timestamp"] = logs["timestamp"].astype("int64")

    if bucket_column:
        logs["bucket"] = raw.loc[logs.index, bucket_column].astype(str)
    else:
        logs["bucket"] = logs["service"] + "@" + logs["timestamp"].astype(str)

    if call_node_column:
        logs["call_node_id"] = raw.loc[logs.index, call_node_column]

    logger.info(
        "Resolved CSV schema: timestamp=%s service=%s message=%s bucket=%s call_node=%s",
        timestamp_column,
        service_column,
        message_column,
        bucket_column,
        call_node_column,
    )
    return logs.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Incremental report and audit export
# --------------------------------------------------------------------------- #

def safe_name(value: str) -> str:
    """Create a deterministic filesystem-safe name component."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "unnamed")
    return value.strip("._") or "unnamed"


def write_report_incrementally(
    report: dict[str, Any],
    reports_dir: Path,
    sequence: int,
) -> dict[str, Any]:
    """Write exactly one report immediately and return its index metadata."""
    service = safe_name(str(report.get("service", "unknown")))
    report_type = safe_name(str(report.get("report_type", "report")))
    chain_id = report.get("chain", {}).get("chain_id", report.get("chain_id", "event"))
    hypothesis_id = report.get("hypothesis", {}).get("hypothesis_id", report.get("hypothesis_id", sequence))

    filename = (
        f"{sequence:09d}_{service}_chain_{chain_id}_"
        f"hypothesis_{hypothesis_id}_{report_type}.json"
    )
    path = reports_dir / report_type / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "sequence": sequence,
        "report_type": report.get("report_type"),
        "service": report.get("service"),
        "chain_id": chain_id,
        "hypothesis_id": hypothesis_id,
        "path": str(path),
    }


def write_partial_index(index: list[dict[str, Any]], reports_dir: Path, final: bool = False) -> None:
    """Persist an index checkpoint so reports remain discoverable during processing."""
    target = reports_dir / ("index.json" if final else "index.partial.json")
    target.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def write_audit(catalogs: dict[str, ServiceCatalog], reports_dir: Path) -> None:
    """Write current chain/hypothesis snapshots for all services."""
    audit_dir = reports_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    for service, catalog in catalogs.items():
        prefix = safe_name(service)
        catalog.store.chains_frame().to_csv(audit_dir / f"{prefix}_chains.csv", index=False)
        catalog.store.hypotheses_frame().to_csv(audit_dir / f"{prefix}_hypotheses.csv", index=False)


# --------------------------------------------------------------------------- #
# Streaming execution
# --------------------------------------------------------------------------- #

def process_incrementally(
    logs: pd.DataFrame,
    processor: RuntimeStreamProcessor,
    catalogs: dict[str, ServiceCatalog],
    reports_dir: Path,
    progress_every: int,
    checkpoint_every: int,
) -> list[dict[str, Any]]:
    """
    Process events one by one and export every produced report immediately.

    This avoids collecting reports in memory and provides visible progress for
    large runtime datasets.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_index: list[dict[str, Any]] = []
    report_sequence = 0
    started_at = time.monotonic()

    ordered = logs.copy()
    ordered["__original_index"] = range(len(ordered))
    ordered = ordered.sort_values(["timestamp", "__original_index"], kind="stable")

    for processed_count, (_, row) in enumerate(ordered.iterrows(), start=1):
        call_node_id = None
        if "call_node_id" in ordered.columns and pd.notna(row.get("call_node_id")):
            call_node_id = str(row["call_node_id"])

        event = RuntimeEvent(
            index=int(row["__original_index"]),
            timestamp=int(row["timestamp"]),
            service=str(row["service"]),
            bucket=str(row["bucket"]),
            message=str(row["message"]),
            observed_call_node_id=call_node_id,
        )
        reports = processor.process(event)

        for report in reports:
            report_index.append(
                write_report_incrementally(report, reports_dir, report_sequence)
            )
            report_sequence += 1

        if progress_every > 0 and processed_count % progress_every == 0:
            elapsed = max(time.monotonic() - started_at, 1e-9)
            logger.info(
                "Progress: logs=%d/%d reports=%d rate=%.1f logs/s",
                processed_count,
                len(ordered),
                report_sequence,
                processed_count / elapsed,
            )
            write_partial_index(report_index, reports_dir, final=False)

        if checkpoint_every > 0 and processed_count % checkpoint_every == 0:
            write_audit(catalogs, reports_dir)

    return report_index


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trie-first multi-service runtime analyzer with incremental report export."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("output"))
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("output/runtime_reports"))

    parser.add_argument("--timestamp-column", default=None)
    parser.add_argument("--service-column", default=None)
    parser.add_argument("--message-column", default=None)
    parser.add_argument("--bucket-column", default=None)
    parser.add_argument("--call-node-column", default=None)

    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--time-gap-sec", type=int, default=30)
    parser.add_argument("--max-active-chains", type=int, default=128)
    parser.add_argument("--max-hypotheses-per-chain", type=int, default=16)
    parser.add_argument("--max-total-active-hypotheses", type=int, default=512)
    parser.add_argument("--max-call-depth", type=int, default=3)
    parser.add_argument("--emit-incomplete", action="store_true")

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Print progress and save index.partial.json every N events",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save audit chain/hypothesis CSV snapshots every N events",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    catalogs = load_catalogs(args.artifacts_dir, args)
    logs = prepare_logs(pd.read_csv(args.logs), args)
    logger.info("Loaded runtime log events: %d", len(logs))

    processor = RuntimeStreamProcessor(catalogs, max_call_depth=args.max_call_depth)
    report_index = process_incrementally(
        logs=logs,
        processor=processor,
        catalogs=catalogs,
        reports_dir=args.reports_dir,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
    )

    if args.emit_incomplete:
        for report in processor.finalize():
            report_index.append(
                write_report_incrementally(report, args.reports_dir, len(report_index))
            )

    write_partial_index(report_index, args.reports_dir, final=True)
    write_audit(catalogs, args.reports_dir)
    logger.info("Done: logs=%d reports=%d output=%s", len(logs), len(report_index), args.reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
