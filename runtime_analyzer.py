"""
runtime_analyzer.py
===================

Analyze runtime logs in one manually specified injection-centered interval and
write exactly one LLM-ready incident artifact.

The analyzer loads paired Trie/FSM/FLOW artifacts, filters input CSV logs to:

    [inject_time - before_sec, inject_time + after_sec]

and writes one output file:

    <output>/incident_context.json

Each completed chain preserves its runtime-constrained `flow_slice` directly.
That FLOW slice contains the original semantic-unit fields (including
`call_code`, `caller_full_name`, `callee_full_name`, `line`, `depth`, and
`call_index`) produced by flow_slicer.py.

Usage:
    python3.12 runtime_analyzer.py \\
      --artifacts-dir output \\
      --logs dataset/logs.csv \\
      --output output/runtime_reports/incident_context.json \\
      --service-column container_name \\
      --inject-time 1731903974

    python3.12 runtime_analyzer.py \\
      --artifacts-dir output \\
      --logs dataset/logs.csv \\
      --output output/runtime_reports/incident_context.json \\
      --service-column container_name \\
      --inject-time 1731903974 \\
      --before-sec 30 \\
      --after-sec 120 \\
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
from src.online.stream_processor import RuntimeEvent, RuntimeStreamProcessor, ServiceCatalog
from src.online.trie_matcher import ServiceTrieMatcher

logger = logging.getLogger("runtime_analyzer")

_TIMESTAMP_ALIASES = ("timestamp", "time", "ts", "unix_timestamp", "event_time")
_SERVICE_ALIASES = (
    "service",
    "microservice",
    "container",
    "container_name",
    "containername",
    "component",
    "application",
    "app",
    "source",
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
    """Restore one StaticLogFSM from a builder.py .fsm.json artifact."""
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
    """Load paired Trie/FSM/FLOW artifacts for one service directory."""
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
        logger.warning(
            "Service=%s has no trie.json; lexical FSM fallback will be used",
            service_dir.name,
        )

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
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts_dir}")

    catalogs: dict[str, ServiceCatalog] = {}
    for service_dir in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        catalog = load_service_catalog(service_dir, args)
        if catalog is None:
            continue

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
# CSV normalization and incident-window filtering
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
        if not candidate:
            continue
        resolved = normalized.get(normalize_header(candidate))
        if resolved is not None:
            return resolved

    if not required:
        return None

    raise ValueError(
        f"Cannot resolve required '{logical_name}' column. "
        f"Actual CSV columns: {list(map(str, columns))}. "
        f"Pass --{logical_name}-column explicitly."
    )


def prepare_logs(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Normalize an arbitrary runtime CSV into canonical stream columns."""
    timestamp_column = resolve_column(raw.columns, args.timestamp_column, _TIMESTAMP_ALIASES, "timestamp")
    service_column = resolve_column(raw.columns, args.service_column, _SERVICE_ALIASES, "service")
    message_column = resolve_column(raw.columns, args.message_column, _MESSAGE_ALIASES, "message")
    bucket_column = resolve_column(raw.columns, args.bucket_column, _BUCKET_ALIASES, "bucket", required=False)
    call_node_column = resolve_column(
        raw.columns,
        args.call_node_column,
        _CALL_NODE_ALIASES,
        "call-node",
        required=False,
    )

    logs = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(raw[timestamp_column], errors="coerce"),
            "service": raw[service_column].astype(str).str.strip(),
            "message": raw[message_column].astype(str),
        }
    )
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


def filter_incident_window(
    logs: pd.DataFrame,
    inject_time: int,
    before_sec: int,
    after_sec: int,
) -> tuple[pd.DataFrame, int, int]:
    """Keep only rows in [inject_time - before_sec, inject_time + after_sec]."""
    if before_sec < 0 or after_sec < 0:
        raise ValueError("--before-sec and --after-sec must be non-negative.")

    start_time = inject_time - before_sec
    end_time = inject_time + after_sec
    filtered = logs.loc[
        logs["timestamp"].between(start_time, end_time, inclusive="both")
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No runtime logs fall inside the requested incident window: "
            f"[{start_time}, {end_time}], inject={inject_time}."
        )

    return filtered.reset_index(drop=True), start_time, end_time


# --------------------------------------------------------------------------- #
# Stream execution and single incident artifact construction
# --------------------------------------------------------------------------- #

def process_window(
    logs: pd.DataFrame,
    processor: RuntimeStreamProcessor,
    progress_every: int,
) -> list[dict[str, Any]]:
    """Process selected logs in timestamp order and retain reports only in memory."""
    ordered = logs.copy()
    ordered["__original_index"] = range(len(ordered))
    ordered = ordered.sort_values(["timestamp", "__original_index"], kind="stable")

    reports: list[dict[str, Any]] = []
    total = len(ordered)

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
        reports.extend(processor.process(event))

        if progress_every > 0 and processed_count % progress_every == 0:
            logger.info("Progress: logs=%d/%d reports=%d", processed_count, total, len(reports))

    return reports


def report_identity(report: dict[str, Any]) -> tuple[Any, ...]:
    """Deduplicate reports emitted through multiple compatible runtime paths."""
    return (
        report.get("report_type"),
        report.get("service"),
        report.get("chain", {}).get("chain_id", report.get("chain_id")),
        report.get("hypothesis", {}).get("hypothesis_id", report.get("hypothesis_id")),
        report.get("runtime_event", {}).get("index"),
    )


def deduplicate_reports(reports: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first emitted instance of each completed or unexplained report."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for report in reports:
        key = report_identity(report)
        if key in seen:
            continue
        seen.add(key)
        unique.append(report)

    return unique


def completed_chain(report: dict[str, Any]) -> dict[str, Any]:
    """Preserve the report's runtime evidence and full constrained FLOW slice."""
    chain = report.get("chain", {})
    hypothesis = report.get("hypothesis", {})
    entrypoint = report.get("entrypoint", {})

    return {
        "chain_id": chain.get("chain_id"),
        "service": report.get("service"),
        "entrypoint": entrypoint.get("full_name"),
        "status": hypothesis.get("status"),
        "score": hypothesis.get("cumulative_score"),
        "time": report.get("time_window", {}),
        "runtime_logs": report.get("runtime_evidence", []),
        "flow": report.get("flow_slice", {}),
        "ambiguity": report.get("ambiguity", {}),
    }


def sort_key_timestamp(item: dict[str, Any]) -> tuple[bool, int, str]:
    """Stable sort key for timeline entries with optional timestamps."""
    timestamp = item.get("timestamp")
    return (
        timestamp is None,
        int(timestamp) if timestamp is not None else 0,
        str(item.get("service", "")),
    )


def build_incident_context(
    reports: list[dict[str, Any]],
    inject_time: int,
    start_time: int,
    end_time: int,
) -> dict[str, Any]:
    """Build one LLM artifact without compacting FLOW semantic-unit code."""
    completed_reports = [
        report
        for report in reports
        if report.get("report_type") == "completed_execution_hypothesis"
    ]
    unexplained_reports = [
        report
        for report in reports
        if report.get("report_type") in {
            "structurally_unexplained_event",
            "unknown_service_event",
        }
    ]
    incomplete_reports = [
        report
        for report in reports
        if report.get("report_type") == "incomplete_execution_hypothesis"
    ]

    chains = [completed_chain(report) for report in completed_reports]
    chains.sort(
        key=lambda chain: (
            chain.get("time", {}).get("start") is None,
            chain.get("time", {}).get("start") or 0,
            str(chain.get("service", "")),
            str(chain.get("chain_id", "")),
        )
    )

    timeline: list[dict[str, Any]] = []
    for chain in chains:
        for event in chain["runtime_logs"]:
            timeline.append(
                {
                    "timestamp": event.get("timestamp"),
                    "service": chain.get("service"),
                    "chain_id": chain.get("chain_id"),
                    "bucket": event.get("bucket"),
                    "message": event.get("message"),
                    "structurally_unexplained": False,
                }
            )

    for report in unexplained_reports:
        event = report.get("runtime_event", {})
        timeline.append(
            {
                "timestamp": event.get("timestamp"),
                "service": report.get("service"),
                "chain_id": None,
                "bucket": event.get("bucket"),
                "message": event.get("message"),
                "structurally_unexplained": True,
                "reason": report.get("classification", {}).get("reason"),
            }
        )

    timeline.sort(key=sort_key_timestamp)

    return {
        "schema_version": "3.0",
        "description": (
            "Injection-centered runtime analysis. Every chain contains a "
            "runtime-constrained FLOW slice with original code snippets in "
            "flow.methods[*].nodes[*].call_code."
        ),
        "incident_window": {
            "inject_time": inject_time,
            "start_timestamp": start_time,
            "end_timestamp": end_time,
            "before_sec": inject_time - start_time,
            "after_sec": end_time - inject_time,
        },
        "analysis_window": {
            "event_count": len(timeline),
            "completed_chain_count": len(chains),
            "services": sorted(
                {
                    item.get("service")
                    for item in timeline
                    if item.get("service")
                }
            ),
        },
        "chains": chains,
        "timeline": timeline,
        "unexplained_events": unexplained_reports,
        "incomplete_hypotheses": incomplete_reports,
        "metadata": {
            "source_report_count": len(reports),
            "completed_chain_count": len(chains),
            "unexplained_event_count": len(unexplained_reports),
            "incomplete_hypothesis_count": len(incomplete_reports),
            "flow_contract": (
                "chain.flow is the original runtime-constrained FLOW slice; "
                "do not replace semantic units with method-only summaries."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one injection-centered log window and write one incident JSON artifact."
    )

    parser.add_argument("--artifacts-dir", type=Path, default=Path("output"))
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/runtime_reports/incident_context.json"),
        help="Single incident JSON output path",
    )

    parser.add_argument(
        "--inject-time",
        type=int,
        required=True,
        help="Unix timestamp of the manually supplied fault injection time",
    )
    parser.add_argument(
        "--before-sec",
        type=int,
        default=10,
        help="Seconds retained before injection (default: 10)",
    )
    parser.add_argument(
        "--after-sec",
        type=int,
        default=60,
        help="Seconds retained after injection (default: 60)",
    )

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
    parser.add_argument(
        "--emit-incomplete",
        action="store_true",
        help="Include end-of-window incomplete hypotheses in incident_context.json",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000,
        help="Print progress every N processed incident-window logs",
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

    if not args.logs.is_file():
        raise FileNotFoundError(f"Runtime log CSV does not exist: {args.logs}")

    catalogs = load_catalogs(args.artifacts_dir, args)
    all_logs = prepare_logs(pd.read_csv(args.logs), args)
    logs, window_start, window_end = filter_incident_window(
        logs=all_logs,
        inject_time=args.inject_time,
        before_sec=args.before_sec,
        after_sec=args.after_sec,
    )

    logger.info(
        "Incident window: inject=%d range=[%d, %d] before=%ds after=%ds logs=%d/%d",
        args.inject_time,
        window_start,
        window_end,
        args.before_sec,
        args.after_sec,
        len(logs),
        len(all_logs),
    )

    processor = RuntimeStreamProcessor(catalogs, max_call_depth=args.max_call_depth)
    reports = process_window(
        logs=logs,
        processor=processor,
        progress_every=args.progress_every,
    )

    if args.emit_incomplete:
        reports.extend(processor.finalize())

    reports = deduplicate_reports(reports)
    context = build_incident_context(
        reports=reports,
        inject_time=args.inject_time,
        start_time=window_start,
        end_time=window_end,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Done: logs=%d reports=%d chains=%d unexplained=%d output=%s",
        len(logs),
        len(reports),
        context["metadata"]["completed_chain_count"],
        context["metadata"]["unexplained_event_count"],
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
