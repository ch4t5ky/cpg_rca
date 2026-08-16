"""
stream_processor.py
===================

Multi-service runtime stream orchestration.

The processor routes each RuntimeEvent to the explicit-hypothesis store of its
service. Completed hypotheses are emitted once and expanded into LLM reports;
unknown events are emitted as structurally unexplained reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from src.online.hypothesis_report import build_llm_report, build_unexplained_report
from src.online.stateful_chain_store import ExplicitHypothesisStore


@dataclass(frozen=True)
class RuntimeEvent:
    """Normalized multi-service runtime log event."""

    index: int
    timestamp: Optional[int]
    service: str
    bucket: str
    message: str
    observed_call_node_id: Optional[str] = None


@dataclass
class ServiceCatalog:
    """Offline artifacts and online matcher state for one service."""

    fsms: Dict[str, Any]
    flows: Dict[str, dict[str, Any]]
    store: ExplicitHypothesisStore


class RuntimeStreamProcessor:
    """
    Route a chronological multi-service log stream into service-local matchers.

    A known service name prevents cross-service lexical matching. Logs with an
    unknown service are emitted as reports but are not force-matched against
    every catalog by default, avoiding broad false-positive hypotheses.
    """

    def __init__(self, catalogs: Dict[str, ServiceCatalog], max_call_depth: int = 3) -> None:
        self.catalogs = {service.lower(): catalog for service, catalog in catalogs.items()}
        self.max_call_depth = max_call_depth
        self.reported_hypotheses: set[tuple[str, int]] = set()

    def _reports_for_completed(self, service: str, catalog: ServiceCatalog) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for chain in catalog.store.active_chains.values():
            fsm = catalog.fsms.get(chain.fsm_key)
            flow = catalog.flows.get(chain.fsm_key)
            if fsm is None or flow is None:
                continue

            for hypothesis in chain.completed_hypotheses:
                key = (service, hypothesis.hypothesis_id)
                if key in self.reported_hypotheses:
                    continue
                reports.append(build_llm_report(
                    service=service,
                    chain=chain,
                    hypothesis=hypothesis,
                    fsm=fsm,
                    flow=flow,
                    max_call_depth=self.max_call_depth,
                ))
                self.reported_hypotheses.add(key)
        return reports

    def process(self, event: RuntimeEvent) -> list[dict[str, Any]]:
        """Process one event and return zero or more newly generated reports."""
        service_key = event.service.lower()
        catalog = self.catalogs.get(service_key)
        if catalog is None:
            return [{
                "report_type": "unknown_service_event",
                "service": event.service,
                "runtime_event": {
                    "index": event.index,
                    "timestamp": event.timestamp,
                    "bucket": event.bucket,
                    "message": event.message,
                },
                "classification": {
                    "reason": "No offline FSM/FLOW catalog was loaded for the event service.",
                },
            }]

        events = catalog.store.process(
            message=event.message,
            timestamp=event.timestamp,
            bucket=event.bucket,
            index=event.index,
            observed_call_node_id=event.observed_call_node_id,
        )

        reports = self._reports_for_completed(event.service, catalog)
        for match_event in events:
            if match_event.verdict == "unknown":
                reports.append(build_unexplained_report(event.service, match_event))
        return reports

    def process_frame(
        self,
        logs: pd.DataFrame,
        timestamp_column: str = "timestamp",
        service_column: str = "service",
        message_column: str = "message",
        bucket_column: str = "bucket",
        call_node_column: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Process a DataFrame ordered by timestamp and original row order."""
        required = {timestamp_column, service_column, message_column}
        missing = required - set(logs.columns)
        if missing:
            raise ValueError(f"Stream DataFrame is missing required columns: {sorted(missing)}")

        ordered = logs.copy()
        ordered["__original_index"] = range(len(ordered))
        ordered = ordered.sort_values([timestamp_column, "__original_index"], kind="stable")

        reports: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            raw_timestamp = row.get(timestamp_column)
            timestamp = int(raw_timestamp) if pd.notna(raw_timestamp) else None
            call_node_id = None
            if call_node_column and call_node_column in row and pd.notna(row.get(call_node_column)):
                call_node_id = str(row.get(call_node_column))

            event = RuntimeEvent(
                index=int(row["__original_index"]),
                timestamp=timestamp,
                service=str(row.get(service_column, "")),
                bucket=str(row.get(bucket_column, "")),
                message=str(row.get(message_column, "")),
                observed_call_node_id=call_node_id,
            )
            reports.extend(self.process(event))
        return reports

    def finalize(self) -> list[dict[str, Any]]:
        """
        Return snapshots of unfinished hypotheses without falsely marking them completed.

        These reports are useful at end-of-stream boundaries or experiment
        windows; they preserve paths that have not yet reached terminal states.
        """
        reports: list[dict[str, Any]] = []
        for service, catalog in self.catalogs.items():
            for chain in catalog.store.active_chains.values():
                for hypothesis in chain.active_hypotheses:
                    reports.append({
                        "report_type": "incomplete_execution_hypothesis",
                        "service": service,
                        "chain_id": chain.chain_id,
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "entrypoint": chain.fsm_key,
                        "status": hypothesis.status,
                        "score": hypothesis.score,
                        "current_state": hypothesis.current_state,
                        "transition_ids": list(hypothesis.transition_ids),
                        "state_path": list(hypothesis.state_path),
                    })
        return reports
