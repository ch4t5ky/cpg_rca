"""
stream_processor.py
===================

Multi-service runtime stream orchestration with Trie-first matching.

For every runtime event:
    message -> service Log Trie -> candidate CPG CALL ids -> FSM hypotheses

Trie ambiguity is preserved as a set of CPG CALL candidates. The explicit
hypothesis store processes each runtime event once and creates child hypotheses
only for FSM transitions that are both Trie-compatible and structurally valid
from the hypothesis current state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from src.online.hypothesis_report import build_llm_report, build_unexplained_report
from src.online.stateful_chain_store import ExplicitHypothesisStore
from src.online.trie_matcher import ServiceTrieMatcher


@dataclass(frozen=True)
class RuntimeEvent:
    """Canonical runtime event routed to one service-local online matcher."""

    index: int
    timestamp: Optional[int]
    service: str
    bucket: str
    message: str
    observed_call_node_id: Optional[str] = None


@dataclass
class ServiceCatalog:
    """All offline artifacts and mutable online state for one service."""

    fsms: Dict[str, Any]
    flows: Dict[str, dict[str, Any]]
    store: ExplicitHypothesisStore
    trie_matcher: Optional[ServiceTrieMatcher] = None


class RuntimeStreamProcessor:
    """
    Route chronological multi-service logs through Trie and FSM matching.

    A service name is used as a routing key. This avoids cross-service lexical
    matching: a frontend log is matched only against frontend Trie/FSM artifacts.
    Unknown services are emitted as reports rather than force-matched against
    every loaded catalog.
    """

    def __init__(
        self,
        catalogs: Dict[str, ServiceCatalog],
        max_call_depth: int = 3,
    ) -> None:
        self.catalogs = {
            service.lower(): catalog
            for service, catalog in catalogs.items()
        }
        self.max_call_depth = max_call_depth
        self.reported_hypotheses: set[tuple[str, int]] = set()

    def _reports_for_completed(
        self,
        service: str,
        catalog: ServiceCatalog,
    ) -> list[dict[str, Any]]:
        """Build exactly one LLM report for each newly completed hypothesis."""
        reports: list[dict[str, Any]] = []

        for chain in catalog.store.active_chains.values():
            fsm = catalog.fsms.get(chain.fsm_key)
            flow = catalog.flows.get(chain.fsm_key)
            if fsm is None or flow is None:
                continue

            for hypothesis in chain.completed_hypotheses:
                report_key = (service.lower(), hypothesis.hypothesis_id)
                if report_key in self.reported_hypotheses:
                    continue

                reports.append(
                    build_llm_report(
                        service=service,
                        chain=chain,
                        hypothesis=hypothesis,
                        fsm=fsm,
                        flow=flow,
                        max_call_depth=self.max_call_depth,
                    )
                )
                self.reported_hypotheses.add(report_key)

        return reports

    def process(self, event: RuntimeEvent) -> list[dict[str, Any]]:
        """
        Process one runtime event and return newly generated reports.

        Priority of evidence:
        1. An externally supplied observed_call_node_id is treated as exact.
        2. Otherwise service Trie returns all compatible logger CPG CALL ids.
        3. If Trie returns no candidate, the store falls back to lexical
           template matching for exploratory robustness.
        """
        catalog = self.catalogs.get(event.service.lower())
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
                    "reason": "No offline Trie/FSM/FLOW catalog was loaded for this service.",
                },
            }]

        trie_matches = []
        call_node_ids: set[str] = set()

        if event.observed_call_node_id:
            call_node_ids.add(str(event.observed_call_node_id))
        elif catalog.trie_matcher is not None:
            trie_matches = catalog.trie_matcher.match(event.message)
            call_node_ids = {
                match.call_node_id
                for match in trie_matches
            }

        # Pass the entire candidate set once. The store retains only transitions
        # valid from each current hypothesis state and avoids duplicate event
        # processing caused by calling process() separately per Trie candidate.
        hypothesis_events = catalog.store.process(
            message=event.message,
            timestamp=event.timestamp,
            bucket=event.bucket,
            index=event.index,
            observed_call_node_ids=call_node_ids or None,
        )

        reports = self._reports_for_completed(event.service, catalog)

        for hypothesis_event in hypothesis_events:
            if hypothesis_event.verdict != "unknown":
                continue

            report = build_unexplained_report(event.service, hypothesis_event)
            report["trie_candidates"] = [
                {
                    "call_node_id": match.call_node_id,
                    "template": match.template,
                    "static_score": match.static_score,
                    "score": match.score,
                }
                for match in trie_matches
            ]
            reports.append(report)

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
        """
        Process a DataFrame in timestamp order while preserving same-time order.

        Required logical columns are timestamp, service, and message. The
        runtime_analyzer.py CLI normalizes arbitrary dataset headers into these
        canonical names before calling this method.
        """
        required = {timestamp_column, service_column, message_column}
        missing = required - set(logs.columns)
        if missing:
            raise ValueError(
                f"Stream DataFrame is missing required columns: {sorted(missing)}"
            )

        ordered = logs.copy()
        ordered["__original_index"] = range(len(ordered))
        ordered = ordered.sort_values(
            [timestamp_column, "__original_index"],
            kind="stable",
        )

        reports: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            raw_timestamp = row.get(timestamp_column)
            timestamp = int(raw_timestamp) if pd.notna(raw_timestamp) else None

            call_node_id = None
            if (
                call_node_column
                and call_node_column in row
                and pd.notna(row.get(call_node_column))
            ):
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
        Return end-of-stream snapshots for hypotheses not yet terminal.

        The method does not force completion. It preserves unfinished structural
        paths as incomplete evidence for later incident analysis.
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
