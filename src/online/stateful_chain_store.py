"""NFA-style runtime chain store for StaticLogFSM.

The store reconstructs *possible* execution chains from an uncorrelated log
stream. It does not greedily assign one log to one CFG path. Each active chain
keeps a frontier -- a set of FSM states still compatible with all logs assigned
to that chain. A runtime event advances every compatible frontier transition;
branches with the same observed template remain alternatives in that frontier.

This is intentionally independent from CPG construction: give it the already
built ``Dict[str, StaticLogFSM]`` and a DataFrame with timestamp/message.
"""

from __future__ import annotations

import re
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from src.offline.finite_state_machine import LogTransition, StaticLogFSM


@dataclass(frozen=True)
class TransitionMatch:
    fsm_key: str
    transition_id: str
    source_state: str
    target_state: str
    log_call_node_id: str
    template: str
    score: float


@dataclass
class ChainLog:
    index: int
    timestamp: Optional[int]
    bucket: str
    message: str
    transition_ids: Tuple[str, ...]
    score: float


@dataclass
class ActiveChain:
    chain_id: int
    fsm_key: str
    frontier: Set[str]
    created_at: Optional[int]
    last_timestamp: Optional[int]
    score: float = 0.0
    status: str = "active"  # active | completed | expired | pruned
    termination_reason: str = ""
    logs: List[ChainLog] = field(default_factory=list)

    @property
    def current_states(self) -> Tuple[str, ...]:
        return tuple(sorted(self.frontier))


@dataclass(frozen=True)
class ChainEvent:
    index: int
    timestamp: Optional[int]
    bucket: str
    message: str
    chain_id: Optional[int]
    fsm_key: Optional[str]
    verdict: str  # new_chain | continue | completed | unknown
    source_states: Tuple[str, ...]
    target_states: Tuple[str, ...]
    transition_ids: Tuple[str, ...]
    templates: Tuple[str, ...]
    score: float


def normalize_text(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"[^\w*]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _template_tokens(template: str) -> List[str]:
    return normalize_text(template.replace("<*>", " <*> ")).split()


def template_score(message: str, template: str) -> float:
    """Score a log-template match without requiring a trie at runtime.

    A wildcard ``<*>`` consumes zero or more message tokens. Static template
    tokens must appear in their original order. Exact static-token coverage is
    promoted to 1.0; otherwise Jaccard is a conservative fallback.
    """
    message_tokens = normalize_text(message).split()
    template_tokens = _template_tokens(template)
    static = [token for token in template_tokens if token != "<*>"]
    if not message_tokens or not static:
        return 0.0

    position = 0
    matched = 0
    for token in static:
        while position < len(message_tokens) and message_tokens[position] != token:
            position += 1
        if position == len(message_tokens):
            break
        matched += 1
        position += 1

    if matched == len(static):
        # Prefer templates whose static part explains a larger share of message.
        specificity = len(static) / max(len(message_tokens), len(static))
        return 0.90 + 0.10 * specificity

    message_set, template_set = set(message_tokens), set(static)
    return len(message_set & template_set) / len(message_set | template_set)


class ChainStore:
    """Keep bounded, competing NFA hypotheses over a runtime log stream."""

    def __init__(
        self,
        fsms: Dict[str, StaticLogFSM],
        threshold: float = 0.55,
        time_gap_sec: int = 30,
        max_active_chains: int = 512,
        max_frontier_states: int = 32,
        allow_new_chains: bool = True,
    ) -> None:
        self.fsms = {key: fsm for key, fsm in fsms.items() if fsm.transitions}
        self.threshold = threshold
        self.time_gap_sec = time_gap_sec
        self.max_active_chains = max_active_chains
        self.max_frontier_states = max_frontier_states
        self.allow_new_chains = allow_new_chains
        self.active_chains: Dict[int, ActiveChain] = {}
        self.history: List[ChainEvent] = []
        self._next_chain_id = 0
        self._outgoing: Dict[str, Dict[str, List[LogTransition]]] = {
            key: self._build_outgoing(fsm) for key, fsm in self.fsms.items()
        }
        self._starts: Dict[str, List[LogTransition]] = {
            key: [
                transition
                for start_state in fsm.start_states
                for transition in self._outgoing[key].get(start_state, [])
            ]
            for key, fsm in self.fsms.items()
        }

    @staticmethod
    def _build_outgoing(fsm: StaticLogFSM) -> Dict[str, List[LogTransition]]:
        outgoing: Dict[str, List[LogTransition]] = defaultdict(list)
        for transition in fsm.transitions:
            outgoing[transition.source_segment_id].append(transition)
        return dict(outgoing)

    def _expire(self, timestamp: Optional[int]) -> None:
        if timestamp is None:
            return
        for chain in self.active_chains.values():
            if chain.status != "active" or chain.last_timestamp is None:
                continue
            if timestamp - chain.last_timestamp > self.time_gap_sec:
                chain.status = "expired"
                chain.termination_reason = "timeout"

    def _matches(
        self,
        fsm_key: str,
        candidates: Iterable[LogTransition],
        message: str,
        observed_call_node_id: Optional[str],
    ) -> List[TransitionMatch]:
        result: List[TransitionMatch] = []
        for transition in candidates:
            if observed_call_node_id and str(transition.log_call_node_id) != str(
                observed_call_node_id
            ):
                continue
            score = (
                1.0
                if observed_call_node_id
                else template_score(message, transition.template)
            )
            if score < self.threshold:
                continue
            result.append(
                TransitionMatch(
                    fsm_key=fsm_key,
                    transition_id=transition.id,
                    source_state=transition.source_segment_id,
                    target_state=transition.target_segment_id,
                    log_call_node_id=transition.log_call_node_id,
                    template=transition.template,
                    score=score,
                )
            )
        return result

    def _advance_chain(
        self,
        chain: ActiveChain,
        index: int,
        timestamp: Optional[int],
        bucket: str,
        message: str,
        observed_call_node_id: Optional[str],
    ) -> Optional[ChainEvent]:
        candidates = [
            transition
            for state_id in chain.frontier
            for transition in self._outgoing[chain.fsm_key].get(state_id, [])
        ]
        matches = self._matches(
            chain.fsm_key, candidates, message, observed_call_node_id
        )
        if not matches:
            return None

        # Preserve all compatible NFA destinations. The cap is a safety valve
        # for very generic templates, ordered by matching confidence.
        matches.sort(key=lambda item: item.score, reverse=True)
        selected = matches[: self.max_frontier_states]
        previous = chain.current_states
        chain.frontier = {item.target_state for item in selected}
        chain.last_timestamp = timestamp
        chain.score += max(item.score for item in selected)
        transition_ids = tuple(dict.fromkeys(item.transition_id for item in selected))
        templates = tuple(dict.fromkeys(item.template for item in selected))
        chain.logs.append(
            ChainLog(
                index,
                timestamp,
                bucket,
                message,
                transition_ids,
                max(item.score for item in selected),
            )
        )

        fsm = self.fsms[chain.fsm_key]
        terminal = all(state_id in fsm.terminals for state_id in chain.frontier)
        verdict = "completed" if terminal else "continue"
        if terminal:
            chain.status = "completed"
            chain.termination_reason = "terminal_state"

        return ChainEvent(
            index=index,
            timestamp=timestamp,
            bucket=bucket,
            message=message,
            chain_id=chain.chain_id,
            fsm_key=chain.fsm_key,
            verdict=verdict,
            source_states=previous,
            target_states=chain.current_states,
            transition_ids=transition_ids,
            templates=templates,
            score=max(item.score for item in selected),
        )

    def _start_chain(
        self,
        fsm_key: str,
        matches: List[TransitionMatch],
        index: int,
        timestamp: Optional[int],
        bucket: str,
        message: str,
    ) -> ChainEvent:
        matches.sort(key=lambda item: item.score, reverse=True)
        selected = matches[: self.max_frontier_states]
        chain_id = self._next_chain_id
        self._next_chain_id += 1
        frontier = {item.target_state for item in selected}
        fsm = self.fsms[fsm_key]
        terminal = all(state_id in fsm.terminals for state_id in frontier)
        chain = ActiveChain(
            chain_id=chain_id,
            fsm_key=fsm_key,
            frontier=frontier,
            created_at=timestamp,
            last_timestamp=timestamp,
            score=max(item.score for item in selected),
            status="completed" if terminal else "active",
            termination_reason="terminal_state" if terminal else "",
        )
        transition_ids = tuple(dict.fromkeys(item.transition_id for item in selected))
        templates = tuple(dict.fromkeys(item.template for item in selected))
        chain.logs.append(
            ChainLog(index, timestamp, bucket, message, transition_ids, chain.score)
        )
        self.active_chains[chain_id] = chain

        return ChainEvent(
            index=index,
            timestamp=timestamp,
            bucket=bucket,
            message=message,
            chain_id=chain_id,
            fsm_key=fsm_key,
            verdict="completed" if terminal else "new_chain",
            source_states=tuple(sorted({item.source_state for item in selected})),
            target_states=chain.current_states,
            transition_ids=transition_ids,
            templates=templates,
            score=chain.score,
        )

    def _prune(self) -> None:
        active = [
            chain for chain in self.active_chains.values() if chain.status == "active"
        ]
        if len(active) <= self.max_active_chains:
            return
        # Keep best-supported recent hypotheses; mark the rest instead of
        # deleting them, so the experiment remains auditable.
        active.sort(
            key=lambda chain: (chain.score, chain.last_timestamp or -1), reverse=True
        )
        for chain in active[self.max_active_chains :]:
            chain.status = "pruned"
            chain.termination_reason = "active_chain_limit"

    def process(
        self,
        message: str,
        timestamp: Optional[int] = None,
        bucket: str = "",
        index: int = -1,
        observed_call_node_id: Optional[str] = None,
    ) -> List[ChainEvent]:
        """Consume one log and return every compatible chain hypothesis.

        ``observed_call_node_id`` is optional. If a prior template matcher has
        already resolved a CPG CALL node, provide it to eliminate lexical
        ambiguity and use an exact static-event match.
        """
        self._expire(timestamp)
        events: List[ChainEvent] = []

        for chain in list(self.active_chains.values()):
            if chain.status != "active":
                continue
            event = self._advance_chain(
                chain, index, timestamp, bucket, message, observed_call_node_id
            )
            if event is not None:
                events.append(event)

        # A log can be the first event of a new request even when it also
        # extends another active hypothesis. Keep both alternatives.
        if self.allow_new_chains:
            for fsm_key, transitions in self._starts.items():
                matches = self._matches(
                    fsm_key, transitions, message, observed_call_node_id
                )
                if matches:
                    events.append(
                        self._start_chain(
                            fsm_key, matches, index, timestamp, bucket, message
                        )
                    )

        if not events:
            events.append(
                ChainEvent(
                    index=index,
                    timestamp=timestamp,
                    bucket=bucket,
                    message=message,
                    chain_id=None,
                    fsm_key=None,
                    verdict="unknown",
                    source_states=(),
                    target_states=(),
                    transition_ids=(),
                    templates=(),
                    score=0.0,
                )
            )

        self.history.extend(events)
        self._prune()
        return events

    def process_frame(
        self,
        logs: pd.DataFrame,
        timestamp_column: str = "timestamp",
        message_column: str = "message",
        bucket_column: str = "bucket",
        call_node_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Process a log DataFrame chronologically and return hypothesis rows."""
        rows: List[Dict[str, Any]] = []
        ordered = logs.sort_values(timestamp_column).reset_index(drop=True)
        for index, row in ordered.iterrows():
            raw_timestamp = row.get(timestamp_column)
            timestamp = int(raw_timestamp) if pd.notna(raw_timestamp) else None
            bucket = str(row.get(bucket_column, ""))
            message = str(row.get(message_column, ""))
            call_node = None
            if call_node_column and pd.notna(row.get(call_node_column)):
                call_node = str(row[call_node_column])
            for event in self.process(message, timestamp, bucket, index, call_node):
                rows.append(
                    {
                        "log_index": event.index,
                        "timestamp": event.timestamp,
                        "bucket": event.bucket,
                        "message": event.message,
                        "chain_id": event.chain_id,
                        "entrypoint": event.fsm_key,
                        "verdict": event.verdict,
                        "source_states": " | ".join(event.source_states),
                        "target_states": " | ".join(event.target_states),
                        "transition_ids": " | ".join(event.transition_ids),
                        "templates": " | ".join(event.templates),
                        "score": event.score,
                    }
                )
        return pd.DataFrame(rows)

    def chains_frame(self) -> pd.DataFrame:
        rows = []
        for chain in self.active_chains.values():
            rows.append(
                {
                    "chain_id": chain.chain_id,
                    "entrypoint": chain.fsm_key,
                    "status": chain.status,
                    "termination_reason": chain.termination_reason,
                    "created_at": chain.created_at,
                    "last_timestamp": chain.last_timestamp,
                    "score": chain.score,
                    "log_count": len(chain.logs),
                    "frontier_size": len(chain.frontier),
                    "frontier": " | ".join(chain.current_states),
                }
            )
        return (
            pd.DataFrame(rows).sort_values(["status", "score"], ascending=[True, False])
            if rows
            else pd.DataFrame()
        )


def classify_logs_with_chain_store(
    logs: pd.DataFrame,
    fsms: Dict[str, StaticLogFSM],
    threshold: float = 0.55,
    time_gap_sec: int = 30,
    max_active_chains: int = 512,
    call_node_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, ChainStore]:
    """Notebook-friendly one-shot wrapper.

    Returns ``(events_df, chains_df, store)``. Keep ``store`` when you want to
    inspect individual chain logs or continue streaming more messages.
    """
    store = ChainStore(
        fsms,
        threshold=threshold,
        time_gap_sec=time_gap_sec,
        max_active_chains=max_active_chains,
    )
    events = store.process_frame(logs, call_node_column=call_node_column)
    return events, store.chains_frame(), store


# Compatibility alias for the current notebook naming convention.
classifylogswithchainstore = classify_logs_with_chain_store


def extract_logs(
    dataset_path: str,
    service: str,
    start_time: int,
    end_time: int,
) -> List[Tuple[int, str, str]]:
    logs: List[Tuple[int, str, str]] = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    timestamp = int(float(row[0]))
                except (ValueError, IndexError):
                    continue
                log_service, message = row[1].strip(), row[2].strip()
                if (
                    log_service.lower() == service.lower()
                    and start_time <= timestamp <= end_time
                ):
                    logs.append((timestamp, f"{log_service}@{timestamp}", message))
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {dataset_path}")
    return logs


def logs_to_df(logs: List[Tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(logs, columns=["timestamp", "bucket", "message"])
