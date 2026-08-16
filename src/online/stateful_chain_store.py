"""
stateful_chain_store.py
=======================

Explicit-hypothesis online matcher for StaticLogFSM.

The matcher reconstructs possible runtime executions from an uncorrelated log
stream. Unlike a compressed NFA frontier, every active PathHypothesis keeps its
own current FSM state, matched transition history, state path, runtime evidence,
and cumulative score.

Formal model
------------
For an entrypoint FSM M = (Q, T), a runtime chain contains a bounded set of
explicit hypotheses:

    H_t = {h_1, ..., h_K},
    h_k = (q_k, p_k, s_k),

where q_k is the current FSM state, p_k is the matched transition/event path,
and s_k is its cumulative match score. A runtime event expands every compatible
hypothesis. Ambiguous transitions create distinct child hypotheses instead of
being merged into one untraceable frontier.

Beam-search limits prevent combinatorial growth:
- max_hypotheses_per_chain bounds hypotheses within one candidate request;
- max_active_chains bounds simultaneously active candidate requests;
- pruned hypotheses are archived rather than discarded for auditability.

Public compatibility API
------------------------
The module keeps extract_logs(), logs_to_df(), and
classify_logs_with_chain_store() for notebook and classifier compatibility.
The latter now returns an ExplicitHypothesisStore instead of the legacy
frontier-only ChainStore.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.offline.finite_state_machine import LogTransition, StaticLogFSM


# --------------------------------------------------------------------------- #
# Runtime evidence and hypothesis model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TransitionMatch:
    """One transition whose template matches one observed runtime log."""

    fsm_key: str
    transition_id: str
    source_state: str
    target_state: str
    log_call_node_id: str
    template: str
    score: float


@dataclass(frozen=True)
class ChainLog:
    """One runtime log attached to one explicit structural hypothesis."""

    index: int
    timestamp: Optional[int]
    bucket: str
    message: str
    transition_id: str
    template: str
    source_state: str
    target_state: str
    score: float


@dataclass
class PathHypothesis:
    """
    One explicit path hypothesis h = (q, p, s) inside a candidate request chain.

    `current_state` is one state, not a merged NFA frontier. `transition_ids`
    and `state_path` preserve enough provenance to build a precise FLOW slice
    later, rather than taking the union of every compatible alternative.
    """

    hypothesis_id: int
    current_state: str
    score: float
    transition_ids: Tuple[str, ...]
    state_path: Tuple[str, ...]
    logs: List[ChainLog] = field(default_factory=list)
    status: str = "active"  # active | completed | expired | pruned | superseded
    termination_reason: str = ""

    @property
    def last_transition_id(self) -> Optional[str]:
        return self.transition_ids[-1] if self.transition_ids else None


@dataclass
class ActiveChain:
    """
    Candidate request/execution chain for one entrypoint FSM.

    Multiple hypotheses can coexist in one chain only when they originate from
    the same first runtime event and entrypoint. They differ by exact FSM path.
    """

    chain_id: int
    fsm_key: str
    created_at: Optional[int]
    last_timestamp: Optional[int]
    hypotheses: Dict[int, PathHypothesis] = field(default_factory=dict)
    status: str = "active"  # active | completed | expired | pruned
    termination_reason: str = ""

    @property
    def score(self) -> float:
        return max((hypothesis.score for hypothesis in self.hypotheses.values()), default=0.0)

    @property
    def active_hypotheses(self) -> List[PathHypothesis]:
        return [hypothesis for hypothesis in self.hypotheses.values() if hypothesis.status == "active"]

    @property
    def completed_hypotheses(self) -> List[PathHypothesis]:
        return [hypothesis for hypothesis in self.hypotheses.values() if hypothesis.status == "completed"]

    @property
    def frontier(self) -> Tuple[str, ...]:
        """Compatibility view: current states of all active hypotheses."""
        return tuple(sorted({hypothesis.current_state for hypothesis in self.active_hypotheses}))


@dataclass(frozen=True)
class HypothesisEvent:
    """Audit event emitted when one runtime log starts or advances a hypothesis."""

    index: int
    timestamp: Optional[int]
    bucket: str
    message: str
    chain_id: Optional[int]
    hypothesis_id: Optional[int]
    fsm_key: Optional[str]
    verdict: str  # new_chain | continue | completed | unknown
    source_state: Optional[str]
    target_state: Optional[str]
    transition_id: Optional[str]
    template: Optional[str]
    step_score: float
    cumulative_score: float


# Backward-compatible alias used by code that expects ChainEvent.
ChainEvent = HypothesisEvent


# --------------------------------------------------------------------------- #
# Text and template matching
# --------------------------------------------------------------------------- #


def normalize_text(value: str) -> str:
    """Normalize runtime text while preserving wildcard marker characters."""
    value = (value or "").lower()
    value = re.sub(r"[^\w*]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _template_tokens(template: str) -> List[str]:
    return normalize_text(template.replace("<*>", " <*> ")).split()


def template_score(message: str, template: str) -> float:
    """
    Score a message-template match without requiring a runtime Trie.

    Static template tokens must occur in message order. `<*>` represents a
    variable-length wildcard region. Full ordered static-token coverage returns
    a high score in [0.90, 1.00]; partial matches use Jaccard similarity as a
    conservative fallback.
    """
    message_tokens = normalize_text(message).split()
    static_tokens = [token for token in _template_tokens(template) if token != "<*>"]
    if not message_tokens or not static_tokens:
        return 0.0

    position = 0
    matched = 0
    for token in static_tokens:
        while position < len(message_tokens) and message_tokens[position] != token:
            position += 1
        if position == len(message_tokens):
            break
        matched += 1
        position += 1

    if matched == len(static_tokens):
        specificity = len(static_tokens) / max(len(message_tokens), len(static_tokens))
        return 0.90 + 0.10 * specificity

    message_set = set(message_tokens)
    template_set = set(static_tokens)
    union = message_set | template_set
    return len(message_set & template_set) / len(union) if union else 0.0


# --------------------------------------------------------------------------- #
# Explicit hypothesis store
# --------------------------------------------------------------------------- #


class ExplicitHypothesisStore:
    """
    Maintain bounded explicit FSM-path hypotheses over a runtime log stream.

    The store retains every generated hypothesis until it becomes completed,
    expires, or is pruned. Only active hypotheses participate in expansion.
    Archived hypotheses remain accessible through chains_frame() and
    hypotheses_frame() for reproducible RCA experiments.
    """

    def __init__(
        self,
        fsms: Dict[str, StaticLogFSM],
        threshold: float = 0.55,
        time_gap_sec: int = 30,
        max_active_chains: int = 512,
        max_hypotheses_per_chain: int = 32,
        max_total_active_hypotheses: int = 2048,
        allow_new_chains: bool = True,
    ) -> None:
        # FSMs without transitions cannot explain any observed logger event.
        self.fsms = {key: fsm for key, fsm in fsms.items() if fsm.transitions}
        self.threshold = threshold
        self.time_gap_sec = time_gap_sec
        self.max_active_chains = max_active_chains
        self.max_hypotheses_per_chain = max_hypotheses_per_chain
        self.max_total_active_hypotheses = max_total_active_hypotheses
        self.allow_new_chains = allow_new_chains

        self.active_chains: Dict[int, ActiveChain] = {}
        self.history: List[HypothesisEvent] = []
        self._next_chain_id = 0
        self._next_hypothesis_id = 0

        self._outgoing: Dict[str, Dict[str, List[LogTransition]]] = {
            key: self._build_outgoing(fsm)
            for key, fsm in self.fsms.items()
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

    def _new_hypothesis_id(self) -> int:
        hypothesis_id = self._next_hypothesis_id
        self._next_hypothesis_id += 1
        return hypothesis_id

    def _expire(self, timestamp: Optional[int]) -> None:
        """Expire all active hypotheses whose latest evidence is outside the time window."""
        if timestamp is None:
            return

        for chain in self.active_chains.values():
            if chain.status != "active" or chain.last_timestamp is None:
                continue
            if timestamp - chain.last_timestamp <= self.time_gap_sec:
                continue

            for hypothesis in chain.active_hypotheses:
                hypothesis.status = "expired"
                hypothesis.termination_reason = "timeout"
            chain.status = "expired"
            chain.termination_reason = "timeout"

    def _matches(
        self,
        fsm_key: str,
        candidates: Iterable[LogTransition],
        message: str,
        observed_call_node_id: Optional[str],
    ) -> List[TransitionMatch]:
        """Return all candidate transitions that meet exact-id or lexical threshold matching."""
        result: List[TransitionMatch] = []
        for transition in candidates:
            if observed_call_node_id and str(transition.log_call_node_id) != str(observed_call_node_id):
                continue

            score = 1.0 if observed_call_node_id else template_score(message, transition.template)
            if score < self.threshold:
                continue

            result.append(TransitionMatch(
                fsm_key=fsm_key,
                transition_id=transition.id,
                source_state=transition.source_segment_id,
                target_state=transition.target_segment_id,
                log_call_node_id=transition.log_call_node_id,
                template=transition.template,
                score=score,
            ))
        return result

    @staticmethod
    def _hypothesis_key(hypothesis: PathHypothesis) -> tuple:
        """
        Deduplicate only structurally equivalent candidates.

        State and complete transition sequence are used intentionally: two paths
        that reach the same state through different transitions remain distinct,
        because they imply different FLOW slices and RCA evidence.
        """
        return hypothesis.current_state, hypothesis.transition_ids

    def _archive_superseded(self, chain: ActiveChain, hypothesis: PathHypothesis) -> None:
        """Mark an expanded parent as superseded while retaining it for audit history."""
        hypothesis.status = "superseded"
        hypothesis.termination_reason = "expanded"

    def _insert_children(
        self,
        chain: ActiveChain,
        children: Iterable[PathHypothesis],
    ) -> List[PathHypothesis]:
        """Deduplicate equivalent children and retain the strongest score for each key."""
        best: Dict[tuple, PathHypothesis] = {}
        for child in children:
            key = self._hypothesis_key(child)
            previous = best.get(key)
            if previous is None or child.score > previous.score:
                best[key] = child

        inserted = list(best.values())
        for hypothesis in inserted:
            chain.hypotheses[hypothesis.hypothesis_id] = hypothesis
        return inserted

    def _enforce_chain_beam(self, chain: ActiveChain) -> None:
        """Keep the highest-scoring active hypotheses within one candidate chain."""
        active = chain.active_hypotheses
        if len(active) <= self.max_hypotheses_per_chain:
            return

        active.sort(
            key=lambda hypothesis: (hypothesis.score, len(hypothesis.transition_ids)),
            reverse=True,
        )
        for hypothesis in active[self.max_hypotheses_per_chain:]:
            hypothesis.status = "pruned"
            hypothesis.termination_reason = "beam_width"

    def _enforce_global_beam(self) -> None:
        """Bound the total number of active hypotheses across all request chains."""
        active = [
            (chain, hypothesis)
            for chain in self.active_chains.values()
            if chain.status == "active"
            for hypothesis in chain.active_hypotheses
        ]
        if len(active) <= self.max_total_active_hypotheses:
            return

        active.sort(
            key=lambda item: (item[1].score, item[0].last_timestamp or -1),
            reverse=True,
        )
        for chain, hypothesis in active[self.max_total_active_hypotheses:]:
            hypothesis.status = "pruned"
            hypothesis.termination_reason = "global_hypothesis_limit"
            self._refresh_chain_status(chain)

    def _enforce_chain_limit(self) -> None:
        """Bound concurrent chains and retain pruned chains for reproducibility."""
        active_chains = [
            chain for chain in self.active_chains.values()
            if chain.status == "active"
        ]
        if len(active_chains) <= self.max_active_chains:
            return

        active_chains.sort(key=lambda chain: (chain.score, chain.last_timestamp or -1), reverse=True)
        for chain in active_chains[self.max_active_chains:]:
            for hypothesis in chain.active_hypotheses:
                hypothesis.status = "pruned"
                hypothesis.termination_reason = "active_chain_limit"
            chain.status = "pruned"
            chain.termination_reason = "active_chain_limit"

    def _refresh_chain_status(self, chain: ActiveChain) -> None:
        """Derive chain lifecycle status from its explicit hypotheses."""
        active = chain.active_hypotheses
        if active:
            chain.status = "active"
            chain.termination_reason = ""
            return

        completed = chain.completed_hypotheses
        if completed:
            chain.status = "completed"
            chain.termination_reason = "terminal_state"
            return

        statuses = {hypothesis.status for hypothesis in chain.hypotheses.values()}
        if statuses and statuses <= {"expired"}:
            chain.status = "expired"
            chain.termination_reason = "timeout"
        elif "pruned" in statuses:
            chain.status = "pruned"
            chain.termination_reason = "hypothesis_limit"

    def _advance_hypothesis(
        self,
        chain: ActiveChain,
        parent: PathHypothesis,
        index: int,
        timestamp: Optional[int],
        bucket: str,
        message: str,
        observed_call_node_id: Optional[str],
    ) -> tuple[List[PathHypothesis], List[HypothesisEvent]]:
        """Expand one active hypothesis into one child per compatible transition."""
        transitions = self._outgoing[chain.fsm_key].get(parent.current_state, [])
        matches = self._matches(chain.fsm_key, transitions, message, observed_call_node_id)
        if not matches:
            return [], []

        fsm = self.fsms[chain.fsm_key]
        children: List[PathHypothesis] = []
        events: List[HypothesisEvent] = []

        for match in matches:
            terminal = match.target_state in fsm.terminals
            chain_log = ChainLog(
                index=index,
                timestamp=timestamp,
                bucket=bucket,
                message=message,
                transition_id=match.transition_id,
                template=match.template,
                source_state=match.source_state,
                target_state=match.target_state,
                score=match.score,
            )
            child = PathHypothesis(
                hypothesis_id=self._new_hypothesis_id(),
                current_state=match.target_state,
                score=parent.score + match.score,
                transition_ids=parent.transition_ids + (match.transition_id,),
                state_path=parent.state_path + (match.target_state,),
                logs=parent.logs + [chain_log],
                status="completed" if terminal else "active",
                termination_reason="terminal_state" if terminal else "",
            )
            children.append(child)
            events.append(HypothesisEvent(
                index=index,
                timestamp=timestamp,
                bucket=bucket,
                message=message,
                chain_id=chain.chain_id,
                hypothesis_id=child.hypothesis_id,
                fsm_key=chain.fsm_key,
                verdict="completed" if terminal else "continue",
                source_state=match.source_state,
                target_state=match.target_state,
                transition_id=match.transition_id,
                template=match.template,
                step_score=match.score,
                cumulative_score=child.score,
            ))

        self._archive_superseded(chain, parent)
        return children, events

    def _advance_chain(
        self,
        chain: ActiveChain,
        index: int,
        timestamp: Optional[int],
        bucket: str,
        message: str,
        observed_call_node_id: Optional[str],
    ) -> List[HypothesisEvent]:
        """Expand every active hypothesis of one chain for one observed event."""
        children: List[PathHypothesis] = []
        events: List[HypothesisEvent] = []

        for hypothesis in list(chain.active_hypotheses):
            new_children, new_events = self._advance_hypothesis(
                chain, hypothesis, index, timestamp, bucket, message, observed_call_node_id
            )
            children.extend(new_children)
            events.extend(new_events)

        if not children:
            return []

        inserted = self._insert_children(chain, children)
        retained_ids = {hypothesis.hypothesis_id for hypothesis in inserted}
        # Events correspond to created children; no filtering is required here,
        # because deduplication key includes complete transition history.
        chain.last_timestamp = timestamp
        self._enforce_chain_beam(chain)
        self._refresh_chain_status(chain)
        return events

    def _start_chain(
        self,
        fsm_key: str,
        matches: List[TransitionMatch],
        index: int,
        timestamp: Optional[int],
        bucket: str,
        message: str,
    ) -> tuple[ActiveChain, List[HypothesisEvent]]:
        """Create one new chain with one initial hypothesis per matching start transition."""
        chain_id = self._next_chain_id
        self._next_chain_id += 1
        chain = ActiveChain(
            chain_id=chain_id,
            fsm_key=fsm_key,
            created_at=timestamp,
            last_timestamp=timestamp,
        )

        fsm = self.fsms[fsm_key]
        children: List[PathHypothesis] = []
        events: List[HypothesisEvent] = []
        for match in matches:
            terminal = match.target_state in fsm.terminals
            chain_log = ChainLog(
                index=index,
                timestamp=timestamp,
                bucket=bucket,
                message=message,
                transition_id=match.transition_id,
                template=match.template,
                source_state=match.source_state,
                target_state=match.target_state,
                score=match.score,
            )
            hypothesis = PathHypothesis(
                hypothesis_id=self._new_hypothesis_id(),
                current_state=match.target_state,
                score=match.score,
                transition_ids=(match.transition_id,),
                state_path=(match.source_state, match.target_state),
                logs=[chain_log],
                status="completed" if terminal else "active",
                termination_reason="terminal_state" if terminal else "",
            )
            children.append(hypothesis)
            events.append(HypothesisEvent(
                index=index,
                timestamp=timestamp,
                bucket=bucket,
                message=message,
                chain_id=chain_id,
                hypothesis_id=hypothesis.hypothesis_id,
                fsm_key=fsm_key,
                verdict="completed" if terminal else "new_chain",
                source_state=match.source_state,
                target_state=match.target_state,
                transition_id=match.transition_id,
                template=match.template,
                step_score=match.score,
                cumulative_score=hypothesis.score,
            ))

        self._insert_children(chain, children)
        self._enforce_chain_beam(chain)
        self._refresh_chain_status(chain)
        self.active_chains[chain_id] = chain
        return chain, events

    def process(
        self,
        message: str,
        timestamp: Optional[int] = None,
        bucket: str = "",
        index: int = -1,
        observed_call_node_id: Optional[str] = None,
    ) -> List[HypothesisEvent]:
        """
        Consume one runtime log and return all compatible explicit hypotheses.

        The event can both extend one or more active hypotheses and create new
        request-chain hypotheses through matching FSM start transitions. An
        `unknown` event is emitted only when neither operation is possible.
        """
        self._expire(timestamp)
        events: List[HypothesisEvent] = []

        for chain in list(self.active_chains.values()):
            if chain.status != "active":
                continue
            events.extend(self._advance_chain(
                chain, index, timestamp, bucket, message, observed_call_node_id
            ))

        if self.allow_new_chains:
            for fsm_key, transitions in self._starts.items():
                matches = self._matches(fsm_key, transitions, message, observed_call_node_id)
                if matches:
                    _, new_events = self._start_chain(
                        fsm_key, matches, index, timestamp, bucket, message
                    )
                    events.extend(new_events)

        if not events:
            events.append(HypothesisEvent(
                index=index,
                timestamp=timestamp,
                bucket=bucket,
                message=message,
                chain_id=None,
                hypothesis_id=None,
                fsm_key=None,
                verdict="unknown",
                source_state=None,
                target_state=None,
                transition_id=None,
                template=None,
                step_score=0.0,
                cumulative_score=0.0,
            ))

        self.history.extend(events)
        self._enforce_global_beam()
        self._enforce_chain_limit()
        return events

    def process_frame(
        self,
        logs: pd.DataFrame,
        timestamp_column: str = "timestamp",
        message_column: str = "message",
        bucket_column: str = "bucket",
        call_node_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Process a DataFrame chronologically and return one row per emitted hypothesis event."""
        required = {timestamp_column, message_column}
        missing = required - set(logs.columns)
        if missing:
            raise ValueError(f"Log DataFrame is missing required columns: {sorted(missing)}")

        rows: List[Dict[str, Any]] = []
        ordered = logs.sort_values(timestamp_column).reset_index(drop=True)
        for index, row in ordered.iterrows():
            raw_timestamp = row.get(timestamp_column)
            timestamp = int(raw_timestamp) if pd.notna(raw_timestamp) else None
            bucket = str(row.get(bucket_column, ""))
            message = str(row.get(message_column, ""))
            call_node = None
            if call_node_column and call_node_column in row and pd.notna(row.get(call_node_column)):
                call_node = str(row[call_node_column])

            for event in self.process(message, timestamp, bucket, index, call_node):
                rows.append({
                    "log_index": event.index,
                    "timestamp": event.timestamp,
                    "bucket": event.bucket,
                    "message": event.message,
                    "chain_id": event.chain_id,
                    "hypothesis_id": event.hypothesis_id,
                    "entrypoint": event.fsm_key,
                    "verdict": event.verdict,
                    "source_state": event.source_state,
                    "target_state": event.target_state,
                    "transition_id": event.transition_id,
                    "template": event.template,
                    "step_score": event.step_score,
                    "score": event.cumulative_score,
                })

        return pd.DataFrame(rows)

    def hypotheses_frame(self) -> pd.DataFrame:
        """Return every hypothesis, including completed, expired, and pruned ones."""
        rows: List[Dict[str, Any]] = []
        for chain in self.active_chains.values():
            for hypothesis in chain.hypotheses.values():
                rows.append({
                    "chain_id": chain.chain_id,
                    "entrypoint": chain.fsm_key,
                    "chain_status": chain.status,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "status": hypothesis.status,
                    "termination_reason": hypothesis.termination_reason,
                    "current_state": hypothesis.current_state,
                    "score": hypothesis.score,
                    "transition_count": len(hypothesis.transition_ids),
                    "transition_ids": " | ".join(hypothesis.transition_ids),
                    "state_path": " | ".join(hypothesis.state_path),
                    "log_count": len(hypothesis.logs),
                })
        return pd.DataFrame(rows).sort_values(
            ["chain_id", "score"], ascending=[True, False]
        ) if rows else pd.DataFrame()

    def chains_frame(self) -> pd.DataFrame:
        """Return one aggregate row per request-chain candidate."""
        rows: List[Dict[str, Any]] = []
        for chain in self.active_chains.values():
            active = chain.active_hypotheses
            completed = chain.completed_hypotheses
            rows.append({
                "chain_id": chain.chain_id,
                "entrypoint": chain.fsm_key,
                "status": chain.status,
                "termination_reason": chain.termination_reason,
                "created_at": chain.created_at,
                "last_timestamp": chain.last_timestamp,
                "score": chain.score,
                "hypotheses_total": len(chain.hypotheses),
                "hypotheses_active": len(active),
                "hypotheses_completed": len(completed),
                "frontier_size": len(chain.frontier),
                "frontier": " | ".join(chain.frontier),
            })
        return pd.DataFrame(rows).sort_values(
            ["status", "score"], ascending=[True, False]
        ) if rows else pd.DataFrame()


# Preserve the former class name as a migration alias.
ChainStore = ExplicitHypothesisStore


def classify_logs_with_chain_store(
    logs: pd.DataFrame,
    fsms: Dict[str, StaticLogFSM],
    threshold: float = 0.55,
    time_gap_sec: int = 30,
    max_active_chains: int = 512,
    max_hypotheses_per_chain: int = 32,
    max_total_active_hypotheses: int = 2048,
    call_node_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, ExplicitHypothesisStore]:
    """
    Notebook-friendly one-shot explicit-hypothesis classifier.

    Returns `(events_df, chains_df, store)`. Call `store.hypotheses_frame()`
    when exact state/transition paths are needed for FLOW slicing or RCA.
    """
    store = ExplicitHypothesisStore(
        fsms=fsms,
        threshold=threshold,
        time_gap_sec=time_gap_sec,
        max_active_chains=max_active_chains,
        max_hypotheses_per_chain=max_hypotheses_per_chain,
        max_total_active_hypotheses=max_total_active_hypotheses,
    )
    events = store.process_frame(logs, call_node_column=call_node_column)
    return events, store.chains_frame(), store


# Compatibility alias for older notebook cells.
classifylogswithchainstore = classify_logs_with_chain_store


# --------------------------------------------------------------------------- #
# CSV ingestion helpers
# --------------------------------------------------------------------------- #


def extract_logs(
    dataset_path: str,
    service: str,
    start_time: int,
    end_time: int,
) -> List[Tuple[int, str, str]]:
    """
    Read (timestamp, bucket, message) rows for one service in a time interval.

    Expected CSV layout: timestamp, service, message. Missing files return an
    empty list after printing an error for notebook compatibility.
    """
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

                log_service = row[1].strip()
                message = row[2].strip()
                if log_service.lower() == service.lower() and start_time <= timestamp <= end_time:
                    logs.append((timestamp, f"{log_service}@{timestamp}", message))
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {dataset_path}")
    return logs


def logs_to_df(logs: List[Tuple[int, str, str]]) -> pd.DataFrame:
    """Convert extract_logs() tuples into ChainStore-compatible DataFrame columns."""
    return pd.DataFrame(logs, columns=["timestamp", "bucket", "message"])
