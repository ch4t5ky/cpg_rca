"""
hypothesis_report.py
====================

Expand completed explicit FSM hypotheses into LLM-ready execution reports.

A report resolves:
    PathHypothesis -> LogTransition -> ExecutionSegment -> FLOW slice

The report is JSON-first. Its `flow_slice` remains FLOW-compatible, allowing an
LLM or graph-processing pipeline to consume the original semantic graph format
with runtime-incompatible methods and branches removed.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.offline.finite_state_machine import ExecutionSegment, LogTransition, StaticLogFSM
from src.online.flow_slicer import slice_flow_for_hypothesis
from src.online.stateful_chain_store import ActiveChain, PathHypothesis


def transition_index(fsm: StaticLogFSM) -> dict[str, LogTransition]:
    """Index FSM transitions by stable transition id."""
    return {transition.id: transition for transition in fsm.transitions}


def expand_segments(hypothesis: PathHypothesis, fsm: StaticLogFSM) -> list[ExecutionSegment]:
    """Resolve the ordered hypothesis state path into ExecutionSegment objects."""
    return [fsm.states[state_id] for state_id in hypothesis.state_path if state_id in fsm.states]


def expand_transitions(hypothesis: PathHypothesis, fsm: StaticLogFSM) -> list[LogTransition]:
    """Resolve ordered hypothesis transition ids into LogTransition objects."""
    index = transition_index(fsm)
    return [index[transition_id] for transition_id in hypothesis.transition_ids if transition_id in index]


def build_llm_report(
    service: str,
    chain: ActiveChain,
    hypothesis: PathHypothesis,
    fsm: StaticLogFSM,
    flow: dict[str, Any],
    max_call_depth: int = 3,
) -> dict[str, Any]:
    """
    Build one self-contained LLM-ready report for a completed hypothesis.

    The report intentionally preserves both runtime evidence and the induced
    FLOW slice. It does not claim that the path is globally unique: the caller
    can include competing active/completed hypotheses for the same chain.
    """
    transitions = expand_transitions(hypothesis, fsm)
    segments = expand_segments(hypothesis, fsm)
    flow_slice = slice_flow_for_hypothesis(
        flow=flow,
        segments=segments,
        transitions=transitions,
        max_call_depth=max_call_depth,
    )

    timestamps = [log.timestamp for log in hypothesis.logs if log.timestamp is not None]
    competing = [
        other for other in chain.hypotheses.values()
        if other.hypothesis_id != hypothesis.hypothesis_id
        and other.status in {"active", "completed"}
    ]

    return {
        "report_type": "completed_execution_hypothesis",
        "service": service,
        "chain": {
            "chain_id": chain.chain_id,
            "chain_status": chain.status,
            "chain_score": chain.score,
            "created_at": chain.created_at,
            "last_timestamp": chain.last_timestamp,
        },
        "hypothesis": {
            "hypothesis_id": hypothesis.hypothesis_id,
            "status": hypothesis.status,
            "termination_reason": hypothesis.termination_reason,
            "cumulative_score": hypothesis.score,
            "transition_ids": list(hypothesis.transition_ids),
            "state_path": list(hypothesis.state_path),
            "transition_count": len(hypothesis.transition_ids),
        },
        "entrypoint": {
            "name": fsm.entrypoint_name,
            "full_name": fsm.entrypoint_full_name,
            "node_id": fsm.entrypoint_node_id,
        },
        "time_window": {
            "start": min(timestamps) if timestamps else None,
            "end": max(timestamps) if timestamps else None,
        },
        "runtime_evidence": [asdict(log) for log in hypothesis.logs],
        "fsm_path": {
            "transitions": [asdict(transition) for transition in transitions],
            "segments": [asdict(segment) for segment in segments],
        },
        "flow_slice": flow_slice,
        "ambiguity": {
            "competing_hypothesis_count": len(competing),
            "competing_hypotheses": [
                {
                    "hypothesis_id": other.hypothesis_id,
                    "status": other.status,
                    "score": other.score,
                    "current_state": other.current_state,
                    "transition_ids": list(other.transition_ids),
                }
                for other in sorted(competing, key=lambda item: item.score, reverse=True)
            ],
        },
    }


def build_unexplained_report(
    service: str,
    event,
) -> dict[str, Any]:
    """Build a minimal structured report for a structurally unexplained event."""
    return {
        "report_type": "structurally_unexplained_event",
        "service": service,
        "runtime_event": {
            "index": event.index,
            "timestamp": event.timestamp,
            "bucket": event.bucket,
            "message": event.message,
        },
        "classification": {
            "verdict": event.verdict,
            "score": event.step_score,
            "reason": "No active-hypothesis continuation or FSM start transition matched the event.",
        },
    }
