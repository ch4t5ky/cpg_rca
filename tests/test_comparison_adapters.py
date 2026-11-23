"""Tests for the acceptance adapters used by the unified FSM experiment.

CpgRcaModel / LogSynthFsmModel are tested on small hand-built artefacts to
pin down the shared full-acceptance semantics (initial state, every event
consumed, terminal state; no skipping). These tests exercise only the
adapters, not the underlying methods.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval" / "comparison" / "baselines" / "LogSynthFSM-KDD-2026"))

from eval.comparison.run import CpgRcaModel, LogSynthFsmModel  # noqa: E402


def make_cpg_fsm_payload():
    # start ->(a) s1 ->(b <*>) s2(terminal); s2 ->(c) end(terminal)
    return {
        "entrypoint_full_name": "ep",
        "start_states": ["start"],
        "terminal_states": ["end"],
        "states": [{"id": s} for s in ("start", "s1", "s2", "end")],
        "transitions": [
            {"source_segment_id": "start", "target_segment_id": "s1",
             "template": "a", "log_call_node_id": 1},
            {"source_segment_id": "s1", "target_segment_id": "s2",
             "template": "b <*>", "log_call_node_id": 2},
            {"source_segment_id": "s2", "target_segment_id": "end",
             "template": "c", "log_call_node_id": 3},
        ],
    }


class TestCpgRcaModel:
    def setup_method(self):
        self.model = CpgRcaModel(make_cpg_fsm_payload())

    def test_full_acceptance(self):
        assert self.model.check(["a", "b 42", "c"])[0] is True

    def test_unknown_event(self):
        ok, n, reason = self.model.check(["a", "zzz", "c"])
        assert (ok, reason) == (False, "unknown_event") and n == 1

    def test_wrong_order_rejected(self):
        ok, _, reason = self.model.check(["b 1", "a"])
        # "b 1" is in the alphabet but not reachable from the start state
        assert ok is False and reason == "no_matching_transition"

    def test_prefix_not_terminal(self):
        ok, n, reason = self.model.check(["a", "b 7"])
        assert ok is False and reason == "not_terminal_state" and n == 2

    def test_no_skipping(self):
        # cannot jump over "b ..." event
        ok, _, reason = self.model.check(["a", "c"])
        assert ok is False and reason == "no_matching_transition"

    def test_wildcard_matching(self):
        assert self.model.alphabet_covers("b anything at all")
        # adapter semantics: a template matches an event by token prefix;
        # an event may carry extra trailing words beyond the template
        assert self.model.alphabet_covers("c d e")
        assert not self.model.alphabet_covers("zzz")

    def test_event_normalizer_closes_alphabet_gap(self):
        # cpg_rca templates are punctuation-free; the event normalizer must
        # apply the same cleaning so "failure, retrying" matches its template
        normalizer = lambda e: e.replace(",", " ")
        model = CpgRcaModel(make_cpg_fsm_payload(), event_normalizer=normalizer)
        assert model.alphabet_covers("c,")
        assert model.check(["a", "b 1 x", "c,"])[0] is True


def make_logsynth_fsm():
    from logsynthfsm.fsm.model import FSM, State, TemplateStateMapping, Transition

    return FSM(
        states=[State("S1", "S1", ""), State("S2", "S2", ""), State("S3", "S3", "")],
        transitions=[
            Transition("S1", "S2", 10, 1.0),
            Transition("S2", "S3", 10, 1.0),
        ],
        template_state_map=[
            TemplateStateMapping("TMP_A", "S1"),
            TemplateStateMapping("TMP_B", "S2"),
            TemplateStateMapping("TMP_C", "S3"),
        ],
    )


class TestLogSynthFsmModel:
    def setup_method(self):
        self.model = LogSynthFsmModel(make_logsynth_fsm())

    def test_start_and_terminal_derivation(self):
        assert self.model.start_states == {"S1"}
        assert self.model.terminal_states == {"S3"}

    def test_full_acceptance(self):
        # templatize keeps plain words intact, so stable ids match TMP_* only
        # via monkeypatched mapping; here we bypass templatize by patching.
        model = self.model
        model.event_template_id = lambda e: {"a": "TMP_A", "b": "TMP_B", "c": "TMP_C"}.get(e)
        assert model.check(["a", "b", "c"])[0] is True
        ok, _, reason = model.check(["a", "b"])
        assert ok is False and reason == "not_terminal_state"
        ok, _, reason = model.check(["a", "c"])
        assert ok is False and reason == "no_matching_transition"
        ok, _, reason = model.check(["a", "b", "zzz"])
        assert ok is False and reason == "unknown_event"
