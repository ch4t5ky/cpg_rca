"""Unit tests for eval/comparison/metrics.py acceptance-quality metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.comparison.metrics import NA, metric_block, summarize  # noqa: E402


def row(case_id, ground_truth, decision, **kw):
    return {
        "method": "m",
        "case_id": case_id,
        "group": kw.get("group", ""),
        "split": kw.get("split", ""),
        "ground_truth": ground_truth,
        "decision": decision,
        "invalid_group": kw.get("invalid_group", ""),
        "n_events": kw.get("n_events", 3),
        "reject_reason": kw.get("reject_reason", ""),
        "check_time_s": kw.get("check_time_s", 0.001),
    }


class TestMetricBlock:
    def test_perfect_acceptance(self):
        rows = [row("a", "valid", "accepted"), row("b", "invalid", "rejected")]
        m = metric_block(rows)
        assert m["tp_valid_accepted"] == 1 and m["fn_valid_rejected"] == 0
        assert m["tn_invalid_rejected"] == 1 and m["fp_invalid_accepted"] == 0
        assert m["recall_valid"] == 1.0
        assert m["specificity_invalid"] == 1.0
        assert m["balanced_accuracy"] == 1.0
        assert m["n_cases"] == 2

    def test_partial_acceptance_counts_as_rejection(self):
        # a partially matched valid trace is decided "rejected", never "accepted"
        m = metric_block([row("a", "valid", "rejected", reject_reason="not_terminal_state")])
        assert m["recall_valid"] == 0.0
        assert m["specificity_invalid"] is NA  # no invalid traces in block

    def test_no_invalid_traces_gives_na(self):
        m = metric_block([row("a", "valid", "accepted")])
        assert m["specificity_invalid"] is NA
        assert m["balanced_accuracy"] is NA
        assert m["recall_valid"] == 1.0

    def test_no_valid_traces_gives_na_recall(self):
        m = metric_block([row("a", "invalid", "rejected")])
        assert m["recall_valid"] is NA
        assert m["balanced_accuracy"] is NA
        assert m["specificity_invalid"] == 1.0

    def test_empty_block(self):
        m = metric_block([])
        assert m["n_cases"] == 0
        assert m["recall_valid"] is NA
        assert m["specificity_invalid"] is NA
        assert m["balanced_accuracy"] is NA

    def test_blocked_rows_are_excluded_from_cases_but_not_confusion(self):
        rows = [row("a", "valid", "accepted"),
                row("b", "valid", "blocked"),
                row("c", "invalid", "rejected")]
        m = metric_block(rows)
        assert m["n_cases"] == 2
        assert m["recall_valid"] == 1.0
        assert m["specificity_invalid"] == 1.0

    def test_balanced_accuracy_formula(self):
        rows = [row("a", "valid", "accepted"), row("b", "valid", "rejected"),
                row("c", "invalid", "rejected"), row("d", "invalid", "accepted")]
        m = metric_block(rows)
        assert m["recall_valid"] == 0.5
        assert m["specificity_invalid"] == 0.5
        assert m["balanced_accuracy"] == pytest.approx(0.5)

    def test_check_time_aggregation_ignores_blocked(self):
        rows = [row("a", "valid", "accepted", check_time_s=0.002),
                row("b", "valid", "blocked", check_time_s=None)]
        m = metric_block(rows)
        assert m["check_time_total_s"] == 0.002
        assert m["check_time_mean_s"] == 0.002


class TestSummarize:
    def _rows(self):
        return [
            row("k1", "valid", "accepted", group="known", split="train"),
            row("k2", "valid", "rejected", group="known", split="test"),
            row("h1", "valid", "accepted", group="heldout", split="test"),
            row("i1", "invalid", "rejected", group="invalid", split="invalid",
                invalid_group="known_alphabet"),
            row("i2", "invalid", "accepted", group="invalid", split="invalid",
                invalid_group="unknown_event"),
        ]

    def test_standard_blocks(self):
        rows = summarize(self._rows(), "m")
        by_block = {r["block"]: r for r in rows}
        assert set(by_block) == {"combined", "valid_all", "valid_known", "valid_heldout",
                                 "valid_test_split", "invalid",
                                 "invalid_known_alphabet", "invalid_unknown_event"}
        assert by_block["combined"]["recall_valid"] == pytest.approx(2 / 3)
        assert by_block["combined"]["specificity_invalid"] == 0.5
        assert by_block["combined"]["balanced_accuracy"] == pytest.approx((2 / 3 + 0.5) / 2)
        assert by_block["valid_all"]["recall_valid"] == pytest.approx(2 / 3)
        assert by_block["valid_known"]["recall_valid"] == 0.5
        assert by_block["valid_heldout"]["recall_valid"] == 1.0
        # test split covers k2 + h1
        assert by_block["valid_test_split"]["recall_valid"] == 0.5
        assert by_block["invalid"]["specificity_invalid"] == 0.5
        # the only order-violation case was rejected: real selectivity signal
        assert by_block["invalid_known_alphabet"]["n_invalid"] == 1
        assert by_block["invalid_known_alphabet"]["specificity_invalid"] == 1.0
        # the unknown-event case was accepted; this group's "specificity" is
        # uninformative anyway (rejections there are trivial)
        assert by_block["invalid_unknown_event"]["n_invalid"] == 1
        assert by_block["invalid_unknown_event"]["specificity_invalid"] == 0.0
        # recall has no valid traces here, so BA is N/A per metric_block
        assert by_block["invalid"]["balanced_accuracy"] is NA
        # blocks without invalid traces report N/A specificity
        assert by_block["valid_all"]["specificity_invalid"] is NA
        assert by_block["valid_all"]["balanced_accuracy"] is NA

    def test_status_error_and_blocked_rows_have_na_metrics(self):
        for status in ("error", "blocked"):
            rows = summarize([], "m", status=status)
            assert len(rows) == 8
            for r in rows:
                assert r["status"] == status
                assert r["recall_valid"] is NA
                assert r["n_cases"] == 0

    def test_extra_fields_propagate(self):
        rows = summarize([], "m", status="blocked", extra={"blocked_reason": "x"})
        assert all(r["blocked_reason"] == "x" for r in rows)
