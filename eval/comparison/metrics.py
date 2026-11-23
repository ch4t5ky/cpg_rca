"""Acceptance-quality metrics for behavioural-model comparison.

Metric definitions
------------------
Recall_valid        : fraction of fully accepted valid traces.
                      A trace is *fully accepted* only when the model can
                      consume every event from an initial state and end in an
                      accepting/terminal state. Partial matches never count.
Specificity_invalid : fraction of independently labelled invalid traces
                      that are rejected (fully or by failing acceptance).
Balanced Accuracy   : (Recall_valid + Specificity_invalid) / 2; N/A when no
                      independently labelled invalid traces exist.

All functions are pure and operate on per-execution row dicts so they can be
unit-tested without any model artefacts:

    row = {
        "method": str,
        "case_id": str,
        "group": str,          # scenario group, e.g. "known" / "heldout"
        "split": str,          # "train" / "test" / "invalid"
        "ground_truth": "valid" | "invalid",
        "decision": "accepted" | "rejected" | "blocked",
        "invalid_group": str,    # for invalid rows: "known_alphabet" |
                                 # "unknown_event" (per-model alphabet split)
        "n_events": int,
        "reject_reason": str,
        "check_time_s": float,
    }
"""

from __future__ import annotations

NA = "N/A"


def metric_block(rows: list[dict]) -> dict:
    """Compute recall/specificity/BA over one set of per-execution rows.

    Numerators and denominators are always included so the summary is
    auditable. Specificity and BA are ``NA`` when the block contains no
    invalid traces.
    """
    valid = [r for r in rows if r["ground_truth"] == "valid"]
    invalid = [r for r in rows if r["ground_truth"] == "invalid"]
    decided = [r for r in rows if r["decision"] != "blocked"]

    tp = sum(1 for r in valid if r["decision"] == "accepted")
    fn = sum(1 for r in valid if r["decision"] == "rejected")
    tn = sum(1 for r in invalid if r["decision"] == "rejected")
    fp = sum(1 for r in invalid if r["decision"] == "accepted")

    recall_valid = tp / (tp + fn) if (tp + fn) else NA
    specificity = tn / (tn + fp) if (tn + fp) else NA
    if specificity is NA or recall_valid is NA:
        balanced = NA
    else:
        balanced = (recall_valid + specificity) / 2

    check_times = [r["check_time_s"] for r in decided if r.get("check_time_s") is not None]
    return {
        "n_cases": len(decided),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "tp_valid_accepted": tp,
        "fn_valid_rejected": fn,
        "tn_invalid_rejected": tn,
        "fp_invalid_accepted": fp,
        "recall_valid": round(recall_valid, 6) if recall_valid is not NA else NA,
        "specificity_invalid": round(specificity, 6) if specificity is not NA else NA,
        "balanced_accuracy": round(balanced, 6) if balanced is not NA else NA,
        "check_time_total_s": round(sum(check_times), 6),
        "check_time_mean_s": round(sum(check_times) / len(check_times), 6) if check_times else NA,
    }


def summarize(rows: list[dict], method: str, status: str = "ok",
              build_time_s: object = NA, coverage: object = NA,
              extra: dict | None = None) -> list[dict]:
    """Build summary rows for one method across the standard evaluation blocks.

    Blocks: ``valid_all`` (every valid trace), ``valid_known`` (scenarios that
    occur in train), ``valid_heldout`` (scenarios unseen in train),
    ``valid_test_split`` (test-split executions only), ``invalid`` (all
    independently labelled invalid traces), ``invalid_known_alphabet``
    (invalid traces whose events are all inside the model's alphabet — a real
    selectivity check), ``invalid_unknown_event`` (invalid traces containing
    an event outside the model's alphabet — rejection there is trivial and
    uninformative), and ``combined`` (valid + invalid, the only block where
    balanced accuracy is defined whenever both classes exist).
    """
    blocks = {
        "combined": list(rows),
        "valid_all": [r for r in rows if r["ground_truth"] == "valid"],
        "valid_known": [r for r in rows if r["ground_truth"] == "valid" and r["group"] == "known"],
        "valid_heldout": [r for r in rows if r["ground_truth"] == "valid" and r["group"] == "heldout"],
        "valid_test_split": [r for r in rows if r["ground_truth"] == "valid" and r["split"] == "test"],
        "invalid": [r for r in rows if r["ground_truth"] == "invalid"],
        "invalid_known_alphabet": [r for r in rows if r["ground_truth"] == "invalid"
                                   and r.get("invalid_group") == "known_alphabet"],
        "invalid_unknown_event": [r for r in rows if r["ground_truth"] == "invalid"
                                  and r.get("invalid_group") == "unknown_event"],
    }
    out = []
    for block_name, block_rows in blocks.items():
        metrics = metric_block(block_rows) if status == "ok" else {
            "n_cases": 0, "n_valid": 0, "n_invalid": 0,
            "tp_valid_accepted": 0, "fn_valid_rejected": 0,
            "tn_invalid_rejected": 0, "fp_invalid_accepted": 0,
            "recall_valid": NA, "specificity_invalid": NA, "balanced_accuracy": NA,
            "check_time_total_s": 0.0, "check_time_mean_s": NA,
        }
        row = {
            "method": method,
            "block": block_name,
            "status": status,
            "build_time_s": build_time_s,
            "event_coverage": coverage,
        }
        row.update(metrics)
        if extra:
            row.update(extra)
        out.append(row)
    return out
