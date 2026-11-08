"""
Ranking-quality metrics for root-cause localization: AC@k, MRR, Recall@k.

Metric definitions
-------------------
AC@k      : fraction of incidents where AT LEAST ONE ground-truth item
            is within the top-k predicted items ("hit@k" elsewhere).
MRR       : mean over incidents of 1 / rank(first ground-truth item
            found in the prediction); 0 if none appear in the prediction.
Recall@k : mean over incidents of |ground_truth intersect top-k| / |ground_truth|

Input format
------------
A JSON file mapping incident_id -> {"prediction": [...], "ground_truth": [...]}
(or --predictions-dir/--ground-truth-dir for two directories of one JSON
file per incident instead -- see load_pair_dirs()).

Usage
-----
    python3 metrics.py --input results.json --ks 1,3,5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def normalize(items: list[str]) -> list[str]:
    return [str(item).strip().lower() for item in items]


def rank_of_best_match(prediction: list[str], ground_truth: list[str]) -> int | None:
    pred_norm = normalize(prediction)
    gt_norm = set(normalize(ground_truth))
    for rank, item in enumerate(pred_norm, start=1):
        if item in gt_norm:
            return rank
    return None


def hit_at_k(prediction: list[str], ground_truth: list[str], k: int) -> bool:
    gt_norm = set(normalize(ground_truth))
    top_k = set(normalize(prediction[:k]))
    return bool(gt_norm & top_k)


def recall_at_k(prediction: list[str], ground_truth: list[str], k: int) -> float:
    gt_norm = set(normalize(ground_truth))
    if not gt_norm:
        return 0.0
    top_k = set(normalize(prediction[:k]))
    return len(gt_norm & top_k) / len(gt_norm)


def reciprocal_rank(prediction: list[str], ground_truth: list[str]) -> float:
    rank = rank_of_best_match(prediction, ground_truth)
    return 1.0 / rank if rank is not None else 0.0


def evaluate_bug(prediction: list[str], ground_truth: list[str], ks: list[int]) -> dict[str, Any]:
    return {
        "rank_of_best_match": rank_of_best_match(prediction, ground_truth),
        "reciprocal_rank": reciprocal_rank(prediction, ground_truth),
        **{f"hit@{k}": hit_at_k(prediction, ground_truth, k) for k in ks},
        **{f"recall@{k}": recall_at_k(prediction, ground_truth, k) for k in ks},
    }


def aggregate(per_bug: dict[str, dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    n = len(per_bug)
    if n == 0:
        return {"bug_count": 0}

    agg: dict[str, Any] = {"bug_count": n}
    agg["mrr"] = sum(v["reciprocal_rank"] for v in per_bug.values()) / n
    for k in ks:
        agg[f"ac@{k}"] = sum(1 for v in per_bug.values() if v[f"hit@{k}"]) / n
        agg[f"recall@{k}"] = sum(v[f"recall@{k}"] for v in per_bug.values()) / n
    found = [v["rank_of_best_match"] for v in per_bug.values() if v["rank_of_best_match"] is not None]
    agg["avg_rank_when_found"] = sum(found) / len(found) if found else None
    agg["not_found_count"] = n - len(found)
    return agg


def load_pair_dirs(predictions_dir: Path, ground_truth_dir: Path) -> dict[str, dict[str, list[str]]]:
    results: dict[str, dict[str, list[str]]] = {}
    for pred_file in sorted(predictions_dir.glob("*.json")):
        incident_id = pred_file.stem
        gt_file = ground_truth_dir / f"{incident_id}.json"
        if not gt_file.is_file():
            continue
        prediction = json.loads(pred_file.read_text())["return"]["potential_root_cause"]
        ground_truth = json.loads(gt_file.read_text())["return"]["real_root_cause"]
        results[incident_id] = {"prediction": prediction, "ground_truth": ground_truth}
    return results


def load_json_input(path: Path) -> dict[str, dict[str, list[str]]]:
    return json.loads(path.read_text())


def run(results: dict[str, dict[str, list[str]]], ks: list[int]) -> dict[str, Any]:
    per_bug = {
        bug_id: evaluate_bug(data["prediction"], data["ground_truth"], ks)
        for bug_id, data in results.items()
    }
    return {"per_bug": per_bug, "aggregate": aggregate(per_bug, ks)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="JSON file: {incident_id: {prediction, ground_truth}}")
    parser.add_argument("--predictions-dir", type=Path, help="directory of one {incident_id}.json prediction file per incident")
    parser.add_argument("--ground-truth-dir", type=Path, help="directory of one {incident_id}.json ground-truth file per incident")
    parser.add_argument("--ks", default="1,3,5", help="comma-separated k values for AC@k/Recall@k")
    parser.add_argument("--output", type=Path, help="optional path to write the full JSON report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ks = [int(k) for k in args.ks.split(",")]

    if args.input:
        results = load_json_input(args.input)
    elif args.predictions_dir and args.ground_truth_dir:
        results = load_pair_dirs(args.predictions_dir, args.ground_truth_dir)
    else:
        print("Provide either --input or both --predictions-dir and --ground-truth-dir", file=sys.stderr)
        return 1

    if not results:
        print("No matched bug_id pairs found.", file=sys.stderr)
        return 1

    report = run(results, ks)

    print(f"Evaluated {report['aggregate']['bug_count']} bugs")
    print(f"  MRR            : {report['aggregate']['mrr']:.3f}")
    for k in ks:
        print(f"  AC@{k}          : {report['aggregate'][f'ac@{k}']:.3f}")
        print(f"  Recall@{k}      : {report['aggregate'][f'recall@{k}']:.3f}")
    print(f"  Avg rank (found): {report['aggregate']['avg_rank_when_found']}")
    print(f"  Not found       : {report['aggregate']['not_found_count']}/{report['aggregate']['bug_count']}")
    print()
    for bug_id, v in report["per_bug"].items():
        print(f"  {bug_id}: rank={v['rank_of_best_match']} RR={v['reciprocal_rank']:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
