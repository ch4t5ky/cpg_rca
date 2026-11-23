"""Train (infer) PRINS system models for dataset incidents.

Runs the PRINS pipeline (projection -> per-component MINT inference ->
stitching -> hybrid determinization) on each selected incident directory and
saves the resulting DFA models plus a run summary.

Requires Java (MINT runs mint-inference.jar).

Usage:
    python eval/comparison/baselines/train_prins.py --filter re3ob_adservice --out output/prins_models
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prins_data import list_incidents, load_incident  # noqa: E402

from PRINS import PRINS  # noqa: E402


def train_one(incident: Path, args) -> dict:
    l_vectors, _ = load_incident(incident, max_traces=args.max_traces)
    if len(l_vectors) < 2:
        return {"incident": incident.name, "error": "too few traces", "num_logs": len(l_vectors)}

    instance = PRINS(incident.name, l_vectors, os.path.join(args.out, incident.name))
    start = time.time()
    try:
        m_sys, p_time, i_time, s_time = instance.run(
            mint_timeout=args.mint_timeout,
            mint_param=args.k,
            ignore_values=True,
            save_pdf=False,
            num_workers=args.workers,
        )
        dfa_sys, det_time = PRINS.postprocess(m_sys, determinize_technique=f"hybrid-{args.u}")
    except subprocess.TimeoutExpired:
        return {"incident": incident.name, "error": f"mint timeout ({args.mint_timeout}s)", "num_logs": len(l_vectors)}
    except subprocess.CalledProcessError:
        return {"incident": incident.name, "error": "mint crash", "num_logs": len(l_vectors)}
    except MemoryError:
        return {"incident": incident.name, "error": "out of memory", "num_logs": len(l_vectors)}
    total = time.time() - start

    model_path = Path(args.out) / f"{incident.name}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(dfa_sys, f, pickle.HIGHEST_PROTOCOL)

    return {
        "incident": incident.name,
        "num_logs": len(l_vectors),
        "states": len(dfa_sys.states),
        "transitions": len(dfa_sys.transitions),
        "projection_s": round(p_time, 3),
        "inference_s": round(i_time, 3),
        "stitching_s": round(s_time, 3),
        "determinization_s": round(det_time, 3),
        "total_s": round(total, 3),
        "model": str(model_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="dataset", help="incident dataset root (default: dataset)")
    parser.add_argument("--filter", default=None, help="substring filter on incident directory names")
    parser.add_argument("--out", default="output/prins_models", help="output directory for models and summary")
    parser.add_argument("--max-traces", type=int, default=None, help="cap traces per incident (random sample)")
    parser.add_argument("--workers", type=int, default=4, help="parallel component-inference workers")
    parser.add_argument("--k", type=int, default=2, help="MINT parameter k (default: 2)")
    parser.add_argument("--u", type=int, default=1, help="HDu determinization threshold u (default: 1)")
    parser.add_argument("--mint-timeout", type=int, default=3600, help="MINT timeout in seconds")
    args = parser.parse_args()

    incidents = list_incidents(args.dataset, args.filter)
    if not incidents:
        print(f"No incident directories found in {args.dataset!r} (filter={args.filter!r})", file=sys.stderr)
        return 1

    results = []
    for incident in incidents:
        print(f"=== {incident.name} ===")
        row = train_one(incident, args)
        print(row)
        results.append(row)
        # persist intermediate summary after each incident
        write_summary(results, args.out)
    return 0


def write_summary(results: list[dict], out: str) -> None:
    import csv

    Path(out).mkdir(parents=True, exist_ok=True)
    path = Path(out) / "summary_train.csv"
    keys = ["incident", "num_logs", "states", "transitions", "projection_s", "inference_s",
            "stitching_s", "determinization_s", "total_s", "model", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"summary written to {path}")


if __name__ == "__main__":
    sys.exit(main())
