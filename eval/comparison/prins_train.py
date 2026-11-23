"""Train PRINS in a clean subprocess (avoids the `src` package-name collision
between the repo (src/offline, src/online) and the vendored PRINS package).

Usage:
    python eval/comparison/prins_train.py FIXTURE WORK_DIR OUT_PKL K U MINT_TIMEOUT

Writes OUT_PKL with {"dfa": ..., "vocab": {...}} and <OUT_PKL>.meta.json with
{"build_time_s": ...}. Trains on train-split executions only.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # eval/comparison
REPO_ROOT = PACKAGE_DIR.parent.parent                   # repo root
BASELINES_DIR = PACKAGE_DIR / "baselines"
PRINS_PKG_DIR = BASELINES_DIR / "PRINS" / "PRINS"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PRINS_PKG_DIR))
sys.path.insert(0, str(BASELINES_DIR))

from prins_data import normalize_message  # noqa: E402


def main() -> int:
    fixture, work_dir, out_pkl, k, u, mint_timeout = (
        Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
        int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
    )

    executions = {}
    for line in (fixture / "executions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        executions[row["execution_id"]] = row
    raw: dict[str, list[dict]] = {}
    for line in (fixture / "logs.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        raw.setdefault(row["execution_id"], []).append(row)
    for rows in raw.values():
        rows.sort(key=lambda e: e["event_index"])
    train_ids = sorted(eid for eid, m in executions.items() if m["split"] == "train")

    # MINT's trace reader requires whitespace-free event identifiers, so the
    # shared canonical messages are encoded as stable single-token ids. This
    # is input encoding only: PRINS/MINT run unchanged on the id sequences.
    vocab: dict[str, str] = {}
    l_vectors = {}
    for exec_id in train_ids:
        entries = []
        for e in raw[exec_id]:
            msg = normalize_message(e["message"])
            if msg not in vocab:
                vocab[msg] = f"E{len(vocab):04d}"
            entries.append({
                "ts": datetime.fromisoformat(e["timestamp"]).timestamp(),
                "tid": vocab[msg],
                "values": [],
                "component": "retry_service",
            })
        l_vectors[exec_id] = entries

    from src.main.PRINS import PRINS

    started = time.perf_counter()
    instance = PRINS("retryservice", l_vectors, str(work_dir))
    m_sys, _p, _i, _s = instance.run(
        mint_timeout=mint_timeout,
        mint_param=k,
        ignore_values=True,
        save_pdf=False,
        num_workers=1,
    )
    dfa_sys, _det = PRINS.postprocess(m_sys, determinize_technique=f"hybrid-{u}")
    build_s = time.perf_counter() - started

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as fh:
        pickle.dump({"dfa": dfa_sys, "vocab": vocab}, fh)
    meta = {"build_time_s": build_s, "train_traces": len(train_ids),
            "k": k, "u": u, "mint_timeout_s": mint_timeout}
    Path(str(out_pkl) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"PRINS trained: {len(train_ids)} traces, build_time_s={build_s:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
