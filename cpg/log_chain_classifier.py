from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cpg.flow import EntrypointFlow
from cpg.log_flow import LogFlowExtractor

START = "__start__"
RETURN = "__return__"
INCOMPLETE = "__incomplete__"


@dataclass
class AnalyzerConfig:
    service_name: str
    dataset_path: str
    start_time: int
    end_time: int
    max_depth: int = 10
    max_loop_unroll: int = 1
    max_paths: int = 256
    flow_max_depth: int = 5
    flow_max_paths: int = 50
    time_gap_sec: int = 30
    match_threshold: float = 0.35
    output_dir: str = "output"


def safe_filename(value: str, max_length: int = 100) -> str:
    value = (value or "").strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return (value or "unnamed")[:max_length]


def bucket_ts(bucket: str) -> Optional[int]:
    m = re.search(r"@(\d+)$", str(bucket))
    return int(m.group(1)) if m else None


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def extract_logs(dataset_path: str, service: str, start_time: int, end_time: int) -> List[Tuple[int, str, str]]:
    logs: List[Tuple[int, str, str]] = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    timestamp = int(float(row[0]))
                    log_service = row[1].strip()
                    message = row[2].strip()
                    if log_service.lower() == service.lower() and start_time <= timestamp <= end_time:
                        bucket = f"{log_service}@{timestamp}"
                        logs.append((timestamp, bucket, message))
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {dataset_path}")
    print(f"✓ Extracted {len(logs)} logs for {service} in time window")
    return logs


def logs_to_df(logs: List[Tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(logs, columns=["timestamp", "bucket", "message"])


def has_start_transition(fsm) -> bool:
    return any(edge.source == START for edge in fsm.edges)


def build_entrypoint_fsms(G, templates, all_entrypoints, service_name: str, output_dir: str, flow_max_depth: int = 5, flow_max_paths: int = 50):
    flow_analyzer = EntrypointFlow(G, max_depth=flow_max_depth, max_paths=flow_max_paths)
    extractor = LogFlowExtractor(templates)
    fsms: Dict[str, Any] = {}
    rows: List[dict] = []
    fsms_dir = Path(output_dir) / service_name / "fsms"
    flow_dir = Path(output_dir) / service_name / "flow"
    fsms_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)

    for index, entrypoint in enumerate(all_entrypoints, start=1):
        base_name = f"{index:03d}_{safe_filename(entrypoint.name)}_{safe_filename(entrypoint.full_name)}"
        png_path = fsms_dir / f"{base_name}.png"
        row = {
            "rank": index,
            "entrypoint_nodeid": entrypoint.node_id,
            "entrypoint_name": entrypoint.name,
            "entrypoint_fullname": entrypoint.full_name,
            "outdegree": getattr(entrypoint, "outdegree", ""),
            "status": "ok",
            "states": 0,
            "edges": 0,
            "terminals": 0,
            "has_start": False,
            "png": str(png_path),
            "warnings": "",
            "error": "",
        }
        try:
            flow_result = flow_analyzer.build(entrypoint.node_id)
            mg = next((e.method_graph for e in flow_result.sequence if e.method_graph.full_name == entrypoint.full_name), None)
            fsm = extractor.extract(flow_result)
            fsms[entrypoint.full_name] = fsm
            if fsm.states and fsm.edges:
                visualize_log_fsm(fsm, output_path=str(png_path))
            else:
                row["status"] = "no_log_fsm"
            row["states"] = len(fsm.states)
            row["edges"] = len(fsm.edges)
            row["terminals"] = len(fsm.terminals)
            row["has_start"] = has_start_transition(fsm)
            row["warnings"] = " | ".join(fsm.warnings)
            if mg is not None:
                try:
                    draw_branching_call_flow_matplotlib(flow_result, method_full_name=entrypoint.full_name, method_graph=mg, filename=base_name, output_dir=str(flow_dir))
                except Exception as exc:
                    row["warnings"] = (row["warnings"] + " | " if row["warnings"] else "") + f"flow_render_failed: {type(exc).__name__}: {exc}"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"[{index:03d}/{len(all_entrypoints):03d}] {entrypoint.name}: {row['status']}, states={row['states']}, edges={row['edges']}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(Path(output_dir) / service_name / "entrypoints_summary.csv", index=False)
    return fsms, summary_df


def build_state_catalog(fsm) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {
        START: {"type": "special", "label": "start"},
        RETURN: {"type": "special", "label": "return"},
        INCOMPLETE: {"type": "special", "label": "incomplete"},
    }
    for sid, point in fsm.states.items():
        catalog[sid] = {"type": "log", "label": normalize_text(getattr(point, "template", "")), "method": getattr(point, "method_fullname", ""), "call_node_id": getattr(point, "call_node_id", "")}
    for sid, point in getattr(fsm, "external_states", {}).items():
        catalog[sid] = {"type": "external", "label": normalize_text(getattr(point, "callee_full_name", "").rsplit(".", 1)[-1]), "method": getattr(point, "caller_full_name", "")}
    return catalog


def build_transition_map(fsm) -> Dict[str, set]:
    transitions: Dict[str, set] = defaultdict(set)
    for e in fsm.edges:
        if e.target in (RETURN, INCOMPLETE):
            continue
        transitions[e.source].add(e.target)
    return transitions


def resolve_best_state(message: str, catalog: Dict[str, Dict[str, Any]], candidate_ids: Optional[Iterable[str]] = None) -> Tuple[str, float]:
    msg = normalize_text(message)
    best_sid = ""
    best_score = 0.0
    ids = candidate_ids if candidate_ids is not None else catalog.keys()
    for sid in ids:
        meta = catalog.get(sid)
        if not meta or meta["type"] not in ("log", "external"):
            continue
        label = meta["label"]
        sc = similarity(msg, label)
        if msg and label and (msg in label or label in msg):
            sc = max(sc, 0.9)
        if sc > best_score:
            best_score = sc
            best_sid = sid
    return best_sid, best_score


def classify_logs_against_fsms(df: pd.DataFrame, fsms: Dict[str, Any], threshold: float = 0.35, time_gap_sec: int = 30) -> pd.DataFrame:
    catalogs = {key: build_state_catalog(fsm) for key, fsm in fsms.items()}
    transitions = {key: build_transition_map(fsm) for key, fsm in fsms.items()}
    start_targets = {key: transitions[key].get(START, set()) for key in fsms}
    current_key = ""
    current_state = START
    chain_id = 0
    last_ts = None
    out: List[Dict[str, Any]] = []
    sorted_df = df.sort_values("timestamp").reset_index(drop=True)

    for idx, r in sorted_df.iterrows():
        ts = r.get("timestamp")
        msg = str(r.get("message", ""))
        bucket = str(r.get("bucket", ""))
        if last_ts is not None and ts is not None and isinstance(ts, (int, float)) and isinstance(last_ts, (int, float)) and ts - last_ts > time_gap_sec:
            current_key = ""
            current_state = START
            chain_id += 1
        best_key = ""
        best_sid = ""
        best_score = 0.0
        checked_flow = ""
        for fsm_key, catalog in catalogs.items():
            trans = transitions[fsm_key]
            expected = trans.get(current_state, set()) if fsm_key == current_key else set()
            candidate_ids = expected if expected else None
            sid, score = resolve_best_state(msg, catalog, candidate_ids=candidate_ids)
            if score > best_score:
                best_score = score
                best_sid = sid
                best_key = fsm_key
                checked_flow = f"{current_state} -> {sid or '∅'} @ {fsm_key}"
        if not best_key:
            verdict = "new_chain"
            chain_id += 1
            next_state = START
            current_key = ""
            current_state = START
        else:
            expected = transitions[best_key].get(current_state, set()) if best_key == current_key else set()
            if best_sid and best_sid in expected:
                verdict = "continue"
                next_state = best_sid
                current_key = best_key
            elif current_state == START and best_sid in start_targets.get(best_key, set()):
                verdict = "new_chain"
                chain_id += 1
                next_state = best_sid
                current_key = best_key
            elif best_score >= threshold and best_sid:
                verdict = "unknown"
                next_state = current_state
            else:
                verdict = "new_chain"
                chain_id += 1
                next_state = START
                current_key = ""
        out.append({**r.to_dict(), "idx": idx, "fsm_key": best_key, "chain_id": chain_id, "verdict": verdict, "current_state": current_state, "next_state": next_state, "matched_state": best_sid, "score": best_score, "checked_flow": checked_flow, "timestamp": ts if ts is not None else bucket_ts(bucket)})
        current_state = next_state if verdict == "continue" else START
        last_ts = ts
    return pd.DataFrame(out)


def draw_chain_timeline(results: pd.DataFrame, outpath: str = "output/fsm_chain_timeline.png") -> str:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(12, len(results) * 0.18), 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    colors = {"continue": "#22c55e", "new_chain": "#ef4444", "unknown": "#94a3b8"}
    rows = results.reset_index(drop=True)
    for i, row in rows.iterrows():
        ax.scatter(i, row["chain_id"], s=34, color=colors.get(row["verdict"], "#94a3b8"))
        if i > 0:
            prev = rows.iloc[i - 1]
            color = "#22c55e" if row["chain_id"] == prev["chain_id"] and row["verdict"] == "continue" else "#ef4444"
            ax.plot([i - 1, i], [prev["chain_id"], row["chain_id"]], color=color, lw=1.4, ls="-" if color == "#22c55e" else "--")
    ax.set_title("FSM chains on timeline", color="white")
    ax.set_xlabel("log index", color="white")
    ax.set_ylabel("chain id", color="white")
    ax.tick_params(colors="white")
    legend = [Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=8) for k, v in colors.items()]
    ax.legend(handles=legend, facecolor="#0f172a", edgecolor="#334155", labelcolor="white")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath


def print_flow_trace(results: pd.DataFrame, limit: int = 200) -> None:
    cols = ["bucket", "message", "fsm_key", "checked_flow", "matched_state", "chain_id", "verdict", "score"]
    display(results[cols].head(limit))


def run_multi_fsm_analyzer(G, templates, all_entrypoints, service_name: str, dataset_path: str, start_time: int, end_time: int, output_dir: str = "output", flow_max_depth: int = 5, flow_max_paths: int = 50, threshold: float = 0.35, time_gap_sec: int = 30):
    logs = extract_logs(dataset_path, service_name, start_time, end_time)
    df_logs = logs_to_df(logs)
    fsms, summary_df = build_entrypoint_fsms(G, templates, all_entrypoints, service_name, output_dir, flow_max_depth=flow_max_depth, flow_max_paths=flow_max_paths)
    classified = classify_logs_against_fsms(df_logs, fsms, threshold=threshold, time_gap_sec=time_gap_sec)
    out_dir = Path(output_dir) / service_name
    out_dir.mkdir(parents=True, exist_ok=True)
    classified.to_csv(out_dir / "fsm_chain_timeline.csv", index=False)
    draw_chain_timeline(classified, outpath=str(out_dir / "fsm_chain_timeline.png"))
    print_flow_trace(classified, limit=200)
    return {"fsms": fsms, "summary_df": summary_df, "classified_df": classified, "logs_df": df_logs}
