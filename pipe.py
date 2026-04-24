"""
pipe.py
=======
Unified Log Analysis Pipeline: Extract -> Match -> Visualize

Uses cpg_trie_parser.py for:
- Building templates from CPG
- Building Trie from templates
- Matching logs to Trie

Arguments:
  --dataset   / -d   CSV log file (timestamp,service,message)
  --service   / -s   Service name to filter (case-insensitive)
  --cpg       / -c   CPG .dot file for that service
  --start     / -st  Window start timestamp (epoch seconds)
  --end       / -e   Window end timestamp (epoch seconds)

Usage:
  python pipe.py --dataset logs.csv --service checkoutservice --cpg export.dot --start 1731903264 --end 1731903310
  python pipe.py -d logs.csv -s cartservice -c export.dot -st 1731903200 -e 1731903400
"""

import sys
import csv
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# Import everything from cpg_trie_parser
from cpg_trie_parser import (
    build_templates_from_cpg,
    build_trie,
    map_logs,
    visualize_trie_matplotlib,
    LogMapping,
)

try:
    from networkx.drawing.nx_pydot import read_dot
    import networkx as nx
    PYDOT_OK = True
except Exception:
    PYDOT_OK = False

# ---------------------------------------------------------------------------
# STAGE 1: EXTRACT logs filtered by service + time window
# ---------------------------------------------------------------------------

def extract_logs(
    csv_path: str,
    service_name: str,
    time_from: int,
    time_to: int,
) -> List[Tuple[str, str]]:
    """
    Read CSV (timestamp,service,message), filter by service and time window.
    Returns list of (bucket, message) tuples ready for map_logs().
    bucket = "<service>@<timestamp>"
    """
    rows: List[Tuple[str, str]] = []

    print(f"[1-EXTRACT] File   : {csv_path}")
    print(f"[1-EXTRACT] Service: {service_name}")
    print(f"[1-EXTRACT] Window : {time_from} -> {time_to}  ({time_to - time_from}s)")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                ts = int(row[0])
                svc = row[1].strip()
                msg = row[2].strip().strip('"')
            except (ValueError, IndexError):
                continue

            if svc.lower() != service_name.lower():
                continue
            if ts < time_from or ts > time_to:
                continue

            bucket = f"{svc}@{ts}"
            rows.append((bucket, msg))

    rows.sort(key=lambda x: x[0])
    print(f"[1-EXTRACT] Found  : {len(rows)} log entries")
    if not rows:
        print("[1-EXTRACT] WARNING: no logs found — check service name and time window")
    return rows

# ---------------------------------------------------------------------------
# STAGE 2: MATCH via Trie (delegates to cpg_trie_parser)
# ---------------------------------------------------------------------------

def match_stage(
    cpg_path: str,
    log_rows: List[Tuple[str, str]],
    trie_image: str = "trie.png",
) -> List[LogMapping]:
    """
    Load CPG, build templates, build Trie, visualize Trie, map logs.
    Returns list of LogMapping from cpg_trie_parser.
    """
    if not PYDOT_OK:
        print("[2-MATCH] ERROR: pydot not installed. Run: pip install pydot")
        sys.exit(1)

    print(f"[2-MATCH] Loading CPG: {cpg_path}")
    G = read_dot(cpg_path)
    print(f"[2-MATCH] CPG loaded : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    templates = build_templates_from_cpg(G, max_ddg_depth=5)
    print(f"[2-MATCH] Templates  : {len(templates)}")

    root = build_trie(templates)
    visualize_trie_matplotlib(root, output_path=trie_image)

    mappings = map_logs(log_rows, root, templates, min_static=1)
    matched = sum(1 for m in mappings if m.matched)
    print(f"[2-MATCH] Matched    : {matched}/{len(mappings)}")

    return mappings

# ---------------------------------------------------------------------------
# STAGE 3: VISUALIZE timeline
# ---------------------------------------------------------------------------

def visualize_timeline(
    mappings: List[LogMapping],
    service_name: str,
    time_from: int,
    time_to: int,
    output_path: str = "log_timeline.png",
) -> None:
    """Draw a horizontal timeline with one box per log entry."""

    print(f"[3-VISUALIZE] Building timeline -> {output_path}")

    if not mappings:
        print("[3-VISUALIZE] Nothing to draw")
        return

    # Parse timestamps from bucket field  "<svc>@<ts>"
    def _ts(m: LogMapping) -> int:
        try:
            return int(m.bucket.split("@")[1])
        except Exception:
            return time_from

    mappings_sorted = sorted(mappings, key=_ts)

    matched_color   = "#3b82f6"   # blue  — matched to CPG
    unmatched_color = "#e5e7eb"   # gray  — unmatched

    fig, ax = plt.subplots(figsize=(max(18, len(mappings) * 2.2), 7))

    # Window background
    ax.add_patch(Rectangle(
        (time_from, -0.5), time_to - time_from, 1,
        facecolor="#f0f9ff", edgecolor="#0ea5e9",
        linewidth=2, alpha=0.25, zorder=0,
    ))

    for i, m in enumerate(mappings_sorted):
        x = _ts(m)
        color      = matched_color   if m.matched else unmatched_color
        edge_color = "#1e40af"       if m.matched else "#9ca3af"
        txt_color  = "white"         if m.matched else "#374151"

        ax.add_patch(FancyBboxPatch(
            (x - 0.35, -0.35), 0.7, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=edge_color,
            linewidth=2, alpha=0.9, zorder=2,
        ))

        label = (m.method_name or "unknown")[:13]
        ax.text(x, 0, label, fontsize=8, ha="center", va="center",
                weight="bold", color=txt_color, zorder=3)

        ax.text(x, -0.55, str(x), fontsize=6.5, ha="center",
                va="top", color="#6b7280")

        if i < len(mappings_sorted) - 1:
            nx_ = _ts(mappings_sorted[i + 1])
            ax.add_patch(FancyArrowPatch(
                (x + 0.35, 0), (nx_ - 0.35, 0),
                arrowstyle="->", color="#6b7280",
                linewidth=1.8, alpha=0.7,
                connectionstyle="arc3,rad=0.12", zorder=1,
            ))

    # Legend
    ax.legend(handles=[
        mpatches.Patch(facecolor=matched_color,   edgecolor="#1e40af", label="Matched to CPG"),
        mpatches.Patch(facecolor=unmatched_color, edgecolor="#9ca3af", label="Unmatched"),
    ], loc="upper right", fontsize=10, framealpha=0.95)

    pad = max(2, (time_to - time_from) * 0.03)
    ax.set_xlim(time_from - pad, time_to + pad)
    ax.set_ylim(-1.1, 1.0)
    ax.set_xlabel("Timestamp (epoch seconds)", fontsize=11, weight="bold")
    ax.set_title(
        f"Service: {service_name.upper()}  |  Window: {time_from} → {time_to}",
        fontsize=13, weight="bold", pad=12,
    )
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[3-VISUALIZE] Saved  -> {output_path}")

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def save_reports(
    mappings: List[LogMapping],
    service_name: str,
    time_from: int,
    time_to: int,
    csv_out: str = "log_analysis_report.csv",
    txt_out: str = "log_analysis_report.txt",
) -> None:
    print(f"[3-VISUALIZE] Reports -> {csv_out}, {txt_out}")

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "service", "timestamp", "method", "message",
                    "template", "call_node_id", "method_node_id", "matched", "score"])
        for m in mappings:
            parts = m.bucket.split("@")
            ts  = parts[1] if len(parts) == 2 else ""
            svc = parts[0] if len(parts) == 2 else m.bucket
            w.writerow([
                m.bucket, svc, ts,
                m.method_name, m.message[:120],
                m.template[:120], m.call_node_id, m.method_node_id,
                "Yes" if m.matched else "No", m.score,
            ])

    matched = sum(1 for m in mappings if m.matched)
    with open(txt_out, "w", encoding="utf-8") as f:
        sep = "=" * 110
        f.write(sep + "\n")
        f.write("LOG ANALYSIS REPORT\n")
        f.write(sep + "\n\n")
        f.write(f"Service    : {service_name.upper()}\n")
        f.write(f"Window     : {time_from} -> {time_to}  ({time_to - time_from}s)\n")
        f.write(f"Total logs : {len(mappings)}\n")
        f.write(f"Matched    : {matched}  ({100*matched/len(mappings) if mappings else 0:.1f}%)\n")
        f.write(f"Unmatched  : {len(mappings) - matched}\n\n")
        f.write("-" * 110 + "\n")
        f.write("LOG SEQUENCE\n")
        f.write("-" * 110 + "\n\n")
        for i, m in enumerate(mappings, 1):
            status = "OK" if m.matched else "!!"
            f.write(f"{i:>3}. [{status}] {m.bucket}\n")
            f.write(f"       Method  : {m.method_name}\n")
            f.write(f"       Message : {m.message}\n")
            if m.matched:
                f.write(f"       Template: {m.template}\n")
                f.write(f"       CALL id : {m.call_node_id}\n")
            f.write("\n")
        f.write(sep + "\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipe.py",
        description="Log Pipeline: Extract -> Match (Trie/CPG) -> Visualize",
    )
    p.add_argument("--dataset", "-d",  required=True, help="CSV log file (timestamp,service,message)")
    p.add_argument("--service", "-s",  required=True, help="Service name to analyze")
    p.add_argument("--cpg",     "-c",  required=True, help="CPG .dot file for the service")
    p.add_argument("--start",   "-st", required=True, type=int, help="Window start (epoch seconds)")
    p.add_argument("--end",     "-e",  required=True, type=int, help="Window end   (epoch seconds)")
    return p

def main() -> None:
    args = build_parser().parse_args()

    if not Path(args.dataset).exists():
        print(f"ERROR: --dataset not found: {args.dataset}"); sys.exit(1)
    if not Path(args.cpg).exists():
        print(f"ERROR: --cpg not found: {args.cpg}"); sys.exit(1)
    if args.start >= args.end:
        print(f"ERROR: --start must be < --end"); sys.exit(1)

    print("=" * 110)
    print("PIPELINE  Extract -> Match (Trie/CPG) -> Visualize")
    print("=" * 110)

    # Stage 1
    log_rows = extract_logs(args.dataset, args.service, args.start, args.end)
    if not log_rows:
        sys.exit(1)

    # Stage 2
    mappings = match_stage(args.cpg, log_rows, trie_image="trie.png")

    # Stage 3
    visualize_timeline(mappings, args.service, args.start, args.end)
    save_reports(mappings, args.service, args.start, args.end)

    matched = sum(1 for m in mappings if m.matched)
    print()
    print("=" * 110)
    print("DONE")
    print(f"  Service   : {args.service.upper()}")
    print(f"  Window    : {args.start} -> {args.end}")
    print(f"  Logs      : {len(mappings)}")
    print(f"  Matched   : {matched} ({100*matched/len(mappings) if mappings else 0:.1f}%)")
    print()
    print("  Files:")
    print("    trie.png                — Trie visualization")
    print("    log_timeline.png        — Timeline visualization")
    print("    log_analysis_report.csv — Structured report")
    print("    log_analysis_report.txt — Readable report")
    print("=" * 110)

if __name__ == "__main__":
    main()
