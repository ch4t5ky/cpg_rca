"""Log observability coverage of the builder.py output (standard library only).

Log point: a logger call with a template in trie.json. For one entrypoint E:
  reachable  log points in the methods of the FLOW of E
  visible    reachable log points that are transitions of the FSM of E
  lost       reachable - visible; "in entry" if the log point is in the entry
             method of E, "in callee" otherwise
  zero-log   E has no visible log point
Per service, "union" counts distinct log points over all entrypoints and
"pooled" adds up the counts of each entrypoint.

FLOW reachability is a lower bound (unresolved calls are not followed), so a
smaller FLOW lowers reachable and lost together: compare reachable too.

Usage: python3 -m eval.coverage ARTIFACTS_DIR OUTPUT_DIR
"""

import json
import subprocess
import sys
from pathlib import Path

FIELDS = ("entrypoints", "log_points", "reachable", "visible", "lost", "pooled_reachable",
          "pooled_visible", "pooled_lost", "pooled_lost_callee", "pooled_lost_entry", "zero_log")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def service_metrics(service_dir):
    service_dir = Path(service_dir)
    points = {str(t["call_node_id"]): str(t["method_node_id"])
              for t in load(service_dir / "trie.json")["templates"]}
    m = dict.fromkeys(FIELDS, 0)
    m["log_points"] = len(points)
    reach_union, visible_union = set(), set()
    for row in load(service_dir / "index.json"):
        if row["status"] != "ok":
            continue
        flow = load(service_dir / row["flow_file"])
        fsm = load(service_dir / row["fsm_file"])
        flow_methods = {str(x["method_node_id"]) for x in flow["methods"].values()}
        reachable = {c for c, method in points.items() if method in flow_methods}
        visible = {str(t["log_call_node_id"]) for t in fsm["transitions"]} & reachable
        lost = reachable - visible
        in_entry = {c for c in lost if points[c] == str(flow["entrypoint_node_id"])}
        reach_union |= reachable
        visible_union |= visible
        m["entrypoints"] += 1
        m["pooled_reachable"] += len(reachable)
        m["pooled_visible"] += len(visible)
        m["pooled_lost"] += len(lost)
        m["pooled_lost_entry"] += len(in_entry)
        m["pooled_lost_callee"] += len(lost) - len(in_entry)
        m["zero_log"] += not visible
    m["reachable"], m["visible"] = len(reach_union), len(visible_union)
    m["lost"] = len(reach_union - visible_union)
    return m


def snapshot(artifacts_dir, args):
    root = Path(artifacts_dir)
    services = {p.parent.name: service_metrics(p.parent) for p in sorted(root.glob("*/index.json"))}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                            capture_output=True, text=True).stdout.strip()
    return {"meta": {"commit": commit, "args": args}, "services": services,
            "totals": {k: sum(s[k] for s in services.values()) for k in FIELDS}}


def markdown(snap):
    lines = ["| service | " + " | ".join(k.replace("_", " ") for k in FIELDS) + " |",
             "|" + "---|" * (len(FIELDS) + 1)]
    for name, s in [*snap["services"].items(), ("total", snap["totals"])]:
        lines.append(f"| {name} | " + " | ".join(str(s[k]) for k in FIELDS) + " |")
    return "\n".join(lines) + "\n"


def main(argv):
    artifacts_dir, out = argv
    snap = snapshot(artifacts_dir, argv)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.json").write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out / "coverage.md").write_text(markdown(snap), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
