"""Coverage metrics and the compare exit code on a tiny synthetic builder output.

Log points: l1, l2 in Entry [100], l3 in Callee [101], l4 in Orphan [999] (no FLOW reaches it).
E1 enters Entry, calls Callee, its FSM has l1.  E2 enters Entry2 [102], calls Callee, its FSM is empty.
"""

import copy
import json

from eval import compare, coverage

LOG_POINTS = {"1": "100", "2": "100", "3": "101", "4": "999"}  # call node id -> method node id
ENTRYPOINTS = {"E1": ("100", ["101"], ["1"]), "E2": ("102", ["101"], [])}  # entry, callees, visible


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def make_service(root):
    svc = root / "svc"
    svc.mkdir(exist_ok=True)
    write(svc / "trie.json", {"templates": [{"call_node_id": c, "method_node_id": m} for c, m in LOG_POINTS.items()]})
    index = []
    for name, (entry, callees, visible) in ENTRYPOINTS.items():
        methods = {f"m{m}": {"method_node_id": m} for m in [entry, *callees]}
        write(svc / f"{name}.flow.json", {"entrypoint_node_id": entry, "methods": methods})
        write(svc / f"{name}.fsm.json", {"transitions": [{"log_call_node_id": c} for c in visible]})
        index.append({"status": "ok", "flow_file": f"{name}.flow.json", "fsm_file": f"{name}.fsm.json"})
    write(svc / "index.json", index)


def test_metrics(tmp_path):
    make_service(tmp_path)
    m = coverage.snapshot(tmp_path, [])["services"]["svc"]
    assert m == {"entrypoints": 2, "log_points": 4, "reachable": 3, "visible": 1, "lost": 2,
                 "pooled_reachable": 4, "pooled_visible": 1, "pooled_lost": 3,
                 "pooled_lost_callee": 2, "pooled_lost_entry": 1, "zero_log": 1}


def run_compare(tmp_path, new):
    old_path, new_path = tmp_path / "old.json", tmp_path / "new.json"
    make_service(tmp_path)
    old = coverage.snapshot(tmp_path, [])
    write(old_path, old)
    write(new_path, new(copy.deepcopy(old)))
    return compare.main([str(old_path), str(new_path)])


def test_compare_exit_codes(tmp_path):
    assert run_compare(tmp_path, lambda s: s) == 0

    def less_visible(s):
        s["services"]["svc"]["visible"] -= 1
        return s

    def more_lost(s):
        s["totals"]["pooled_lost"] += 1
        return s

    assert run_compare(tmp_path, less_visible) == 1
    assert run_compare(tmp_path, more_lost) == 1
