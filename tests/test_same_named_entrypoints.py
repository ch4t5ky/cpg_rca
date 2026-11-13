"""Same-named entrypoints through builder, runtime catalog and rendering (detector, FLOW and FSM are fakes)."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import builder  # noqa: E402
import runtime_analyzer  # noqa: E402
import visualize  # noqa: E402

# "__init__" and "init" differ, but visualize.safe_name() would map both to "init".
NAMES = ["main", "main", "__init__", "init"]


def build(tmp_path, monkeypatch):
    eps = [SimpleNamespace(name=n, node_id=str(i), full_name=n) for i, n in enumerate(NAMES)]
    monkeypatch.setattr(builder, "EntrypointDetector", lambda g: SimpleNamespace(detect=lambda: eps))
    monkeypatch.setattr(builder, "EntrypointFlow", lambda g, max_depth, max_paths: SimpleNamespace(
        build=lambda nid: SimpleNamespace(id=nid, name=NAMES[int(nid)], semantic_graphs={})))
    monkeypatch.setattr(builder, "LogFlowExtractor", lambda t: SimpleNamespace(
        extract=lambda flow: SimpleNamespace(id=flow.id, name=flow.name, states=[1], transitions=[1], warnings=[])))
    monkeypatch.setattr(builder, "serialize_flow", lambda flow: {"entrypoint_node_id": flow.id, "entrypoint_name": flow.name})
    monkeypatch.setattr(builder, "serialize_fsm", lambda fsm: {
        "entrypoint_node_id": fsm.id, "entrypoint_name": fsm.name, "states": [{"id": "s"}],
        "transitions": [{"id": "t", "source_segment_id": "s", "target_segment_id": "s"}]})
    out = tmp_path / "svc"
    return out, builder.build_entrypoint_artifacts(None, [], out)


def test_same_named_entrypoints_get_distinct_files(tmp_path, monkeypatch):
    out, index = build(tmp_path, monkeypatch)
    assert [r["flow_file"] for r in index] == ["main.flow.json", "main__1.flow.json", "__init__.flow.json", "init.flow.json"]
    for i, row in enumerate(index):
        assert json.loads((out / row["flow_file"]).read_text())["entrypoint_node_id"] == str(i)


def test_runtime_catalog_pairs_each_fsm_with_its_own_flow(tmp_path, monkeypatch):
    out, _ = build(tmp_path, monkeypatch)
    args = argparse.Namespace(threshold=0.5, time_gap_sec=60, max_active_chains=10,
                              max_hypotheses_per_chain=10, max_total_active_hypotheses=100)
    catalog = runtime_analyzer.load_service_catalog(out, args)
    assert len(catalog.fsms) == 4
    for key, fsm in catalog.fsms.items():
        assert catalog.flows[key]["entrypoint_node_id"] == fsm.entrypoint_node_id


def test_render_paths_are_distinct(tmp_path, monkeypatch):
    out, _ = build(tmp_path, monkeypatch)
    paths = []
    monkeypatch.setattr(visualize, "load_flow_graphs", lambda p: ("x", {"m": SimpleNamespace(nodes={}, edges=[])}))
    monkeypatch.setattr(visualize, "draw_semantic_graph_graphviz", lambda semantic_graph, filename, output_dir, fmt, rankdir: paths.append(("flow", output_dir, filename)))
    monkeypatch.setattr(visualize, "draw_log_fsm_graphviz", lambda fsm, filename, output_dir, fmt, rankdir: paths.append(("fsm", output_dir, filename)))
    visualize.render_service_artifacts(out, tmp_path / "render")
    assert len(paths) == len(set(paths)) == 8
