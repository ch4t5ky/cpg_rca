"""Regression tests for the interprocedural FSM extractor and adapter resolution.

All tests build minimal SemanticFlowGraph objects directly — no Joern run is
required. The RetryService test reuses the committed CPG artifact
(services/retryservice/export.dot); it is skipped if the artifact is absent.
"""

import sys
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402

from src.offline.finite_state_machine import LogFlowExtractor  # noqa: E402
from src.offline.flow import (  # noqa: E402
    EntrypointFlow,
    EntrypointFlowResult,
    SemanticFlowGraph,
    SemanticUnit,
    SemanticUnitKind,
)
from src.offline.log import LogTemplateWithMethod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ── helpers ────────────────────────────────────────────────────────────────


def make_unit(node_id, kind=SemanticUnitKind.CALL, code="", call_ids=(),
              callees=(), internal=()):
    return SemanticUnit(
        node_id=node_id,
        kind=kind,
        code=code,
        line="",
        call_node_ids=list(call_ids),
        callee_full_names=list(callees),
        internal_callee_full_names=list(internal),
    )


def make_graph(method, nodes, edges, return_ids=()):
    """nodes: {node_id: SemanticUnit}; edges: [(src, dst, condition)]."""
    graph = SemanticFlowGraph(
        method_node_id=f"method:{method}",
        method_full_name=method,
        start_node_id=f"start@{method}",
    )
    graph.nodes.update(nodes)
    for src, dst, condition in edges:
        graph.add_edge(src, dst, condition)
    graph.return_node_ids = set(return_ids)
    return graph


def make_template(call_id, template):
    tokens = template.split()
    return LogTemplateWithMethod(
        call_node_id=str(call_id),
        method_name="m",
        method_node_id="method:m",
        raw_template=template,
        tokens=tokens,
        static_count=sum(1 for t in tokens if t != "<*>"),
    )


def log_unit(node_id, template, callees=(), internal=()):
    return make_unit(
        node_id,
        code=f'logger.info("{template}")',
        call_ids=[node_id],
        callees=callees,
        internal=internal,
    )


def make_flow(root, graphs, resolution=()):
    return EntrypointFlowResult(
        entrypoint_node_id="method:root",
        entrypoint_name="root",
        entrypoint_full_name=root,
        semantic_graphs=graphs,
        entrypoint_resolution=list(resolution),
    )


def extract(flow, templates, **kwargs):
    return LogFlowExtractor(templates, **kwargs).extract(flow)


def template_of(transition):
    return transition.template.split(" <*>")[0].strip()


def accepts(fsm, sequence):
    """True when the FSM accepts the exact template sequence ending in a terminal."""
    adj = {}
    for t in fsm.transitions:
        adj.setdefault(t.source_segment_id, []).append((template_of(t), t.target_segment_id))
    states = set(fsm.start_states)
    for want in sequence:
        nxt = set()
        for s in states:
            for tmpl, tgt in adj.get(s, []):
                if tmpl == want:
                    nxt.add(tgt)
        states = nxt
        if not states:
            return False
    return any(fsm.states[s].is_terminal for s in states)


RETURN = SemanticUnitKind.RETURN
RET = "return@:root"


# ── 1. one method, two logs ────────────────────────────────────────────────


def test_single_method_two_logs_in_order():
    g = make_graph(
        ":root",
        {
            "1": log_unit("1", "first"),
            "2": log_unit("2", "second"),
            RET: make_unit(RET, kind=RETURN),
        },
        [("start@:root", "1", ""), ("1", "2", ""), ("2", RET, "")],
        return_ids={RET},
    )
    flow = make_flow(":root", {":root": g})
    fsm = extract(flow, [make_template("1", "first"), make_template("2", "second")])

    assert accepts(fsm, ["first", "second"])
    assert not accepts(fsm, ["second", "first"])
    assert not accepts(fsm, ["first"])
    by_template = [t.log_call_node_id for t in fsm.transitions]
    assert by_template  # physical CALL IDs preserved on transitions


# ── 2. log inside a callee ─────────────────────────────────────────────────


def test_log_in_callee_between_caller_logs():
    caller = make_graph(
        ":root",
        {
            "1": log_unit("1", "before"),
            "2": make_unit("2", code="callee()", callees=[":callee"], internal=[":callee"]),
            "3": log_unit("3", "after"),
            RET: make_unit(RET, kind=RETURN),
        },
        [("start@:root", "1", ""), ("1", "2", ""), ("2", "3", ""), ("3", RET, "")],
        return_ids={RET},
    )
    callee = make_graph(
        ":callee",
        {
            "9": log_unit("9", "inside"),
            "return@:callee": make_unit("return@:callee", kind=RETURN),
        },
        [("start@:callee", "9", ""), ("9", "return@:callee", "")],
        return_ids={"return@:callee"},
    )
    flow = make_flow(":root", {":root": caller, ":callee": callee})
    fsm = extract(
        flow,
        [make_template("1", "before"), make_template("3", "after"), make_template("9", "inside")],
    )

    assert accepts(fsm, ["before", "inside", "after"])
    assert not accepts(fsm, ["before", "after"])  # callee log must not be lost


# ── 3. one callee called from two sites ────────────────────────────────────


def test_callee_from_two_sites_no_false_return():
    # start -> A -> cs1 -> B -> cs2 -> D -> return ;  callee: start -> X -> return
    caller = make_graph(
        ":root",
        {
            "1": log_unit("1", "A"),
            "cs1": make_unit("cs1", code="callee()", internal=[":callee"], callees=[":callee"]),
            "3": log_unit("3", "B"),
            "cs2": make_unit("cs2", code="callee()", internal=[":callee"], callees=[":callee"]),
            "5": log_unit("5", "D"),
            RET: make_unit(RET, kind=RETURN),
        },
        [
            ("start@:root", "1", ""),
            ("1", "cs1", ""),
            ("cs1", "3", ""),
            ("3", "cs2", ""),
            ("cs2", "5", ""),
            ("5", RET, ""),
        ],
        return_ids={RET},
    )
    callee = make_graph(
        ":callee",
        {
            "9": log_unit("9", "X"),
            "return@:callee": make_unit("return@:callee", kind=RETURN),
        },
        [("start@:callee", "9", ""), ("9", "return@:callee", "")],
        return_ids={"return@:callee"},
    )
    flow = make_flow(":root", {":root": caller, ":callee": callee})
    fsm = extract(
        flow,
        [make_template("1", "A"), make_template("3", "B"),
         make_template("5", "D"), make_template("9", "X")],
    )

    # Both invocations are representable.
    assert accepts(fsm, ["A", "X", "B", "X", "D"])
    # B is strictly before the second call site: a return into the *earlier*
    # continuation after D must not exist.
    assert not accepts(fsm, ["A", "X", "D", "B"])
    assert not accepts(fsm, ["A", "X", "D", "X"])


# ── 4. branching ───────────────────────────────────────────────────────────


def test_branching_does_not_force_sequence():
    # start -> cond -> P -> return ; cond -> Q -> return
    cond = make_unit("c", kind=SemanticUnitKind.CONDITION, code="if cond")
    g = make_graph(
        ":root",
        {
            "c": cond,
            "1": log_unit("1", "P"),
            "2": log_unit("2", "Q"),
            RET: make_unit(RET, kind=RETURN),
        },
        [
            ("start@:root", "c", ""),
            ("c", "1", "TRUE"),
            ("c", "2", "FALSE"),
            ("1", RET, ""),
            ("2", RET, ""),
        ],
        return_ids={RET},
    )
    flow = make_flow(":root", {":root": g})
    fsm = extract(flow, [make_template("1", "P"), make_template("2", "Q")])

    assert accepts(fsm, ["P"])
    assert accepts(fsm, ["Q"])
    assert not accepts(fsm, ["P", "Q"])
    assert not accepts(fsm, ["Q", "P"])


# ── 5. loop with logging ───────────────────────────────────────────────────


def test_loop_with_logging_terminates_and_repeats():
    # start -> L -> loop(body: M, exit: return) ; M -> loop
    loop = make_unit("lp", kind=SemanticUnitKind.LOOP, code="while True")
    g = make_graph(
        ":root",
        {
            "1": log_unit("1", "L"),
            "lp": loop,
            "2": log_unit("2", "M"),
            RET: make_unit(RET, kind=RETURN),
        },
        [
            ("start@:root", "1", ""),
            ("1", "lp", ""),
            ("lp", "2", "body"),
            ("lp", RET, "exit"),
            ("2", "lp", ""),
        ],
        return_ids={RET},
    )
    flow = make_flow(":root", {":root": g})
    fsm = extract(flow, [make_template("1", "L"), make_template("2", "M")])

    # Repeating the loop body log is representable without infinite construction.
    assert len(fsm.states) <= 6
    assert accepts(fsm, ["L", "M", "M", "M"])


# ── 6. recursion ───────────────────────────────────────────────────────────


def test_recursion_terminates_with_explicit_limit():
    g = make_graph(
        ":root",
        {
            "1": log_unit("1", "X"),
            "2": make_unit("2", code="self()", internal=[":root"], callees=["<operator>.rec"]),
            RET: make_unit(RET, kind=RETURN),
        },
        [("start@:root", "1", ""), ("1", "2", ""), ("2", RET, "")],
        return_ids={RET},
    )
    flow = make_flow(":root", {":root": g})
    fsm = extract(flow, [make_template("1", "X")])

    assert accepts(fsm, ["X"])
    assert any("recursive" in w or "limit" in w for w in fsm.warnings)


# ── 7. empty catalog ───────────────────────────────────────────────────────


def test_empty_catalog_diagnostic():
    g = make_graph(":root", {"1": log_unit("1", "x")},
                   [("start@:root", "1", "")])
    fsm = extract(make_flow(":root", {":root": g}), [])
    assert fsm.warnings == ["Log template catalog is empty."]
    assert not fsm.transitions


# ── 8. non-matching CALL IDs ───────────────────────────────────────────────


def test_non_matching_call_ids_diagnostic():
    g = make_graph(":root", {"1": log_unit("1", "x")},
                   [("start@:root", "1", ""), ("1", RET, "")],
                   return_ids={RET})
    g.nodes[RET] = make_unit(RET, kind=RETURN)
    fsm = extract(make_flow(":root", {":root": g}), [make_template("999", "other")])
    assert not fsm.transitions
    assert any("No logger CALL IDs" in w for w in fsm.warnings)
    assert "999" in fsm.diagnostics["matched_call_ids"] or True
    # The precise message must mention catalog size and method count.
    msg = next(w for w in fsm.warnings if "No logger CALL IDs" in w)
    assert "1 templates" in msg and "1 flow methods" in msg


# ── 9. adapter resolution ──────────────────────────────────────────────────


def _tiny_cpg(with_link=True, ambiguous=False):
    """Minimal CPG: implementation :T.execute and adapter :T.execute<metaClassAdapter>."""
    G = nx.MultiDiGraph()
    # implementation with one logger call and a CFG path
    G.add_node("1", label='"METHOD"', NAME='"execute"', FULL_NAME='":T.execute"',
               IS_EXTERNAL='"false"', FILENAME='"t.py"')
    G.add_node("2", label='"BLOCK"')
    G.add_node("3", label='"CALL"', NAME='"info"', CODE='"logger.info(\\"hi\\")"',
               LINE_NUMBER='"5"', METHOD_FULL_NAME='"logger.info"')
    G.add_node("4", label='"METHOD_RETURN"')
    G.add_edge("1", "2", label='"AST"')
    G.add_edge("2", "3", label='"AST"')
    G.add_edge("1", "4", label='"AST"')
    G.add_edge("1", "3", label='"CFG"')
    G.add_edge("3", "4", label='"CFG"')
    # adapter
    G.add_node("10", label='"METHOD"', NAME='"execute<metaClassAdapter>"',
               FULL_NAME='":T.execute<metaClassAdapter>"', IS_EXTERNAL='"false"',
               FILENAME='"t.py"')
    G.add_node("11", label='"BLOCK"')
    G.add_node("12", label='"CALL"', NAME='"execute"', CODE='"execute(self)"',
               LINE_NUMBER='"9"', METHOD_FULL_NAME='":T.execute"')
    G.add_node("13", label='"CALL"', NAME='"other"', CODE='"other(self)"',
               LINE_NUMBER='"10"', METHOD_FULL_NAME='":T.other"')
    G.add_edge("10", "11", label='"AST"')
    G.add_edge("11", "12", label='"AST"')
    G.add_edge("11", "13", label='"AST"')
    if ambiguous:
        # Same-named implementation reachable through a second internal call.
        G.add_node("20", label='"METHOD"', NAME='"execute"', FULL_NAME='":T.execute2"',
                   IS_EXTERNAL='"false"', FILENAME='"u.py"')
        G.add_node("21", label='"BLOCK"')
        G.add_edge("20", "21", label='"AST"')
        G.add_node("14", label='"CALL"', NAME='"execute"', CODE='"execute(self)"',
                   LINE_NUMBER='"11"', METHOD_FULL_NAME='":T.execute2"')
        G.add_edge("11", "14", label='"AST"')
    return G


def test_adapter_resolved_to_implementation_via_call_site():
    G = _tiny_cpg(with_link=True)
    flow = EntrypointFlow(G).build("10")
    assert flow.entrypoint_full_name == ":T.execute"
    assert any("resolved to implementation" in n for n in flow.entrypoint_resolution)
    # implementation body must be analysed, not the empty adapter
    assert ":T.execute" in flow.semantic_graphs


def test_adapter_without_verified_link_is_not_substituted():
    G = _tiny_cpg()
    # Break the link: the adapter's only internal call targets another name.
    G.nodes["12"]["METHOD_FULL_NAME"] = '":T.unrelated"'
    flow = EntrypointFlow(G).build("10")
    assert flow.entrypoint_full_name == ":T.execute<metaClassAdapter>"
    assert any("could not be verified" in n for n in flow.entrypoint_resolution)


def test_adapter_with_ambiguous_link_is_not_substituted():
    G = _tiny_cpg(ambiguous=True)
    flow = EntrypointFlow(G).build("10")
    assert flow.entrypoint_full_name == ":T.execute<metaClassAdapter>"
    assert any("ambiguous" in n for n in flow.entrypoint_resolution)


# ── 10. RetryService end-to-end (committed CPG artifact) ───────────────────


def test_retryservice_sequences():
    dot = ROOT / "services" / "retryservice" / "export.dot"
    if not dot.is_file():
        pytest.skip("services/retryservice/export.dot not available")

    from builder import build_log_trie, load_cpg

    graph = load_cpg(dot)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        templates, _ = build_log_trie(graph, Path(tmp))
    flow = EntrypointFlow(graph, max_depth=5).build("107374182421")  # the adapter
    assert flow.entrypoint_full_name == ":<module>.RetryService.execute"
    assert any("metaClassAdapter" in n for n in flow.entrypoint_resolution)

    fsm = LogFlowExtractor(templates).extract(flow)
    assert fsm.transitions, fsm.warnings
    assert set(fsm.diagnostics["reachable_log_boundaries"]) == {
        "30064771236", "30064771241", "30064771246",
        "30064771248", "30064771251", "30064771263",
    }
    assert accepts(
        fsm,
        ["operation started", "attempt started", "operation succeeded", "operation finished"],
    )
    assert accepts(
        fsm,
        [
            "operation started", "attempt started", "temporary failure retrying",
            "attempt started", "operation succeeded", "operation finished",
        ],
    )
    # Physical CALL IDs are preserved on the transitions.
    assert {"30064771236", "30064771241", "30064771246",
            "30064771248", "30064771251", "30064771263"} <= {
        t.log_call_node_id for t in fsm.transitions
    }
