import os
import textwrap

import graphviz

from src.offline.flow import SemanticFlowGraph, SemanticUnit, SemanticUnitKind


__all__ = ["draw_semantic_graph_graphviz"]


def draw_semantic_graph_graphviz(
    semantic_graph: SemanticFlowGraph,
    filename: str | None = None,
    output_dir: str = "output",
    fmt: str = "png",
    rankdir: str = "LR",
    max_code_length: int = 58,
) -> str:
    """
    Render one SemanticFlowGraph with Graphviz and return the generated file.

    Visual semantics:
    - START: green ellipse with input parameters.
    - CONDITION: green diamond.
    - LOOP: green hexagon.
    - CALL with internal callees: blue ellipse.
    - RETURN: red ellipse with output parameters.
    - Other semantic operations: purple ellipse.

    Edge labels preserve CFG conditions such as TRUE, FALSE, body, and
    next iteration. The renderer is intentionally generic: it can render a
    full FLOW artifact or a future runtime-constrained FLOW slice unchanged.
    """
    graph = graphviz.Digraph(
        "semantic_flow",
        graph_attr={
            "rankdir": rankdir,
            "splines": "spline",
            "nodesep": "0.35",
            "ranksep": "0.80",
            "fontname": "Helvetica",
            "label": f"{semantic_graph.method_full_name}\nSemantic control-flow graph",
            "labelloc": "t",
            "bgcolor": "#ffffff",
            "fontcolor": "#0f172a",
            "fontsize": "19",
        },
        node_attr={
            "style": "filled",
            "fontname": "Helvetica",
            "fontsize": "10",
            "fontcolor": "#0f172a",
            "margin": "0.16,0.10",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
            "fontcolor": "#334155",
            "penwidth": "1.5",
            "color": "#94a3b8",
        },
        format=fmt,
    )

    def format_parameters(parameters: list[tuple[str, str]]) -> str:
        if not parameters:
            return "—"
        return "\\n".join(
            f"{name}: {type_name}" if name else type_name
            for name, type_name in parameters
        )

    def start_label() -> str:
        return f"START\\nIN:\\n{format_parameters(semantic_graph.input_parameters)}"

    def return_label() -> str:
        return f"RETURN\\nOUT:\\n{format_parameters(semantic_graph.output_parameters)}"

    graph.node(
        semantic_graph.start_node_id,
        label=start_label(),
        shape="ellipse",
        fillcolor="#059669",
        color="#6ee7b7",
        fontcolor="#ffffff",
        penwidth="2",
    )

    def short_method(full_name: str) -> str:
        return full_name.rsplit(".", 1)[-1] if "." in full_name else full_name

    def compact_code(code: str) -> str:
        normalized = " ".join((code or "").split())
        return textwrap.shorten(normalized, width=max_code_length, placeholder=" …")

    def unit_style(unit: SemanticUnit) -> tuple[str, str, str]:
        if unit.kind is SemanticUnitKind.CONDITION:
            return "diamond", "#DCFCE7", "#16A34A"
        if unit.kind is SemanticUnitKind.LOOP:
            return "hexagon", "#D1FAE5", "#15803D"
        if unit.kind is SemanticUnitKind.RETURN:
            return "ellipse", "#FEE2E2", "#DC2626"
        if unit.internal_callee_full_names:
            return "ellipse", "#DBEAFE", "#2563EB"
        return "ellipse", "#EDE9FE", "#7C3AED"

    def unit_label(unit: SemanticUnit) -> str:
        if unit.kind is SemanticUnitKind.RETURN:
            return return_label()

        if unit.internal_callee_full_names:
            callees = ", ".join(short_method(name) for name in unit.internal_callee_full_names)
            return f"CALL INTERNAL\\n{callees}"

        title = unit.kind.value.upper()
        code = compact_code(unit.code)
        return f"{title}\\n{code}" if code else title

    for unit_id, unit in semantic_graph.nodes.items():
        shape, fillcolor, border = unit_style(unit)
        node_kwargs = {
            "label": unit_label(unit),
            "shape": shape,
            "fillcolor": fillcolor,
            "color": border,
        }
        if unit.kind is SemanticUnitKind.RETURN:
            node_kwargs.update({"fontcolor": "#991B1B", "penwidth": "2.0"})
        graph.node(unit_id, **node_kwargs)

    return_ids = [
        unit_id
        for unit_id, unit in semantic_graph.nodes.items()
        if unit.kind is SemanticUnitKind.RETURN
    ]
    if return_ids:
        with graph.subgraph() as terminal_rank:
            terminal_rank.attr(rank="sink")
            for return_id in return_ids:
                terminal_rank.node(return_id)

    edge_colors = {
        "TRUE": "#22c55e",
        "FALSE": "#ef4444",
        "body": "#22c55e",
        "next iteration": "#f59e0b",
    }
    for edge in semantic_graph.edge_list:
        graph.edge(
            edge.source_id,
            edge.target_id,
            label=edge.condition or None,
            color=edge_colors.get(edge.condition, "#94a3b8"),
        )

    os.makedirs(output_dir, exist_ok=True)
    name = filename or f"semantic_{short_method(semantic_graph.method_full_name)}"
    return graph.render(filename=name, directory=output_dir, cleanup=True)