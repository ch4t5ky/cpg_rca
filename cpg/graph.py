"""
cpg.graph
~~~~~~~~~
CodePropertyGraph — indexed, read-only wrapper around a Joern nx.MultiDiGraph.

Responsibilities
----------------
* Decode raw DOT attribute values (HTML-entities, quoted strings).
* Classify nodes: METHOD vs CALL, internal vs external.
* Provide O(1) FULLNAME → node-id lookup via a pre-built index.
* Act as the single gateway to the raw graph for all other cpg classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, Optional

import networkx as nx

from .constants import (
    ATTR_FULLNAME,
    ATTR_IS_EXTERNAL,
    ATTR_LABEL,
    DEFAULT_PROJECT_PREFIXES,
    LABEL_METHOD,
)
from .utils import unescape

if TYPE_CHECKING:
    from .method import Method


class CodePropertyGraph:
    """
    Low-level accessor for a Joern-generated Code Property Graph.

    Parameters
    ----------
    graph            : nx.MultiDiGraph loaded from the service .dot export.
    project_prefixes : Tuple of FULLNAME prefixes that identify *internal*
                       (service-owned) methods.  Defaults to
                       ``DEFAULT_PROJECT_PREFIXES``.  Override per-service::

                           cpg = CodePropertyGraph(G, project_prefixes=("main.",))
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        project_prefixes: tuple[str, ...] = DEFAULT_PROJECT_PREFIXES,
    ) -> None:
        self._g        = graph
        self._prefixes = project_prefixes

        # Pre-build a FULLNAME → node-id index for all internal METHOD nodes.
        # Computed once; every lookup is O(1) afterwards.
        self._method_by_fullname: Dict[str, str] = {
            self.attr(nid, ATTR_FULLNAME): nid
            for nid in graph.nodes
            if self.is_internal_method(nid)
        }

    # ------------------------------------------------------------------
    # Core attribute access
    # ------------------------------------------------------------------

    @property
    def raw_graph(self) -> nx.MultiDiGraph:
        """The underlying networkx graph (treat as read-only)."""
        return self._g

    def attr(self, node_id: str, key: str) -> str:
        """Return the decoded value of *key* for *node_id*, or '' if absent."""
        raw = self._g.nodes.get(node_id, {}).get(key, "")
        return unescape(raw)

    def label(self, node_id: str) -> str:
        """Upper-cased node label (e.g. 'METHOD', 'CALL', 'IDENTIFIER')."""
        return self.attr(node_id, ATTR_LABEL).upper()

    # ------------------------------------------------------------------
    # Node classification
    # ------------------------------------------------------------------

    def is_internal_method(self, node_id: str) -> bool:
        """
        Return True if *node_id* is a METHOD node owned by this service.

        A node is *internal* when:
        - its label is ``METHOD``,
        - its FULLNAME is non-empty and does not start with ``<``,
        - ``IS_EXTERNAL`` is not ``true``,
        - its FULLNAME starts with one of ``project_prefixes``.
        """
        if self.label(node_id) != LABEL_METHOD:
            return False
        fn = self.attr(node_id, ATTR_FULLNAME)
        if not fn or fn.startswith("<"):
            return False
        if self.attr(node_id, ATTR_IS_EXTERNAL).lower() == "true":
            return False
        return fn.startswith(self._prefixes)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def find_method_node(self, fullname: str) -> Optional[str]:
        """Return the node-id for a known internal method FULLNAME, or None."""
        return self._method_by_fullname.get(fullname)

    def iter_internal_methods(self) -> Iterator[str]:
        """Yield node-ids of every internal METHOD node."""
        yield from self._method_by_fullname.values()

    def resolve_method(self, node_id: str) -> "Method":
        """Build and return a :class:`~cpg.method.Method` for *node_id*."""
        from .method import Method
        return Method._build(self, node_id)
