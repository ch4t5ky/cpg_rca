"""
log2cpg2.py
===========
CPG processing module.

Design rules
------------
1. No module-level graph cache — the caller decides lifetime.
2. `process_service_cpg()` is the single entry point for consuming a CPG:
      load → index → extract activity + call-edges → FREE graph from memory.
   The large nx.MultiDiGraph never outlives that one call.
3. Only methods that appear in the supplied log messages are fully resolved;
   all other nodes are never walked.

Public API
----------
    process_service_cpg(dot_path, service_name, log_rows)
        → (ActivityPartial, CallEdges)

    # Lower-level helpers (used internally, exposed for testing)
    build_method_index(G, service_name) → MethodIndex
    fast_match(message, index)          → dict
    extract_call_edges(index)           → List[CallEdge]
"""

from __future__ import annotations

import gc
import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
from networkx.drawing.nx_pydot import read_dot

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRING_LITERAL_RE = re.compile(r'["`\'](.*?)["`\']', re.DOTALL)

LOG_METHOD_NAMES: Set[str] = {
    "info", "warn", "warning", "debug", "error", "trace", "fatal", "log",
}
ANON_NAMES: Set[str] = {
    "<lambda>", "<anonymous>", "<arrow>", "<init>", "",
}

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# (service, method_name) → {bucket_str → count}
ActivityPartial = Dict[Tuple[str, str], Dict[str, int]]

# (caller "svc::method", callee "svc::method", kind)   kind ∈ CALL | RETURN
CallEdge = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class MethodRecord:
    node_id:      str
    name:         str
    full_name:    str
    line:         str
    tokens:       Set[str]       = field(default_factory=set)
    call_tokens:  Set[str]       = field(default_factory=set)
    callee_names: Set[str]       = field(default_factory=set)


@dataclass
class MethodIndex:
    service_name: str
    records:      List[MethodRecord]        = field(default_factory=list)
    token_map:    Dict[str, List[int]]      = field(default_factory=dict)
    # short name / full_name  →  record index  (O(1) exact lookup)
    name_map:     Dict[str, int]            = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph helpers  (work on MultiDiGraph; never convert — Joern uses parallel edges)
# ---------------------------------------------------------------------------

def _clean(v) -> str:
    return html.unescape(str(v).strip().strip('"'))


def _tokenize(s: str) -> Set[str]:
    s = html.unescape(s).lower()
    s = re.sub(r"[^a-z ]+", " ", s)
    return {t for t in s.split() if len(t) >= 3}


def _ast_children(G: nx.MultiDiGraph, n) -> List:
    return [v for _, v, d in G.edges(n, data=True)
            if _clean(d.get("label", "")) == "AST"]


def _ast_subtree(G: nx.MultiDiGraph, root) -> Set:
    """BFS over AST-labelled edges downward from *root*."""
    visited, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        stack.extend(_ast_children(G, n))
    return visited


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_method_index(G: nx.MultiDiGraph, service_name: str) -> MethodIndex:
    """
    Single pass over the CPG graph.
    Builds an inverted token index (for fast_match) and a name map
    (for call-edge resolution).  Only METHOD nodes with real names are kept.
    """
    idx = MethodIndex(service_name=service_name)

    for nid, attrs in G.nodes(data=True):
        if _clean(attrs.get("label", "")).upper() != "METHOD":
            continue

        name      = _clean(attrs.get("NAME", ""))
        full_name = _clean(attrs.get("FULL_NAME", ""))
        line      = _clean(attrs.get("LINE_NUMBER", ""))

        if not name or name in ANON_NAMES or name.startswith("<"):
            continue

        subtree       = _ast_subtree(G, nid)
        all_tokens:   Set[str] = _tokenize(name)
        call_tokens:  Set[str] = set()
        callee_names: Set[str] = set()

        for sub_nid in subtree:
            sub = G.nodes[sub_nid]
            code = _clean(sub.get("CODE", ""))
            if code:
                all_tokens.update(_tokenize(code))
                for lit in STRING_LITERAL_RE.findall(html.unescape(code)):
                    all_tokens.update(_tokenize(lit))

            if _clean(sub.get("label", "")).upper() == "CALL":
                callee = _clean(sub.get("NAME", ""))
                if callee and callee.lower() not in LOG_METHOD_NAMES:
                    callee_names.add(callee)
                ctoks = _tokenize(callee)
                call_tokens.update(ctoks)
                all_tokens.update(ctoks)

        rec_idx = len(idx.records)
        rec = MethodRecord(
            node_id=str(nid),
            name=name,
            full_name=full_name,
            line=line,
            tokens=all_tokens,
            call_tokens=call_tokens,
            callee_names=callee_names,
        )
        idx.records.append(rec)

        for tok in all_tokens:
            idx.token_map.setdefault(tok, []).append(rec_idx)
        idx.name_map[name] = rec_idx
        if full_name:
            idx.name_map[full_name] = rec_idx

    return idx


# ---------------------------------------------------------------------------
# Fast match
# ---------------------------------------------------------------------------

def fast_match(message: str, index: MethodIndex) -> dict:
    """
    Jaccard-similarity match of *message* against the index.
    Only candidates sharing ≥1 token are evaluated → O(k) per call.
    Returns a dict with function_name / matched / score etc.
    """
    UNKNOWN = dict(function_name="<unknown>", full_name="",
                   line="", score=0.0, matched=False)

    msg_toks = _tokenize(message)
    if not msg_toks or not index.records:
        return UNKNOWN

    candidates: Set[int] = set()
    for tok in msg_toks:
        candidates.update(index.token_map.get(tok, []))

    if not candidates:
        return UNKNOWN

    best_score, best_rec = 0.0, None
    for ci in candidates:
        rec   = index.records[ci]
        union = msg_toks | rec.tokens
        if union:
            score = len(msg_toks & rec.tokens) / len(union)
            if score > best_score:
                best_score, best_rec = score, rec

    if best_rec is None or best_score == 0.0:
        return UNKNOWN

    return dict(
        function_name=best_rec.name,
        full_name=best_rec.full_name,
        line=best_rec.line,
        score=round(best_score, 4),
        matched=True,
    )


# ---------------------------------------------------------------------------
# Call-edge extraction
# ---------------------------------------------------------------------------

def extract_call_edges(index: MethodIndex) -> List[CallEdge]:
    """
    Resolve every callee_name in the index to a known method (intra-service only
    at this stage — inter-service resolution happens in timeline.py after all
    indexes have been built and stored in the cache).

    Returns list of (caller_q, callee_q, kind)  where kind ∈ {CALL, RETURN}.
    """
    edges: List[CallEdge] = []
    svc = index.service_name

    for rec in index.records:
        caller_q = f"{svc}::{rec.name}"
        for callee_raw in rec.callee_names:
            if callee_raw in index.name_map:
                callee_rec = index.records[index.name_map[callee_raw]]
                callee_q   = f"{svc}::{callee_rec.name}"
                edges.append((caller_q, callee_q, "CALL"))
                edges.append((callee_q, caller_q, "RETURN"))

    return edges


# ---------------------------------------------------------------------------
# Public entry point  ← this is what timeline.py calls
# ---------------------------------------------------------------------------

def process_service_cpg(
    dot_path: str,
    service_name: str,
    log_rows: List[Tuple[str, str]],   # [(bucket_str, message), ...]
) -> Tuple[ActivityPartial, List[CallEdge], MethodIndex]:
    """
    Load the CPG at *dot_path*, extract what is needed from *log_rows*,
    then **delete the graph from memory** before returning.

    Steps
    -----
    1. Read the DOT file into a MultiDiGraph.
    2. Build an in-memory MethodIndex (tokens, call-edges, callee names).
    3. Map every log message to its best-matching method → build ActivityPartial.
    4. Extract intra-service CALL / RETURN edges.
    5. del G  +  gc.collect()  — graph is freed here.
    6. Return (activity, call_edges, index).
       The index is lightweight (string data only, no graph references)
       and is kept so that inter-service call resolution can happen later
       in timeline.py without re-loading any DOT file.

    Parameters
    ----------
    dot_path     : path to export.dot for this service
    service_name : name used as the service label
    log_rows     : list of (bucket_str, message) for this service only

    Returns
    -------
    activity   : {(service_name, method_name): {bucket_str: count}}
    call_edges : intra-service CALL / RETURN edge list
    index      : lightweight MethodIndex (no graph, safe to cache / pickle)
    """
    if not Path(dot_path).exists():
        print(f"[CPG] MISSING: {dot_path}")
        return {}, [], MethodIndex(service_name=service_name)

    print(f"[CPG] Loading: {dot_path}")
    G = read_dot(dot_path)
    print(f"[CPG] {service_name}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    # Build index (pure Python objects — no graph pointers)
    index = build_method_index(G, service_name)
    print(f"[CPG] {service_name}: {len(index.records)} methods indexed")

    # Map log rows → activity counts
    from collections import defaultdict
    activity: Dict[Tuple[str, str], Dict[str, int]] = \
        defaultdict(lambda: defaultdict(int))

    for bucket, message in log_rows:
        result = fast_match(message, index)
        method = result["function_name"]   # "<unknown>" if no match
        activity[(service_name, method)][bucket] += 1

    # Intra-service call edges
    call_edges = extract_call_edges(index)

    # ── FREE THE GRAPH ────────────────────────────────────────────────────────
    del G
    gc.collect()
    print(f"[CPG] {service_name}: graph freed from memory")
    # ─────────────────────────────────────────────────────────────────────────

    return (
        {k: dict(v) for k, v in activity.items()},
        call_edges,
        index,
    )


# ---------------------------------------------------------------------------
# Inter-service call-edge resolution  (called AFTER all CPGs are processed)
# ---------------------------------------------------------------------------

def resolve_inter_service_edges(
    indexes: Dict[str, MethodIndex],
) -> List[CallEdge]:
    """
    Given all lightweight MethodIndexes (no graphs needed), find cross-service
    CALL / RETURN edges by matching each service's callee_names against every
    other service's name_map.

    Call this once after all process_service_cpg() calls are done.
    """
    edges: List[CallEdge] = []

    for src_svc, src_idx in indexes.items():
        for rec in src_idx.records:
            caller_q = f"{src_svc}::{rec.name}"
            for callee_raw in rec.callee_names:
                # Skip names already resolved intra-service
                if callee_raw in src_idx.name_map:
                    continue
                # Check every other service
                for dst_svc, dst_idx in indexes.items():
                    if dst_svc == src_svc:
                        continue
                    if callee_raw in dst_idx.name_map:
                        dst_rec  = dst_idx.records[dst_idx.name_map[callee_raw]]
                        callee_q = f"{dst_svc}::{dst_rec.name}"
                        edges.append((caller_q, callee_q, "CALL"))
                        edges.append((callee_q, caller_q, "RETURN"))
                        break   # first match wins

    return edges


# ---------------------------------------------------------------------------
# Call graph traversal  (get full call graph from a specific method)
# ---------------------------------------------------------------------------

def get_method_call_graph(
    method_name: str,
    service_name: str,
    all_edges: List[CallEdge],
    max_depth: int = None,
    include_potential_external: bool = False,
    indexes: Dict[str, MethodIndex] = None,
) -> Dict:
    """
    Get the full call graph starting from a specific method.
    
    Performs BFS traversal over CALL edges to find all methods reachable
    from the given root method. Separates internal (same-service) calls
    from external (cross-service) calls.
    
    **Internal vs External Classification:**
    - Internal: Both caller AND callee are in the ROOT service
    - External: Either caller OR callee is outside the ROOT service
    - Outbound: ROOT service calls external service
    - External Chain: External service calls another method (could be same or different external service)
    
    This means if frontend calls userservice, which then calls another
    userservice method, the frontend→userservice edge is "outbound" and 
    userservice→userservice is "external_chain" (both are also in "external").
    
    Parameters
    ----------
    method_name  : short method name (e.g., "handleRequest")
    service_name : service name (e.g., "frontend") - the ROOT service
    all_edges    : list of (caller_qualified, callee_qualified, kind) tuples
    max_depth    : optional maximum traversal depth (None = unlimited)
    include_potential_external : if True, also report unresolved callees that
                                 might be external calls (HTTP/gRPC/etc)
    indexes      : optional dict of {service: MethodIndex}, needed if
                   include_potential_external=True
    
    Returns
    -------
    dict with:
        'root'                  : str - qualified root method 'service::method'
        'root_service'          : str - the root service name
        'internal_calls'        : list[(caller, callee)] - both in root service
        'external_calls'        : list[(caller, callee)] - crosses service boundary
        'outbound_calls'        : list[(caller, callee)] - root → other service
        'external_chain_calls'  : list[(caller, callee)] - other → other service
        'all_reachable'         : list[str] - all qualified methods reachable
        'depth_map'             : dict[str, int] - qualified_method -> depth
        'call_count'            : int - total unique call edges found
        'services'              : set[str] - all services involved
        'potential_external'    : list[str] - unresolved callees (if enabled)
        'potential_external_calls' : list[(caller, callee_name)] - (if enabled)
    
    Example
    -------
    >>> edges = [
    ...     ("frontend::handleRequest", "frontend::getUser", "CALL"),
    ...     ("frontend::getUser", "userservice::fetchUser", "CALL"),
    ...     ("userservice::fetchUser", "userservice::queryDB", "CALL"),
    ... ]
    >>> result = get_method_call_graph("handleRequest", "frontend", edges)
    >>> result['internal_calls']
    [('frontend::handleRequest', 'frontend::getUser')]
    >>> result['outbound_calls']
    [('frontend::getUser', 'userservice::fetchUser')]
    >>> result['external_chain_calls']
    [('userservice::fetchUser', 'userservice::queryDB')]
    """
    root_qualified = f"{service_name}::{method_name}"
    
    # Build adjacency map: qualified_method -> [qualified_callees]
    call_map = {}
    for caller, callee, kind in all_edges:
        if kind == "CALL":
            call_map.setdefault(caller, []).append(callee)
    
    # BFS traversal
    visited = set()
    depth_map = {root_qualified: 0}
    queue = [(root_qualified, 0)]
    
    internal_calls = []        # root_svc -> root_svc
    external_calls = []        # any edge crossing service boundary
    outbound_calls = []        # root_svc -> other_svc  
    external_chain_calls = []  # other_svc -> other_svc (or any_svc)
    
    while queue:
        current, depth = queue.pop(0)
        
        if current in visited:
            continue
        visited.add(current)
        
        # Check depth limit
        if max_depth is not None and depth >= max_depth:
            continue
        
        # Get callees
        callees = call_map.get(current, [])
        
        for callee in callees:
            # Extract services
            caller_service = current.split("::")[0] if "::" in current else ""
            callee_service = callee.split("::")[0] if "::" in callee else ""
            
            # Classify based on ROOT service (service_name parameter)
            caller_is_root = (caller_service == service_name)
            callee_is_root = (callee_service == service_name)
            
            if caller_is_root and callee_is_root:
                # Both in root service → internal
                internal_calls.append((current, callee))
            else:
                # At least one is external → external
                external_calls.append((current, callee))
                
                # Further classify external calls
                if caller_is_root and not callee_is_root:
                    # Root calling out to external service
                    outbound_calls.append((current, callee))
                else:
                    # External -> external chain (or external -> root, rare)
                    external_chain_calls.append((current, callee))
            
            # Add to queue if not visited
            if callee not in visited:
                depth_map[callee] = depth + 1
                queue.append((callee, depth + 1))
    
    # Extract all services involved
    all_methods = list(visited)
    services = {m.split("::")[0] for m in all_methods if "::" in m}
    
    result = {
        'root': root_qualified,
        'root_service': service_name,
        'internal_calls': internal_calls,
        'external_calls': external_calls,
        'outbound_calls': outbound_calls,
        'external_chain_calls': external_chain_calls,
        'all_reachable': sorted(all_methods),
        'depth_map': depth_map,
        'call_count': len(internal_calls) + len(external_calls),
        'services': services,
    }
    
    # Optional: detect potential external calls (unresolved callees)
    if include_potential_external and indexes:
        potential_external = []
        potential_external_calls = []
        
        # Get the source service's index
        if service_name in indexes:
            src_index = indexes[service_name]
            
            # Collect all known methods across all services
            all_known_methods = set()
            for idx in indexes.values():
                all_known_methods.update(idx.name_map.keys())
            
            # Check each method in our call graph
            for method_qualified in visited:
                method_svc = method_qualified.split("::")[0]
                method_short = method_qualified.split("::")[-1] if "::" in method_qualified else method_qualified
                
                # Get this method's record to see what it tries to call
                if method_svc in indexes:
                    idx = indexes[method_svc]
                    if method_short in idx.name_map:
                        rec = idx.records[idx.name_map[method_short]]
                        
                        # Check for unresolved callees
                        for callee_raw in rec.callee_names:
                            # Skip if it's already resolved in edges
                            already_resolved = any(
                                c == f"{method_svc}::{callee_raw}" or
                                callee_raw in idx.name_map
                                for c, _ in internal_calls + external_calls
                            )
                            
                            # If not resolved anywhere, it might be external
                            if not already_resolved and callee_raw not in all_known_methods:
                                potential_external.append(callee_raw)
                                potential_external_calls.append((method_qualified, callee_raw))
        
        result['potential_external'] = sorted(set(potential_external))
        result['potential_external_calls'] = potential_external_calls
    
    return result


def find_method_from_log(
    message: str,
    indexes: Dict[str, MethodIndex],
    min_score: float = 0.01,
) -> Dict:
    """
    Find the best matching method across all services for a log message.
    
    Parameters
    ----------
    message    : log message text
    indexes    : dict of {service_name: MethodIndex}
    min_score  : minimum Jaccard score to consider a match (default: 0.01)
    
    Returns
    -------
    dict with:
        'service'       : str - service name (or None)
        'method'        : str - method name (or '<unknown>')
        'qualified'     : str - 'service::method' (or None)
        'score'         : float - match score
        'matched'       : bool - whether a match was found
        'full_name'     : str - full method signature
        'line'          : str - line number
        'all_candidates': list - all matches across services (for debugging)
    
    Example
    -------
    >>> indexes = {
    ...     'frontend': build_method_index(graph_fe, 'frontend'),
    ...     'backend': build_method_index(graph_be, 'backend'),
    ... }
    >>> result = find_method_from_log("User login failed", indexes)
    >>> result['qualified']
    'frontend::handleLogin'
    """
    best_service = None
    best_result = None
    best_score = 0.0
    all_candidates = []
    
    # Try matching against each service
    for service, index in indexes.items():
        result = fast_match(message, index)
        
        if result['matched'] and result['score'] >= min_score:
            all_candidates.append({
                'service': service,
                'method': result['function_name'],
                'score': result['score'],
            })
            
            if result['score'] > best_score:
                best_score = result['score']
                best_result = result
                best_service = service
    
    if best_result is None:
        return {
            'service': None,
            'method': '<unknown>',
            'qualified': None,
            'score': 0.0,
            'matched': False,
            'full_name': '',
            'line': '',
            'all_candidates': all_candidates,
        }
    
    return {
        'service': best_service,
        'method': best_result['function_name'],
        'qualified': f"{best_service}::{best_result['function_name']}",
        'score': best_result['score'],
        'matched': True,
        'full_name': best_result.get('full_name', ''),
        'line': best_result.get('line', ''),
        'all_candidates': all_candidates,
    }