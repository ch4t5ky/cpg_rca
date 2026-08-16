import html
import re
import string
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_METHOD_NAMES: Set[str] = {
    # Java logging frameworks
    "info",
    "warn",
    "warning",
    "debug",
    "error",
    "trace",
    "fatal",
    "log",
    # System output streams (for catching shutdown hooks, startup messages, etc.)
    "println",  # System.out.println(), System.err.println()
    "print",  # System.out.print(), System.err.print()
    # Go logrus / zap / zerolog
    "infof",
    "warnf",
    "debugf",
    "errorf",
    "fatalf",
    "panicf",
    "infoln",
    "warnln",
    "debugln",
    "errorln",
    "printf",
    "msg",
    "msgf",
    # SLF4J / Logback
    "slf4j",
    # Apache Commons Logging
    "commons",
    # Log4j
    "log4j",
    # Custom logger patterns
    "write",  # Writer.write()
    "WriteLine",
    "flush",  # BufferedWriter.flush()
}

AST_LABEL = "AST"
CALL_LABEL = "CALL"
REACHING_DEF_LABEL = "REACHING_DEF"
CDG_LABEL = "CDG"
METHOD_LABEL = "METHOD"
WILDCARD = "<*>"
STRING_RE = re.compile(r'["`\'](.*?)["`\']', re.DOTALL)
LITERAL_LABELS: Set[str] = {
    "LITERAL",
    "STRING",
    "STRING_LITERAL",
    "NUMBER_LITERAL",
    "FIELD_IDENTIFIER",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LogTemplate:
    call_node_id: str
    raw_template: str
    tokens: List[str]
    static_count: int


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    terminals: List[LogTemplate] = field(default_factory=list)


@dataclass
class LogMapping:
    bucket: str
    message: str
    call_node_id: str
    method_name: str
    method_node_id: str
    template: str
    score: int
    matched: bool


# ---------------------------------------------------------------------------
# Basic graph helpers
# ---------------------------------------------------------------------------


def _clean(v) -> str:
    return html.unescape(str(v).strip().strip('"')).strip()


def _label(G: nx.MultiDiGraph, nid) -> str:
    return _clean(G.nodes[nid].get("label", "")).upper()


def _code(G: nx.MultiDiGraph, nid) -> str:
    return _clean(G.nodes[nid].get("CODE", ""))


def _ast_children(G: nx.MultiDiGraph, nid) -> List:
    return [
        v
        for _, v, d in G.edges(nid, data=True)
        if _clean(d.get("label", "")).upper() == AST_LABEL
    ]


def _reaching_def_predecessors(G: nx.MultiDiGraph, nid) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for u, v, d in G.in_edges(nid, data=True):
        if _clean(d.get("label", "")).upper() == REACHING_DEF_LABEL:
            out.append((u, _clean(d.get("VARIABLE", ""))))
    return out


def _normalize_code(s: str) -> str:
    s = str(s)
    s = html.unescape(s)
    s = s.replace(r"\"", '"')
    s = s.replace(r"\'", "'")
    s = s.replace(r"\\", "\\")
    s = re.sub(r"\s+", " ", s)
    return s.strip().strip('"').strip()


def _arg_nodes(G: nx.MultiDiGraph, call_nid) -> List:
    args = []
    for v in _ast_children(G, call_nid):
        lbl = _label(G, v)
        if lbl in {"METHOD", "TYPE_REF", "NAMESPACE_BLOCK", "BLOCK"}:
            continue

        idx = G.nodes[v].get("ARGUMENT_INDEX", 999999)
        try:
            idx = int(str(idx).strip('"'))
        except Exception:
            idx = 999999

        args.append((idx, v))

    args.sort(key=lambda x: x[0])
    return [v for _, v in args]


def _tokenize(s: str) -> List[str]:
    return s.strip().split()


# ---------------------------------------------------------------------------
# FIND PARENT METHOD FOR LOG CALL
# ---------------------------------------------------------------------------


def _find_parent_method(
    G: nx.MultiDiGraph, call_node_id: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the parent METHOD node for a given CALL node.
    Returns (method_name, method_node_id) or (None, None)
    """
    visited = set()
    queue = deque([call_node_id])

    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)

        if _label(G, nid) == METHOD_LABEL:
            method_name = _clean(G.nodes[nid].get("NAME", ""))
            return (method_name, str(nid))

        for parent_id, _, edge_data in G.in_edges(nid, data=True):
            if parent_id not in visited:
                queue.append(parent_id)

    return (None, None)


# ---------------------------------------------------------------------------
# SELECTIVE TEXT CLEANING (KEEPS HYPHENS)
# ---------------------------------------------------------------------------


def _clean_text_selective(text: str) -> str:
    """
    Remove ALL escape sequences and special symbols EXCEPT hyphens (-).
    Keeps hyphens for compound words like "user-failed", "connection-timeout".
    """
    if not text:
        return ""

    text = str(text)

    text = text.replace(r"\\", " ")
    text = text.replace(r"\"", " ")
    text = text.replace(r"\'", " ")
    text = text.replace(r"\/", " ")
    text = re.sub(r"\[a-zA-Z]", " ", text)

    PUNCTUATION_TO_REMOVE = set(string.punctuation)
    PUNCTUATION_TO_REMOVE.discard("-")  # KEEP hyphens

    for p in PUNCTUATION_TO_REMOVE:
        text = text.replace(p, " ")

    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    return text


def _normalize_code_selective(s: str) -> str:
    """Normalize CODE attributes with selective cleaning"""
    s = str(s)
    s = s.strip().strip("'\"")
    s = _clean_text_selective(s)
    return s


# ---------------------------------------------------------------------------
# Backward traversal over REACHING_DEF (OPTIMIZED)
# ---------------------------------------------------------------------------


def _backward_reaching_def(
    G: nx.MultiDiGraph,
    starts: List,
    max_depth: int = 5,
    max_nodes: int = 100,
) -> List[Tuple[str, str, str]]:
    """Optimized traversal with hard limits"""
    visited = set()
    queue = deque((n, 0, "") for n in starts)
    result: List[Tuple[str, str, str]] = []

    while queue and len(visited) < max_nodes:
        nid, depth, variable = queue.popleft()
        if nid in visited or depth > max_depth:
            continue

        visited.add(nid)
        code = _code(G, nid)
        label = _label(G, nid)
        if code:
            result.append((code, label, variable))

        if depth < max_depth and len(visited) < max_nodes:
            for pred, var_name in _reaching_def_predecessors(G, nid):
                if pred not in visited:
                    queue.append((pred, depth + 1, var_name))

    return result


# ---------------------------------------------------------------------------
# FIX: Build template — stops after first LITERAL block (no DDG-chain pollution)
# ---------------------------------------------------------------------------


def _build_template(code_labels: List[Tuple[str, str, str]]) -> str:
    """
    Build template from the FIRST LITERAL block only.

    FIX: stops traversal after finding the log message literal to avoid
    DDG-chain pollution. Without this fix, REACHING_DEF edges pull in tokens
    from sibling/predecessor nodes, turning 'request started <*>' into
    'request started <*> session <*>' — which breaks trie matching.

    Removes leading wildcards, keeps trailing for variable matching.
    """
    if not code_labels:
        return ""

    max_entries = min(len(code_labels), 50)
    code_labels = code_labels[:max_entries]

    parts: List[str] = []
    seen_codes: Set[str] = set()
    need_wildcard = False
    found_literal = False  # FIX: track whether we've seen a literal

    for code, label, variable in code_labels:
        if not code or not code.strip():
            continue

        code_clean = _normalize_code_selective(code)

        if not code_clean:
            continue

        code_key = code_clean.lower()

        if code_key in seen_codes:
            continue
        seen_codes.add(code_key)

        if label in LITERAL_LABELS:
            if need_wildcard and parts and parts[-1] != WILDCARD:
                parts.append(WILDCARD)
                need_wildcard = False

            words = code_clean.split()
            if words:
                parts.extend(words)
            found_literal = True
            continue

        # FIX: stop after first literal — no more DDG pollution
        if found_literal:
            need_wildcard = True
            break

        need_wildcard = True

    while parts and parts[0] == WILDCARD:
        parts.pop(0)

    merged = []
    for p in parts:
        if not (p == WILDCARD and merged and merged[-1] == WILDCARD):
            merged.append(p)

    if merged and merged[-1] != WILDCARD:
        merged.append(WILDCARD)

    return " ".join(merged).strip() if merged else ""


@dataclass
class LogTemplateWithMethod:
    call_node_id: str
    method_name: str
    method_node_id: str
    raw_template: str
    tokens: List[str]
    static_count: int


def build_templates_from_cpg(
    G: nx.MultiDiGraph,
    max_ddg_depth: int = 5,
) -> List[LogTemplateWithMethod]:
    """Build templates and track which method each log belongs to"""
    templates: List[LogTemplateWithMethod] = []

    for nid, attrs in G.nodes(data=True):
        if _label(G, nid) != CALL_LABEL:
            continue

        call_name = _clean(attrs.get("NAME", "")).lower()
        if call_name not in LOG_METHOD_NAMES:
            continue

        method_name, method_node_id = _find_parent_method(G, nid)
        if not method_name:
            method_name = "unknown"
            method_node_id = "unknown"

        args = _arg_nodes(G, nid)
        if not args:
            continue

        code_labels = _backward_reaching_def(G, args, max_depth=max_ddg_depth)
        if not code_labels:
            continue

        raw = _build_template(code_labels)
        if not raw:
            continue

        tokens = _tokenize(raw)
        static_count = sum(1 for t in tokens if t != WILDCARD)

        templates.append(
            LogTemplateWithMethod(
                call_node_id=str(nid),
                method_name=method_name,
                method_node_id=str(method_node_id),
                raw_template=raw,
                tokens=tokens,
                static_count=static_count,
            )
        )

    return templates


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------


def build_trie(templates: List[LogTemplateWithMethod]) -> TrieNode:
    root = TrieNode()

    for tmpl in templates:
        node = root
        for tok in tmpl.tokens:
            key = WILDCARD if tok == WILDCARD else tok.lower()
            if key not in node.children:
                node.children[key] = TrieNode()
            node = node.children[key]
        node.terminals.append(
            LogTemplate(
                call_node_id=tmpl.call_node_id,
                raw_template=tmpl.raw_template,
                tokens=tmpl.tokens,
                static_count=tmpl.static_count,
            )
        )

    return root


def _trie_match(
    node: TrieNode,
    msg: List[str],
    pos: int,
    results: List[LogTemplate],
) -> None:
    """
    FIX: Prefix match support.

    If the message ends but the template has only wildcard edges remaining,
    still count it as a match. This fixes cases where DDG pollution added
    extra tokens to the template (e.g. 'request started <*> session <*>')
    but the actual log message is simply 'request started'.
    """
    if pos == len(msg):
        # Standard terminal match
        results.extend(node.terminals)
        # FIX: prefix match — collect terminals reachable only via wildcards
        queue = []
        if WILDCARD in node.children:
            queue.append(node.children[WILDCARD])
        seen = set()
        while queue:
            n = queue.pop()
            nid = id(n)
            if nid in seen:
                continue
            seen.add(nid)
            results.extend(n.terminals)
            if WILDCARD in n.children:
                queue.append(n.children[WILDCARD])
        return

    tok = msg[pos].lower()

    if tok in node.children:
        _trie_match(node.children[tok], msg, pos + 1, results)

    if WILDCARD in node.children:
        wc = node.children[WILDCARD]
        for end in range(pos + 1, len(msg) + 1):
            _trie_match(wc, msg, end, results)


def map_logs(
    log_rows: List[Tuple[str, str]],
    root: TrieNode,
    templates_with_methods: List[LogTemplateWithMethod],
    min_static: int = 1,
) -> List[LogMapping]:
    """Map logs to templates and include method information"""
    mappings: List[LogMapping] = []

    method_lookup: Dict[str, Tuple[str, str]] = {}
    for tmpl in templates_with_methods:
        method_lookup[tmpl.call_node_id] = (tmpl.method_name, tmpl.method_node_id)

    for bucket, message in log_rows:
        message_clean = _clean_text_selective(message)
        tokens = message_clean.strip().split()

        candidates: List[LogTemplate] = []
        _trie_match(root, tokens, 0, candidates)
        candidates = [c for c in candidates if c.static_count >= min_static]

        if candidates:
            best = max(candidates, key=lambda t: (t.static_count, len(t.tokens)))
            method_name, method_node_id = method_lookup.get(
                best.call_node_id, ("unknown", "unknown")
            )
            mappings.append(
                LogMapping(
                    bucket=bucket,
                    message=message,
                    call_node_id=best.call_node_id,
                    method_name=method_name,
                    method_node_id=method_node_id,
                    template=best.raw_template,
                    score=best.static_count,
                    matched=True,
                )
            )
        else:
            mappings.append(
                LogMapping(
                    bucket=bucket,
                    message=message,
                    call_node_id="",
                    method_name="",
                    method_node_id="",
                    template="",
                    score=0,
                    matched=False,
                )
            )

    return mappings