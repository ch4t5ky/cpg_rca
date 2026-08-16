"""
trie_matcher.py
===============

Online matcher for builder.py Log Trie artifacts.

The matcher loads output/<service>/trie.json and resolves a runtime message to
one or more static logger CPG CALL candidates. The result is intentionally a
set of candidates: identical text may be emitted by multiple source locations,
and the FSM hypothesis store is responsible for retaining only structurally
valid continuations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WILDCARD = "<*>"


@dataclass(frozen=True)
class TrieMatch:
    """One static logger-call candidate returned by a service Log Trie."""

    call_node_id: str
    template: str
    tokens: tuple[str, ...]
    static_score: int
    score: float


def normalize_text(value: str) -> str:
    """Normalize runtime text using the same token policy as log.py templates."""
    value = (value or "").lower()
    value = re.sub(r"[^\w*]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _template_tokens(template: str) -> tuple[str, ...]:
    return tuple(normalize_text(template.replace(WILDCARD, f" {WILDCARD} ")).split())


def _matches_template(message_tokens: tuple[str, ...], template_tokens: tuple[str, ...]) -> bool:
    """
    Match a message against a template with variable-length <*> wildcards.

    Dynamic programming is used because wildcard nodes can consume zero or more
    message tokens. This preserves exact Trie semantics without assuming a
    unique greedy expansion.
    """
    memo: dict[tuple[int, int], bool] = {}

    def visit(message_index: int, template_index: int) -> bool:
        key = (message_index, template_index)
        if key in memo:
            return memo[key]

        if template_index == len(template_tokens):
            memo[key] = message_index == len(message_tokens)
            return memo[key]

        token = template_tokens[template_index]
        if token == WILDCARD:
            # Wildcard can consume zero tokens or extend over one token.
            result = visit(message_index, template_index + 1) or (
                message_index < len(message_tokens) and visit(message_index + 1, template_index)
            )
            memo[key] = result
            return result

        result = (
            message_index < len(message_tokens)
            and message_tokens[message_index] == token
            and visit(message_index + 1, template_index + 1)
        )
        memo[key] = result
        return result

    return visit(0, 0)


class ServiceTrieMatcher:
    """
    Match runtime messages against one service's serialized Log Trie.

    The current builder.py trie.json contains both a recursive `trie` object and
    a flat `templates` catalog. The flat catalog is used for deterministic
    matching and returns every matching CPG CALL candidate. The serialized Trie
    remains the canonical offline index and visualization artifact.
    """

    def __init__(self, trie_path: Path) -> None:
        payload = json.loads(trie_path.read_text(encoding="utf-8"))
        self.trie_path = trie_path
        self.templates = [
            {
                "call_node_id": str(raw["call_node_id"]),
                "raw_template": str(raw["raw_template"]),
                "tokens": tuple(raw.get("tokens") or _template_tokens(str(raw["raw_template"]))),
                "static_count": int(raw.get("static_count", 0)),
            }
            for raw in payload.get("templates", [])
        ]
        self._first_static_token_index: dict[str, list[dict[str, Any]]] = {}
        self._wildcard_first: list[dict[str, Any]] = []
        for template in self.templates:
            first_static = next((token for token in template["tokens"] if token != WILDCARD), None)
            if first_static:
                self._first_static_token_index.setdefault(first_static, []).append(template)
            else:
                self._wildcard_first.append(template)

    def match(self, message: str) -> list[TrieMatch]:
        """Return all static CPG CALL candidates compatible with one runtime message."""
        message_tokens = tuple(normalize_text(message).split())
        if not message_tokens:
            return []

        # A static token can occur after a leading wildcard, so consider every
        # message token as an index key. Deduplication keeps candidate scanning bounded.
        candidates: dict[str, dict[str, Any]] = {}
        for token in message_tokens:
            for template in self._first_static_token_index.get(token, []):
                candidates[template["call_node_id"]] = template
        for template in self._wildcard_first:
            candidates[template["call_node_id"]] = template

        matches: list[TrieMatch] = []
        for template in candidates.values():
            template_tokens = tuple(
                WILDCARD if token == WILDCARD else normalize_text(token)
                for token in template["tokens"]
            )
            if not _matches_template(message_tokens, template_tokens):
                continue
            static_count = template["static_count"]
            specificity = static_count / max(static_count, len(message_tokens))
            matches.append(TrieMatch(
                call_node_id=template["call_node_id"],
                template=template["raw_template"],
                tokens=template_tokens,
                static_score=static_count,
                score=0.90 + 0.10 * specificity,
            ))

        # Multiple templates can map to the same CPG call id only in malformed
        # catalogs; retain the strongest one deterministically.
        best: dict[str, TrieMatch] = {}
        for match in matches:
            previous = best.get(match.call_node_id)
            if previous is None or (match.static_score, match.score) > (previous.static_score, previous.score):
                best[match.call_node_id] = match
        return sorted(best.values(), key=lambda item: (item.static_score, item.score), reverse=True)

    def match_call_node_ids(self, message: str) -> set[str]:
        """Convenience API for FSM matcher integration."""
        return {match.call_node_id for match in self.match(message)}
