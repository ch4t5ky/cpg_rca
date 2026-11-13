import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline.entrypoint import EntrypointDetector  # noqa: E402


def cpg(*methods):
    """methods: (name, file, callee name). Every method calls its callee; `helper` is a leaf."""
    g = nx.MultiDiGraph()
    g.add_node("helper", label='"METHOD"', NAME='"helper"', FULL_NAME='"helper"', FILENAME='"util.go"')
    for name, file, callee in methods:
        g.add_node(name, label='"METHOD"', NAME=f'"{name}"', FULL_NAME=f'"{name}"', FILENAME=f'"{file}"')
        g.add_node(name + "-call", label='"CALL"', METHOD_FULL_NAME=f'"{callee}"')
        g.add_edge(name, name + "-call", label='"CONTAINS"')
    return g


def found(skip, *methods):
    return sorted(e.name for e in EntrypointDetector(cpg(*methods), skip_non_request=skip).detect())


def test_default_unchanged():
    assert found(False, ("TestQuote", "quote_test.go", "Quote"), ("Quote", "quote.go", "helper"), ("main", "main.go", "helper")) == ["TestQuote", "main"]


@pytest.mark.parametrize("caller_file", ["quote_test.go", "test_quote.py", "pkg/tests/q.py", "QuoteTest.java", "quote.pb.go", "q_pb2.py", "q_pb2_grpc.py"])
def test_test_and_generated_callers_are_ignored(caller_file):
    # Without the filter Quote has a caller; with it, Quote is the only entrypoint.
    assert found(True, ("Caller", caller_file, "Quote"), ("Quote", "quote.go", "helper")) == ["Quote"]


def test_main_is_hidden_but_still_a_caller():
    assert found(True, ("main", "main.go", "Run"), ("Run", "run.go", "helper")) == []


def test_production_file_named_like_a_test_is_not_ignored():
    assert found(True, ("Caller", "src/Contest.java", "Quote"), ("Quote", "quote.go", "helper")) == ["Caller"]


def test_method_in_init_py_is_not_hidden():
    assert found(True, ("setup", "pkg/__init__.py", "helper")) == ["setup"]
