"""Format placeholders in log literals become <*>, only where the CPG shows formatting.

Read the CASES table first: one row = one log call as written in the service code,
the template that must be built from it, one runtime line, and whether that line
must match the template. The helpers below the table build the small CPG for a row.
Rows that already pass on master guard cases that a careless rewrite would break.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import networkx as nx
import pydot
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.offline import log as L  # noqa: E402

# --------------------------------------------------------------------------- #
# The table. Source snippets are written as in the service code.
# Languages: go, java, java-jul (java.util.logging), python.
# template "" = no template is built. The builder ends every template with a wildcard.
# --------------------------------------------------------------------------- #
CASES = [
    # id,              lang,       source in the service code,                                                                 template built from the literal,                                       runtime line,                                             must match
    # --- Go: printf-style calls (fmt.Errorf, logrus Infof ...) ---
    ("go-errorf",       "go",       'fmt.Errorf("failed to convert price of %q to %s", id, cur)',                              "failed to convert price of <*> to <*>",                           'failed to convert price of "USD" to EUR',               True),
    ("go-errorf-other", "go",       'fmt.Errorf("failed to convert price of %q to %s", id, cur)',                              "failed to convert price of <*> to <*>",                           "connection refused by peer",                             False),
    ("go-glued",        "go",       'fmt.Errorf("took %dms total", ms)',                                                       "took <*> total <*>",                                                  "took 12ms total",                                        True),
    ("go-glued-float",  "go",       'fmt.Errorf("took %dms total", ms)',                                                       "took <*> total <*>",                                                  "took 1.5 ms total",                                      True),
    ("go-glued-short",  "go",       'fmt.Errorf("took %dms total", ms)',                                                       "took <*> total <*>",                                                  "took 12ms",                                              False),
    ("go-percent",      "go",       'logrus.Infof("progress 100%% done")',                                                     "progress 100 done <*>",                                               "progress 100% done",                                     True),
    ("go-backslash",    "go",       r'logrus.Infof("keep\\name done")',                                                        "keep name done <*>",                                                  r"keep\name done",                                        True),
    ("go-raw-backslash", "go",      r"logrus.Infof(`keep\\name done`)",                                                        "keep name done <*>",                                                  r"keep\\name done",                                       True),
    ("go-only-verb",    "go",       'fmt.Errorf("%s", err)',                                                                   "",                                                                    "boom",                                                   False),
    # --- Go: a call that is not printf-style keeps the text of the literal ---
    ("go-info-no-f",    "go",       'logrus.Info("took %dms total")',                                                          "took dms total <*>",                                                  "took 12ms total",                                        False),
    # --- Java ---
    ("java-format",     "java",     'logger.warn(String.format("Port %s is invalid, use default port %d.", a, b))',            "port <*> is invalid use default port <*>",                        "Port abc is invalid, use default port 9555.",            True),
    ("java-format-other", "java",   'logger.warn(String.format("Port %s is invalid, use default port %d.", a, b))',            "port <*> is invalid use default port <*>",                        "Port abc is fine",                                       False),
    ("java-percent-n",  "java",     'logger.warn(String.format("left%nright"))',                                               "left right <*>",                                                      "left\nright",                                            True),
    ("java-braces",     "java",     'logger.info("request {} failed", id)',                                                    "request <*> failed <*>",                                              "request 7 failed",                                       True),
    ("java-braces-other", "java",   'logger.info("request {} failed", id)',                                                    "request <*> failed <*>",                                              "request 7 started",                                      False),
    ("java-braces-no-arg", "java",  'logger.info("request {} failed")',                                                        "request failed <*>",                                                  "request 7 failed",                                       False),
    ("java-numbered",   "java",     'logger.info("request {0} failed", id)',                                                   "request 0 failed <*>",                                                "request 7 failed",                                       False),
    ("java-println",    "java",     'System.out.println("request {} failed")',                                                 "request failed <*>",                                                  "request 7 failed",                                       False),
    ("java-jul",        "java-jul", 'logger.info("request {} failed", id)',                                                    "request failed <*>",                                                  "request 7 failed",                                       False),
    # --- Python ---
    ("py-format",       "python",   'logger.info("A request to send order confirmation email to {} has been received.".format(mail))',
                                                                                                                              "a request to send order confirmation email to <*> has been received <*>",
                                                                                                                                                                                              "A request to send order confirmation email to jack02@example.org has been received.", True),
    ("py-format-numbered", "python", 'logger.info("request {0} failed".format(x))',                                           "request <*> failed <*>",                                              "request 7 failed",                                       True),
    ("py-format-escaped", "python", 'logger.info("literal {{0}} done".format(x))',                                            "literal 0 done <*>",                                                  "literal OTHER done",                                     False),
    ("py-format-escaped-and-real", "python", 'logger.info("set {{{}}} done".format(x))',                                      "set <*> done <*>",                                                    "set 7 done",                                             True),
    ("py-no-format",    "python",   'logger.info("literal {0} done")',                                                         "literal 0 done <*>",                                                  "literal OTHER done",                                     False),
    ("py-percent",      "python",   'logger.info("value %ld done" % x)',                                                       "value ld done <*>",                                                   "value 5 done",                                           False),
]


@pytest.mark.parametrize("case_id, lang, source, template, line, matches", CASES, ids=[c[0] for c in CASES])
def test_template_and_match(case_id, lang, source, template, line, matches):
    graph = cpg_from_source(lang, source)
    templates = L.build_templates_from_cpg(graph)
    assert [t.raw_template for t in templates] == ([template] if template else [])
    mapped = L.map_logs([("bucket", line)], L.build_trie(templates), templates)
    assert mapped[0].matched is matches


# --------------------------------------------------------------------------- #
# Real CPG excerpts: the helper below imitates these shapes, so check them once.
# Each is cut down to the nodes that the template builder reads.
# --------------------------------------------------------------------------- #

EMAILSERVICE_FORMAT = """
digraph {
  "m" [label="METHOD" NAME="SendOrderConfirmation" FILENAME="email_server.py"];
  "info" [label="CALL" NAME="info" METHOD_FULL_NAME="logging.py:<module>.getLogger.<returnValue>.info"];
  "logger" [label="IDENTIFIER" NAME="logger" CODE="logger" ARGUMENT_INDEX="0"];
  "blk" [label="BLOCK" CODE="tmp5 = ..." ARGUMENT_INDEX="1"];
  "asg" [label="CALL" NAME="<operator>.assignment" CODE="tmp5 = 'A request to send order confirmation email to {} has been received.'"];
  "fmt" [label="CALL" NAME="format" CODE="'A request to send order confirmation email to {} has been received.'.format(request.email)"];
  "lit" [label="LITERAL" CODE="'A request to send order confirmation email to {} has been received.'" ARGUMENT_INDEX="2"];
  "m" -> "info" [label="AST"];
  "info" -> "logger" [label="AST"];
  "info" -> "blk" [label="AST"];
  "blk" -> "asg" [label="AST"];
  "blk" -> "fmt" [label="AST"];
  "asg" -> "lit" [label="AST"];
  "lit" -> "asg" [label="REACHING_DEF"];
  "lit" -> "logger" [label="REACHING_DEF"];
}
"""

CHECKOUTSERVICE_ERRORF = """
digraph {
  "m" [label="METHOD" NAME="main.main.go" FILENAME="main.go"];
  "call" [label="CALL" NAME="Errorf" METHOD_FULL_NAME="fmt.Errorf" CODE="fmt.Errorf(\\"failed to convert price of %q to %s\\", item.GetProductId(), userCurrency)"];
  "lit" [label="LITERAL" CODE="\\"failed to convert price of %q to %s\\"" ARGUMENT_INDEX="1"];
  "m" -> "call" [label="AST"];
  "call" -> "lit" [label="AST"];
  "lit" -> "call" [label="REACHING_DEF"];
}
"""


def test_real_excerpt_emailservice_python_format():
    """emailservice email_server.py:110 (export.dot L5169): Joern lowers 'text {}'.format(x), so the literal is not under info()."""
    check_excerpt(
        EMAILSERVICE_FORMAT,
        "a request to send order confirmation email to <*> has been received <*>",
        "A request to send order confirmation email to jack02@example.org has been received.",
    )


def test_real_excerpt_checkoutservice_go_errorf():
    """checkoutservice main.go:350 (export.dot L14055): fmt.Errorf literal with escaped quotes in CODE, as exported."""
    check_excerpt(CHECKOUTSERVICE_ERRORF, "failed to convert price of <*> to <*>", 'failed to convert price of "USD" to EUR')


def check_excerpt(dot, template, line):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pydot calls deprecated pyparsing names
        parsed = pydot.graph_from_dot_data(dot)[0]
    graph = nx.MultiDiGraph()
    for node in parsed.get_nodes():
        graph.add_node(node.get_name().strip('"'), **node.get_attributes())
    for edge in parsed.get_edges():
        src, dst = edge.get_source().strip('"'), edge.get_destination().strip('"')
        graph.add_edge(src, dst, **edge.get_attributes())
    templates = L.build_templates_from_cpg(graph)
    assert [t.raw_template for t in templates] == [template]
    assert L.map_logs([("bucket", line)], L.build_trie(templates), templates)[0].matched


def test_runtime_line_cleaning_is_unchanged():
    assert L._clean_text_selective("Port 8080 is 100% invalid {}") == "port 8080 is 100 invalid"


# --------------------------------------------------------------------------- #
# Helpers: build the small CPG for one table row. Not needed to read the table.
# --------------------------------------------------------------------------- #

CALL_RE = re.compile(r"([\w.]+)\.(\w+)\((.*)\)")  # receiver.name(args)
LITERAL_RE = re.compile(r'"(?:[^"\\]|\\.)*"|`[^`]*`')
FILES = {"go": "main.go", "java": "A.java", "java-jul": "A.java", "python": "app.py"}
PY_LOGGER = "logging.py:<module>.getLogger.<returnValue>.info"


def cpg_from_source(lang, source):
    """Tiny CPG of one log call: METHOD -> call(literal), or call(wrapper(literal))."""
    receiver, name, args = CALL_RE.fullmatch(source).groups()
    wrapper = None  # the call that holds the literal when it is not the log call itself
    if lang.startswith("java"):
        found = re.fullmatch(r"String\.format\((.*)\)", args)  # String.format(literal, ...)
        if found:
            wrapper, args = "format", found.group(1)
    elif lang == "python":
        found = re.fullmatch(r"(.*)\.format\(.*\)", args)  # "literal".format(...)
        if found:
            wrapper, args = "format", found.group(1)
        found = re.fullmatch(r"(.*) % .*", args)  # "literal" % x
        if found:
            wrapper, args = "<operator>.modulo", found.group(1)
    literal = LITERAL_RE.search(args)
    # The literal is argument number (commas before it + 1); the rest are arguments after it.
    index = args[: literal.start()].count(",") + 1
    extra = 0 if wrapper else len([a for a in args[literal.end():].split(",") if a.strip()])

    full_name = {
        "go": f"{receiver}.{name}",
        "java": f"org.slf4j.Logger.{name}",
        "java-jul": f"java.util.logging.Logger.{name}",
        "python": PY_LOGGER,
    }[lang]
    # export.dot doubles every backslash of the source literal.
    code = literal.group(0).replace("\\", "\\\\")

    g = nx.MultiDiGraph()
    g.add_node("m", label="METHOD", NAME="run", FILENAME=FILES[lang])

    def add(nid, parent, label, **attrs):
        g.add_node(nid, label=label, **attrs)
        g.add_edge(parent, nid, label="AST")

    add("log", "m", "CALL", NAME=name, METHOD_FULL_NAME=full_name, CODE=source)
    if lang == "python" and wrapper == "format":
        # Joern lowers 'text {}'.format(x) to: tmp = 'text {}' ; tmp.format(x), both in one BLOCK.
        add("logger", "log", "IDENTIFIER", NAME="logger", CODE="logger", ARGUMENT_INDEX="0")
        add("blk", "log", "BLOCK", CODE="tmp = ...", ARGUMENT_INDEX="1")
        add("asg", "blk", "CALL", NAME="<operator>.assignment", CODE=f"tmp = {code}")
        add("lit", "asg", "LITERAL", CODE=code, ARGUMENT_INDEX="2")
        add("fmt", "blk", "CALL", NAME="format", CODE=f"{code}.format(x)")
        g.add_edge("lit", "asg", label="REACHING_DEF")
        g.add_edge("lit", "logger", label="REACHING_DEF")
        return g

    parent = "log"
    if wrapper:
        add("fmt", "log", "CALL", NAME=wrapper, CODE=f"{wrapper}(...)", ARGUMENT_INDEX="1")
        parent = "fmt"
    add("lit", parent, "LITERAL", CODE=code, ARGUMENT_INDEX=str(index))
    g.add_edge("lit", parent, label="REACHING_DEF")
    for i in range(extra):
        add(f"arg{i}", "log", "IDENTIFIER", NAME="x", CODE="x", ARGUMENT_INDEX=str(index + 1 + i))
    return g
