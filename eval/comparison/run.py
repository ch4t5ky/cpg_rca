"""Unified acceptance-quality experiment: cpg_rca vs PRINS vs LogSynthFSM.

Test service : services/retryservice (synthetic RetryService with a hidden
               call-site oracle and a fixed train/test split).
Dataset      : eval/comparison/data/fixture_data (logs.jsonl,
               executions.jsonl, oracle_events.jsonl). Regenerate with
               ``python services/retryservice/retry_service.py --repetitions 20``.
Competitors  : eval/comparison/baselines/ (vendored PRINS replication package
               and LogSynthFSM package).

Protocol
--------
* Identical train/test executions for every method; train data is used only
  to train PRINS. cpg_rca uses only the source code CPG; LogSynthFSM would
  use train logs through its parsing + State Discovery stage.
* Common observable alphabet: every message is canonicalized with
  ``normalize_message`` (numbers/hex/uuids -> ``<*>``), the same function for
  all methods.
* Full-acceptance semantics for everyone: a trace is accepted iff the model
  consumes *all* events starting from an initial state and ends in an
  accepting/terminal state. No message may be skipped.
* Valid groups: ``known`` = scenarios present in train (immediate_success,
  one_retry_success); ``heldout`` = test-only scenarios (two_retries_success,
  retry_exhausted). Invalid traces come from
  eval/comparison/invalid_traces.json and are labelled from the source-code
  semantics of retry_service.py — never derived from any evaluated FSM.
  ``retry budget exhausted`` (ERROR) is a *valid* terminal scenario.
* LogSynthFSM's State Discovery stage requires an LLM provider
  (openai/ollama). When neither is available the method is reported as
  ``blocked`` — no substitute implementation is used. When a provider is
  available, the experiment runs the method's own regex parsing stage and
  its LLMStateDiscoveryAgent on the train logs and evaluates the resulting
  FSM with the same full-acceptance semantics (start states = states without
  incoming transitions, terminal states = states without outgoing ones).

Outputs (into --out, default eval/comparison/reports/full):
    per_execution.csv, summary.csv, config.json
    cpg_rca/      trie.json, fsm.json (built model)
    PRINS/        prins_retryservice.pkl(+meta), vocab.json, prins_work/
    LogSynthFSM/  logsynthfsm_work/ (when an LLM provider is available)

Usage:
    python eval/comparison/run.py                # full-pipeline comparison run
    python eval/comparison/run.py --retrain-prins
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # eval/comparison
REPO_ROOT = PACKAGE_DIR.parent.parent                   # repo root
BASELINES_DIR = PACKAGE_DIR / "baselines"
PRINS_PKG_DIR = BASELINES_DIR / "PRINS" / "PRINS"
LOGSYNTH_DIR = BASELINES_DIR / "LogSynthFSM-KDD-2026"

for extra in (str(PRINS_PKG_DIR), str(REPO_ROOT), str(LOGSYNTH_DIR), str(BASELINES_DIR)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from eval.comparison.metrics import NA, summarize  # noqa: E402
from prins_data import normalize_message  # noqa: E402


# ── canonicalization ───────────────────────────────────────────────────────


def canon(message: str) -> str:
    """One shared observable-event alphabet for all methods."""
    return normalize_message(message)


# ── data loading ───────────────────────────────────────────────────────────


def load_fixture(fixture: Path):
    executions = {}
    for line in (fixture / "executions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        executions[row["execution_id"]] = row

    raw_traces: dict[str, list[dict]] = defaultdict(list)
    for line in (fixture / "logs.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        raw_traces[row["execution_id"]].append(row)
    traces = {k: sorted(v, key=lambda e: e["event_index"]) for k, v in raw_traces.items()}

    oracle: dict[tuple[str, int], dict] = {}
    oracle_path = fixture / "oracle_events.jsonl"
    if oracle_path.is_file():
        for line in oracle_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            oracle[(row["execution_id"], row["event_index"])] = row

    invalid_path = Path(__file__).with_name("invalid_traces.json")
    invalid_specs = json.loads(invalid_path.read_text(encoding="utf-8"))

    train_scenarios = {
        r["scenario"] for r in executions.values() if r["split"] == "train"
    }
    return executions, traces, oracle, invalid_specs, train_scenarios


# ── cpg_rca model ──────────────────────────────────────────────────────────


class CpgRcaModel:
    """Acceptance wrapper over the cpg_rca StaticLogFSM JSON artifact."""

    WILDCARD = "<*>"

    def __init__(self, fsm_payload: dict, event_normalizer=None):
        self.entrypoint = fsm_payload["entrypoint_full_name"]
        self.start_states = set(fsm_payload["start_states"])
        self.terminal_states = set(fsm_payload["terminal_states"])
        self.states = {s["id"]: s for s in fsm_payload["states"]}
        self.transitions = fsm_payload["transitions"]
        self.templates = sorted({t["template"] for t in self.transitions})
        self.event_normalizer = event_normalizer
        # template -> physical CPG call node ids (for log-to-code matching)
        self.template_call_ids: dict[str, set[str]] = defaultdict(set)
        for t in self.transitions:
            self.template_call_ids[t["template"]].add(str(t["log_call_node_id"]))

    def norm(self, event: str) -> str:
        """Normalize an observed event exactly like the online stage does.

        cpg_rca extracts log templates from CPG literals with punctuation
        removed and lower-cased (``_clean_text_selective``), and its runtime
        matcher normalizes incoming messages with the same token policy
        (``src/online/trie_matcher.normalize_text``) before matching. The
        offline trie building is untouched — this is the online/matching
        layer of the method, applied at evaluation time.
        """
        if self.event_normalizer is None:
            return event
        return self.event_normalizer(event)

    def matches(self, template: str, event: str) -> bool:
        t_tokens = template.split()
        e_tokens = event.split()
        i = j = 0
        while i < len(t_tokens):
            if t_tokens[i] == self.WILDCARD:
                i += 1
                if i == len(t_tokens):
                    return True  # trailing wildcard absorbs the rest
                if j >= len(e_tokens):
                    return False
                j += 1
            else:
                if j >= len(e_tokens) or t_tokens[i] != e_tokens[j]:
                    return False
                i += 1
                j += 1
        return True

    def alphabet_covers(self, event: str) -> bool:
        event = self.norm(event)
        return any(self.matches(t, event) for t in self.templates)

    def check(self, events: list[str]) -> tuple[bool, int, str]:
        """Full acceptance: initial state -> all events -> terminal state."""
        current = set(self.start_states)
        for index, event in enumerate(events, start=1):
            event = self.norm(event)
            matched = [
                t for t in self.transitions
                if t["source_segment_id"] in current and self.matches(t["template"], event)
            ]
            if not matched:
                if not self.alphabet_covers(event):
                    return False, index - 1, "unknown_event"
                return False, index - 1, "no_matching_transition"
            current = {t["target_segment_id"] for t in matched}
        if current & self.terminal_states:
            return True, len(events), ""
        return False, len(events), "not_terminal_state"


def resolve_entrypoint_id(graph, method_name: str = "execute") -> str:
    """Find the CPG METHOD node id for the service's main entrypoint.

    Joern ids are opaque and change on every CPG re-export, so the node is
    resolved by name: the METHOD node whose NAME is ``method_name`` and whose
    FULL_NAME contains ``RetryService``. When several overlays duplicate the
    method (e.g. ``execute<metaClassAdapter>``), the metaClassAdapter variant
    is preferred to match the offline builder's behaviour.
    """
    from src.offline.flow import _clean

    candidates = []
    for node_id, data in graph.nodes(data=True):
        if _clean(data.get("label", "")).upper() != "METHOD":
            continue
        name = _clean(data.get("NAME", ""))
        full_name = _clean(data.get("FULL_NAME", ""))
        if name.split("<")[0] == method_name and "RetryService" in full_name:
            candidates.append((node_id, name))
    if not candidates:
        raise ValueError(f"no METHOD node named {method_name!r} for RetryService in CPG")
    candidates.sort(key=lambda item: "<metaClassAdapter>" not in item[1])
    return str(candidates[0][0])


def build_cpg_rca(dot_path: Path, out_dir: Path, clean_events: bool = True):
    """Build (and time) the cpg_rca model from the service CPG."""
    import builder
    from src.offline.finite_state_machine import LogFlowExtractor
    from src.offline.flow import EntrypointFlow

    event_normalizer = None
    if clean_events:
        # the same normalization the online stage applies to runtime
        # messages before template matching (src/online/trie_matcher.py)
        from src.online.trie_matcher import normalize_text as event_normalizer

    started = time.perf_counter()
    graph = builder.load_cpg(dot_path)
    templates, _root = builder.build_log_trie(graph, out_dir)
    entrypoint_id = resolve_entrypoint_id(graph)
    flow = EntrypointFlow(graph, max_depth=5, max_paths=50).build(entrypoint_id)
    fsm = LogFlowExtractor(templates).extract(flow)
    payload = builder.serialize_fsm(fsm)
    build_s = time.perf_counter() - started
    return CpgRcaModel(payload, event_normalizer), payload, build_s, graph


def cpg_call_line_map(graph) -> dict[str, str]:
    """CPG CALL node id -> LINE_NUMBER (for log-to-code accuracy)."""
    from src.offline.flow import _clean

    lines = {}
    for node_id, data in graph.nodes(data=True):
        if _clean(data.get("label", "")).upper() == "CALL":
            lines[str(node_id)] = _clean(data.get("LINE_NUMBER", ""))
    return lines


# ── PRINS model ────────────────────────────────────────────────────────────


class PrinsModel:
    def __init__(self, dfa, vocab: dict[str, str] | None = None):
        self.dfa = dfa
        self.vocab = vocab or {}
        # dfa.alphabet entries are (tid, guard) pairs in this PRINS build
        self.alphabet = {a[0] if isinstance(a, tuple) else a for a in dfa.alphabet}

    def event_id(self, event: str) -> str | None:
        return self.vocab.get(event)

    def alphabet_covers(self, event: str) -> bool:
        return self.event_id(event) in self.alphabet

    def check(self, events: list[str]) -> tuple[bool, int, str]:
        dfa = self.dfa
        current = dfa.initial_state
        for index, event in enumerate(events, start=1):
            tid = self.event_id(event)
            if tid not in self.alphabet:
                return False, index - 1, "unknown_event"
            nxt = dfa.make_guarded_transition(current, {"tid": tid, "values": []})
            if nxt is None:
                nxt = dfa.make_guarded_transition(
                    current, {"tid": tid, "values": []}, ignore_guard=True
                )
            if nxt is None:
                return False, index - 1, "no_matching_transition"
            current = nxt
        if current in dfa.accepting_states:
            return True, len(events), ""
        return False, len(events), "not_terminal_state"


def build_prins(traces: dict[str, list[dict]], train_ids: list[str], work_dir: Path,
                k: int, u: int, mint_timeout: int, prins_pkl: Path, repo_root: Path):
    """Train PRINS in an isolated subprocess and load the resulting DFA.

    A separate interpreter avoids the `src` package-name collision between
    this repo (src/offline, src/online) and the vendored PRINS package, which
    otherwise makes the PRINS import order-dependent.
    """
    script = Path(__file__).with_name("prins_train.py")
    fixture = repo_root / "eval" / "comparison" / "data" / "fixture_data"
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script), str(fixture), str(work_dir), str(prins_pkl),
         str(k), str(u), str(mint_timeout)],
        capture_output=True, text=True, timeout=mint_timeout * 4,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"prins_train failed: {proc.stderr[-2000:] or proc.stdout[-2000:]}")
    with open(prins_pkl, "rb") as fh:
        cached = pickle.load(fh)
    build_s = json.loads(Path(str(prins_pkl) + ".meta.json").read_text(encoding="utf-8"))["build_time_s"]
    _ = started  # build_s is measured inside the subprocess (same machine)
    return PrinsModel(cached["dfa"], cached["vocab"]), build_s


# ── LogSynthFSM model (FSM stage: parsing + State Discovery) ───────────────


def logsynthfsm_provider() -> str:
    """Return the LLM provider name when available, else ''."""
    try:
        import urllib.request

        from logsynthfsm.llm.local_models import OllamaClient
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"logsynthfsm package not importable: {type(exc).__name__}: {exc}"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    base = getattr(OllamaClient, "base_url", None) or "http://localhost:11434"
    try:
        urllib.request.urlopen(base + "/api/tags", timeout=3)
        return "ollama"
    except Exception:
        return (
            "LogSynthFSM State Discovery is LLM-backed (openai/ollama); no "
            "OPENAI_API_KEY is set and no ollama server responds on "
            "localhost:11434. The FSM stage cannot run without an LLM "
            "provider and no substitute implementation is allowed."
        )


class LogSynthFsmModel:
    """Full-acceptance wrapper over the LogSynthFSM State Discovery FSM.

    LogSynthFSM states carry no explicit initial/accepting flags, so the
    standard full-acceptance semantics are derived from its own transition
    graph: start states have no incoming transitions, terminal states have
    no outgoing transitions (every state is both when the FSM has no
    transitions).
    """

    def __init__(self, fsm):
        from logsynthfsm.parsing.parameter_extractor import templatize
        from logsynthfsm.parsing.regex_parser import stable_template_id

        self.fsm = fsm
        self._templatize = templatize
        self._template_id = stable_template_id
        self.template_to_state = {m.template_id: m.state_id for m in fsm.template_state_map}
        self.transitions = {(t.src_state, t.dst_state) for t in fsm.transitions}
        state_ids = fsm.state_ids()
        if self.transitions:
            with_incoming = {dst for _, dst in self.transitions}
            with_outgoing = {src for src, _ in self.transitions}
            self.start_states = state_ids - with_incoming
            self.terminal_states = state_ids - with_outgoing
        else:
            self.start_states = set(state_ids)
            self.terminal_states = set(state_ids)

    def event_template_id(self, event: str) -> str:
        template, _ = self._templatize(event)
        return self._template_id(template)

    def alphabet_covers(self, event: str) -> bool:
        return self.event_template_id(event) in self.template_to_state

    def check(self, events: list[str]) -> tuple[bool, int, str]:
        prev_state = None
        for index, event in enumerate(events, start=1):
            tid = self.event_template_id(event)
            if tid not in self.template_to_state:
                return False, index - 1, "unknown_event"
            dst = self.template_to_state[tid]
            if prev_state is None:
                if dst not in self.start_states:
                    return False, index - 1, "no_matching_transition"
            elif (prev_state, dst) not in self.transitions:
                return False, index - 1, "no_matching_transition"
            prev_state = dst
        if prev_state is not None and prev_state in self.terminal_states:
            return True, len(events), ""
        return False, len(events), "not_terminal_state"


def build_logsynthfsm(traces: dict[str, list[dict]], train_ids: list[str],
                      work_dir: Path, provider: str):
    """Run the LogSynthFSM FSM stage (regex parsing + LLM State Discovery)."""
    from logsynthfsm.fsm.state_discovery import LLMStateDiscoveryAgent, write_fsm_csvs
    from logsynthfsm.llm.factory import build_llm_client
    from logsynthfsm.parsing.regex_parser import RegexLogParser, write_events_csv

    raw_dir = work_dir / "train_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parser = RegexLogParser()
    events = []
    for exec_id in train_ids:
        path = raw_dir / f"{exec_id}.log"
        with open(path, "w", encoding="utf-8") as fh:
            for e in sorted(traces[exec_id], key=lambda x: x["event_index"]):
                fh.write(canon(e["message"]) + "\n")
        events.extend(parser.parse_file(path, source_dataset=exec_id))

    events_path = work_dir / "retryservice_parsed_events.csv"
    write_events_csv(events, events_path)

    client = build_llm_client(provider)
    agent = LLMStateDiscoveryAgent(client)
    started = time.perf_counter()
    fsm = agent.discover_from_csv(events_path)
    build_s = time.perf_counter() - started
    write_fsm_csvs(fsm, work_dir / "fsm", "retryservice")
    return LogSynthFsmModel(fsm), fsm, build_s


# ── evaluation ─────────────────────────────────────────────────────────────


def evaluate_model(method: str, model, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        started = time.perf_counter()
        accepted, n_processed, reason = model.check(case["events"])
        check_s = time.perf_counter() - started
        # Negative traces containing an event outside *this* model's alphabet
        # are rejected trivially (unknown event) — that says nothing about the
        # model's understanding of structure. Split them from the cases whose
        # events are all known, where rejection is a real selectivity signal.
        if case["ground_truth"] == "invalid":
            invalid_group = ("unknown_event"
                             if any(not model.alphabet_covers(e) for e in case["events"])
                             else "known_alphabet")
        else:
            invalid_group = ""
        rows.append({
            "method": method,
            "case_id": case["case_id"],
            "group": case.get("group", ""),
            "split": case.get("split", ""),
            "scenario": case.get("scenario", ""),
            "ground_truth": case["ground_truth"],
            "decision": "accepted" if accepted else "rejected",
            "invalid_group": invalid_group,
            "n_events": n_processed,
            "trace_events": len(case["events"]),
            "reject_reason": reason,
            "check_time_s": round(check_s, 6),
        })
    return rows


def event_coverage(model, cases: list[dict]) -> float:
    distinct = {e for c in cases for e in c["events"]}
    if not distinct:
        return 1.0
    covered = sum(1 for e in distinct if model.alphabet_covers(e))
    return round(covered / len(distinct), 6)


# ── main ───────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", type=Path,
                        default=Path(__file__).parent / "data" / "fixture_data")
    parser.add_argument("--dot", type=Path,
                        default=REPO_ROOT / "services" / "retryservice" / "export.dot")
    parser.add_argument("--out", type=Path, default=None,
                        help="report directory (default: eval/comparison/reports/full)")
    parser.add_argument("--retrain-prins", action="store_true",
                        help="re-run PRINS training even if a cached model exists")
    parser.add_argument("--prins-k", type=int, default=2)
    parser.add_argument("--prins-u", type=int, default=1)
    parser.add_argument("--mint-timeout", type=int, default=300)
    args = parser.parse_args(argv)

    args.variant = "full"
    if args.out is None:
        args.out = PACKAGE_DIR / "reports" / args.variant
    args.out = args.out.resolve()

    args.out.mkdir(parents=True, exist_ok=True)
    executions, traces, oracle, invalid_specs, train_scenarios = load_fixture(args.fixture)

    # ── assemble evaluation cases (valid: all executions of both splits) ──
    cases = []
    for exec_id, meta in sorted(executions.items()):
        group = "known" if meta["scenario"] in train_scenarios else "heldout"
        cases.append({
            "case_id": exec_id,
            "group": group,
            "split": meta["split"],
            "scenario": meta["scenario"],
            "ground_truth": "valid" if meta["expected_valid"] else "invalid",
            "events": [canon(e["message"]) for e in traces[exec_id]],
        })
    for spec in invalid_specs:
        cases.append({
            "case_id": spec["case_id"],
            "group": "invalid",
            "split": "invalid",
            "scenario": "invalid",
            "ground_truth": "invalid",
            "events": [canon(m) for m in spec["events"]],
            "invalid_reason": spec["reason"],
        })

    train_ids = sorted(k for k, m in executions.items() if m["split"] == "train")
    per_execution_rows: list[dict] = []
    summary_rows: list[dict] = []
    config: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "python": sys.version,
        "platform": platform.platform(),
        "java": subprocess.run(["java", "-version"], capture_output=True, text=True)
        .stderr.splitlines()[0],
        "fixture": str(args.fixture),
        "service_cpg": str(args.dot),
        "split": {
            "train": sorted({executions[k]["scenario"] for k in train_ids}),
            "test": sorted({executions[k]["scenario"] for k in executions if executions[k]["split"] == "test"}),
            "group_rule": "known = scenarios present in train; heldout = test-only scenarios",
            "generator": "services/retryservice/retry_service.py --repetitions 20 (deterministic)",
        },
        "seeds": {"prins_max_traces_sample": 42, "comment": "no random sampling in this run"},
        "alphabet": "normalize_message (numbers/hex/uuids -> <*>) for every method",
        "acceptance": "full acceptance: initial state, every event consumed, terminal state; no skipping",
        "invalid_traces": {
            "source": "eval/comparison/invalid_traces.json",
            "labelling": "independent: derived from retry_service.py source semantics, not from any evaluated FSM",
            "n": len(invalid_specs),
        },
        "methods": {},
    }

    # ── cpg_rca ──────────────────────────────────────────────────────────
    cpg_cfg = {"flow_max_depth": 5, "flow_max_paths": 50,
               "entrypoint": "RetryService.execute (resolved by name; Joern ids are not stable)",
               "event_cleaning":
               "events normalized with src/online/trie_matcher.normalize_text "
               "(the online-stage runtime policy; offline artifacts unchanged)"}
    config["methods"]["cpg_rca"] = {"type": "code-based (CPG -> semantic flow -> FSM)", **cpg_cfg}
    cpg_dir = args.out / "cpg_rca"
    cpg_dir.mkdir(parents=True, exist_ok=True)
    cpg_cfg["artifacts_dir"] = str(cpg_dir)
    try:
        cpg_model, cpg_payload, cpg_build_s, cpg_graph = build_cpg_rca(
            args.dot, cpg_dir, clean_events=True)
        (cpg_dir / "fsm.json").write_text(
            json.dumps(cpg_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        cpg_cfg["entrypoint_resolved"] = cpg_payload["entrypoint_full_name"]
        cpg_cfg["entrypoint_node_id"] = resolve_entrypoint_id(cpg_graph)
        rows = evaluate_model("cpg_rca", cpg_model, cases)
        per_execution_rows.extend(rows)
        coverage = event_coverage(cpg_model, cases)
        summary_rows.extend(summarize(rows, "cpg_rca", build_time_s=round(cpg_build_s, 3),
                                      coverage=coverage))
        config["methods"]["cpg_rca"]["build_time_s"] = round(cpg_build_s, 3)
        config["methods"]["cpg_rca"]["event_coverage"] = coverage

        # log-to-code matching accuracy against the hidden call-site oracle
        line_map = cpg_call_line_map(cpg_graph)
        total = matched = correct = 0
        for (exec_id, event_index), orc in sorted(oracle.items()):
            message = orc["event"]
            call_ids = set()
            for tmpl, ids in cpg_model.template_call_ids.items():
                if cpg_model.matches(tmpl, cpg_model.norm(canon(message))):
                    call_ids |= ids
            if not call_ids:
                continue
            matched += 1
            total += 1
            if str(orc["line"]) in {line_map.get(cid, "") for cid in call_ids}:
                correct += 1
        log2code = {
            "events_with_oracle": len(oracle),
            "matched_to_call_site": matched,
            "line_correct": correct,
            "accuracy": round(correct / matched, 6) if matched else "N/A",
            "note": "oracle line compared with CPG CALL node LINE_NUMBER of every template-matched call id",
        }
        config["methods"]["cpg_rca"]["log_to_code"] = log2code
        for row in summary_rows:
            if row["method"] == "cpg_rca":
                row["log_to_code_accuracy"] = log2code["accuracy"]
    except Exception as exc:  # noqa: BLE001
        config["methods"]["cpg_rca"]["status"] = "error"
        config["methods"]["cpg_rca"]["error"] = f"{type(exc).__name__}: {exc}"
        summary_rows.extend(summarize([], "cpg_rca", status="error"))

    # ── PRINS ────────────────────────────────────────────────────────────
    prins_cfg = {"k": args.prins_k, "u": args.prins_u, "mint_timeout_s": args.mint_timeout,
                 "ignore_values": True, "train_traces": len(train_ids),
                 "alphabet_encoding": "canonical messages mapped to whitespace-free ids E0000.. (MINT input format)"}
    config["methods"]["PRINS"] = {"type": "log-based (train traces -> MINT -> hybrid determinization)", **prins_cfg}
    prins_dir = args.out / "PRINS"
    prins_dir.mkdir(parents=True, exist_ok=True)
    prins_cfg["artifacts_dir"] = str(prins_dir)
    prins_pkl = prins_dir / "prins_retryservice.pkl"
    try:
        if prins_pkl.is_file() and not args.retrain_prins:
            with open(prins_pkl, "rb") as fh:
                cached = pickle.load(fh)
            prins_model = PrinsModel(cached["dfa"], cached["vocab"])
            prins_build_s = "cached"
        else:
            prins_model, prins_build_s = build_prins(
                traces, train_ids, prins_dir / "prins_work",
                k=args.prins_k, u=args.prins_u, mint_timeout=args.mint_timeout,
                prins_pkl=prins_pkl, repo_root=REPO_ROOT,
            )
        rows = evaluate_model("PRINS", prins_model, cases)
        per_execution_rows.extend(rows)
        coverage = event_coverage(prins_model, cases)
        summary_rows.extend(summarize(rows, "PRINS", build_time_s=prins_build_s,
                                      coverage=coverage, extra={"log_to_code_accuracy": "N/A"}))
        config["methods"]["PRINS"]["build_time_s"] = prins_build_s
        config["methods"]["PRINS"]["event_coverage"] = coverage
        config["methods"]["PRINS"]["states"] = len(prins_model.dfa.states)
        config["methods"]["PRINS"]["transitions"] = len(prins_model.dfa.transitions)
        config["methods"]["PRINS"]["log_to_code"] = "N/A (no code-level output)"
        (prins_dir / "vocab.json").write_text(
            json.dumps(prins_model.vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        config["methods"]["PRINS"]["status"] = "error"
        config["methods"]["PRINS"]["error"] = f"{type(exc).__name__}: {exc}"
        summary_rows.extend(summarize([], "PRINS", status="error"))

    # ── LogSynthFSM (FSM stage: parsing + State Discovery) ────────────────
    lsf_cfg = {
        "type": "log-based (train traces -> regex parsing -> LLM State Discovery FSM)",
        "scope": "FSM stage only (not the synthesis / anomaly-detection pipeline)",
        "parse_mode": "regex",
        "train_traces": len(train_ids),
    }
    config["methods"]["LogSynthFSM"] = lsf_cfg
    lsf_dir = args.out / "LogSynthFSM"
    lsf_dir.mkdir(parents=True, exist_ok=True)
    lsf_cfg["artifacts_dir"] = str(lsf_dir)
    provider = logsynthfsm_provider()
    if provider not in ("openai", "ollama"):
        lsf_cfg["status"] = "blocked"
        lsf_cfg["blocked_reason"] = provider
        summary_rows.extend(summarize([], "LogSynthFSM", status="blocked",
                                      extra={"blocked_reason": provider}))
    else:
        lsf_cfg["provider"] = provider
        try:
            lsf_model, lsf_fsm, lsf_build_s = build_logsynthfsm(
                traces, train_ids, lsf_dir / "logsynthfsm_work", provider)
            rows = evaluate_model("LogSynthFSM", lsf_model, cases)
            per_execution_rows.extend(rows)
            coverage = event_coverage(lsf_model, cases)
            summary_rows.extend(summarize(rows, "LogSynthFSM", build_time_s=round(lsf_build_s, 3),
                                          coverage=coverage,
                                          extra={"log_to_code_accuracy": "N/A"}))
            lsf_cfg["build_time_s"] = round(lsf_build_s, 3)
            lsf_cfg["event_coverage"] = coverage
            lsf_cfg["states"] = len(lsf_fsm.states)
            lsf_cfg["transitions"] = len(lsf_fsm.transitions)
            lsf_cfg["log_to_code"] = "N/A (no code-level output)"
        except Exception as exc:  # noqa: BLE001
            lsf_cfg["status"] = "error"
            lsf_cfg["error"] = f"{type(exc).__name__}: {exc}"
            summary_rows.extend(summarize([], "LogSynthFSM", status="error"))

    # ── speed comparison block ────────────────────────────────────────────
    for method in ("cpg_rca", "PRINS", "LogSynthFSM"):
        m_rows = [r for r in per_execution_rows if r["method"] == method]
        check_total = sum(float(r["check_time_s"]) for r in m_rows)
        build_time = config["methods"][method].get("build_time_s", NA)
        try:
            build_time = round(float(build_time), 3)
        except (TypeError, ValueError):
            pass  # keep "cached" / NA as-is
        speed_row = {
            "method": method,
            "block": "speed",
            "status": config["methods"][method].get("status", "ok" if m_rows else NA),
            "build_time_s": build_time,
            "event_coverage": config["methods"][method].get("event_coverage", NA),
            "n_cases": len(m_rows),
            "check_time_total_s": round(check_total, 6),
            "check_time_mean_s": round(check_total / len(m_rows), 6) if m_rows else NA,
            "traces_per_second": round(len(m_rows) / check_total, 1) if check_total else NA,
            "log_to_code_accuracy": (
                config["methods"][method].get("log_to_code", {}).get("accuracy", "N/A")
                if isinstance(config["methods"][method].get("log_to_code"), dict) else "N/A"),
        }
        summary_rows.append(speed_row)

    # ── write outputs ─────────────────────────────────────────────────────
    per_path = args.out / "per_execution.csv"
    with open(per_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_execution_rows[0].keys()),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_execution_rows)

    summary_path = args.out / "summary.csv"
    fieldnames = list(dict.fromkeys(k for row in summary_rows for k in row))
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    config_path = args.out / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {per_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
