"""Runtime log order and chain expiry keep sub-second precision."""

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_analyzer  # noqa: E402
from runtime_analyzer import build_incident_context, prepare_logs, process_window  # noqa: E402
from src.offline.finite_state_machine import ExecutionSegment, LogTransition, StaticLogFSM  # noqa: E402
from src.online.stateful_chain_store import ExplicitHypothesisStore  # noqa: E402
from src.online.stream_processor import RuntimeStreamProcessor, ServiceCatalog  # noqa: E402

BASE = 1731903240  # 2024-11-18T04:14:00Z


class Recorder:
    """Stands in for the stream processor and records events in processing order."""

    def __init__(self):
        self.events = []

    def process(self, event):
        self.events.append(event)
        return []


def read_csv(text):
    """Same read as runtime_analyzer.main: every cell is a string."""
    return pd.read_csv(io.StringIO(text), dtype=str)


def run(raw, processor=None, unit="s"):
    processor = processor or Recorder()
    args = argparse.Namespace(
        timestamp_column=None, service_column=None, message_column=None,
        bucket_column=None, call_node_column=None, timestamp_unit=unit,
    )
    reports = process_window(prepare_logs(raw, args), processor, progress_every=0)
    return processor, reports


def alpha_omega_chain(first, second, time_gap_sec, unit="s"):
    """Run alpha then omega on one service with a two-step FSM; returns the chain."""
    fsm = StaticLogFSM("m", "handle", "svc.handle")
    fsm.states = {
        "q0": ExecutionSegment("q0", "svc.handle", "svc.handle", (), (), (), None, None, True, False, "BETWEEN_LOGS"),
        "q1": ExecutionSegment("q1", "svc.handle", "svc.handle", (), (), (), None, None, False, False, "BETWEEN_LOGS"),
        "q2": ExecutionSegment("q2", "svc.handle", "svc.handle", (), (), (), None, None, False, True, "RETURN_SEGMENT"),
    }
    fsm.transitions = [
        LogTransition("t0", "q0", "q1", "alpha", "l0", "svc.handle", "m"),
        LogTransition("t1", "q1", "q2", "omega", "l1", "svc.handle", "m"),
    ]
    store = ExplicitHypothesisStore({"k": fsm}, threshold=0.5, time_gap_sec=time_gap_sec)
    processor = RuntimeStreamProcessor({"svc": ServiceCatalog(fsms={"k": fsm}, flows={"k": {}}, store=store)})
    raw = pd.DataFrame({"timestamp": [str(first), str(second)], "service": "svc", "message": ["alpha", "omega"]})
    _, reports = run(raw, processor, unit)
    return store.active_chains[0], reports


def test_order_inside_one_second_across_services():
    # "late" is first in the file but happened 0.6 s after "early".
    raw = read_csv(f"timestamp,service,message\n{BASE * 1000 + 700},web,late\n{BASE * 1000 + 100},db,early\n")
    events = run(raw, unit="ms")[0].events
    assert [e.message for e in events] == ["early", "late"]
    assert events[0].timestamp == pd.Timestamp("2024-11-18T04:14:00.100Z")


def test_nanoseconds_keep_order_and_bad_rows_are_dropped():
    start = BASE * 10**9
    raw = read_csv(f"timestamp,service,message\n{start + 2},web,b\n,web,blank\n{start + 1},web,a\nnot-a-time,web,bad\n")
    events = run(raw, unit="ns")[0].events
    assert [e.message for e in events] == ["a", "b"]
    assert (events[1].timestamp - events[0].timestamp) == pd.Timedelta(1, unit="ns")


def test_whole_seconds_give_master_order_and_buckets():
    raw = read_csv(f"timestamp,service,message\n{BASE + 1},web,c\n{BASE},db,a\n{BASE},web,b\n")
    events = run(raw)[0].events
    assert [e.message for e in events] == ["a", "b", "c"]  # equal times keep file order
    assert [e.bucket for e in events] == [f"db@{BASE}", f"web@{BASE}", f"web@{BASE + 1}"]


def test_chain_expiry_uses_fractional_gap():
    assert alpha_omega_chain(100.1, 130.9, time_gap_sec=30)[0].status == "expired"
    assert alpha_omega_chain(100.1, 130.0, time_gap_sec=30)[0].status != "expired"


def test_json_has_iso_timestamps_and_schema_4():
    _, reports = alpha_omega_chain(BASE * 1000 + 100, BASE * 1000 + 3100, time_gap_sec=30, unit="ms")
    inject = pd.Timestamp("2024-11-18T04:14:00Z")
    context = build_incident_context(reports, inject, inject, inject + pd.Timedelta(seconds=60))
    payload = json.loads(json.dumps(context, default=runtime_analyzer.json_default))
    assert payload["schema_version"] == "4.0"
    assert payload["incident_window"]["inject_time"] == "2024-11-18T04:14:00+00:00"
    assert [item["timestamp"] for item in payload["timeline"]] == [
        "2024-11-18T04:14:00.100000+00:00",
        "2024-11-18T04:14:03.100000+00:00",
    ]
