"""Adapter from this repo's incident dataset to the PRINS input format.

An incident directory (e.g. ``dataset/re3ob_adservice_f3_1``) contains
``logs.csv`` (timestamp, container_name, message) and ``traces.csv``
(traceID, startTime, duration). Each traceID defines a time window; the log
entries inside that window form one system trace (execution), as required by
PRINS.

Log messages have no pre-extracted event templates, so pseudo-templates are
derived by normalizing volatile substrings (numbers, hex ids, uuids) to ``<*>``;
each distinct normalized message becomes one event type (``tid``).
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PRINS_PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRINS", "PRINS")
if PRINS_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PRINS_PACKAGE_DIR)

_HEX_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HEX_LONG = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_NUMBERS = re.compile(r"\d+")


def normalize_message(message: str) -> str:
    """Collapse volatile parts of a log message into a pseudo event template."""
    msg = _HEX_UUID.sub("<*>", message)
    msg = _HEX_LONG.sub("<*>", msg)
    msg = _NUMBERS.sub("<*>", msg)
    return re.sub(r"\s+", " ", msg).strip()


def _tid_for(template: str) -> str:
    digest = hashlib.md5(template.encode("utf-8")).hexdigest()[:8]
    words = re.sub(r"[^A-Za-z]+", "_", template)[:30].strip("_") or "evt"
    return f"E_{words}_{digest}"


def load_incident(incident_dir: str | Path, max_traces: int | None = None):
    """Load an incident directory as PRINS l_vectors.

    Returns (l_vectors, templates) where l_vectors maps log_id -> list of
    entries {'ts', 'tid', 'values', 'component'} and templates maps
    tid -> pseudo template string.
    """
    incident_dir = Path(incident_dir)

    # trace time windows
    by_trace: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(incident_dir / "traces.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                start = float(row["startTime"]) / 1e6
                end = start + float(row["duration"]) / 1e6
            except (ValueError, KeyError):
                continue
            by_trace[row["traceID"]].append((start, end))
    windows = sorted((min(s for s, _ in spans), max(e for _, e in spans)) for spans in by_trace.values())
    starts = [w[0] for w in windows]

    # assign log entries to trace windows
    buckets: dict[int, list[tuple[float, str, str]]] = defaultdict(list)
    with open(incident_dir / "logs.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ts = float(row["timestamp"])
            except (ValueError, KeyError):
                continue
            comp = row["container_name"]
            tid = _tid_for(normalize_message(row["message"]))
            for i in range(bisect.bisect_right(starts, ts), len(windows)):
                if windows[i][0] > ts:
                    break
                if ts <= windows[i][1]:
                    buckets[i].append((ts, comp, tid))

    l_vectors = {}
    templates = {}
    for log_id, (_, entries) in enumerate(sorted(buckets.items()), start=1):
        if not entries:
            continue
        l_vectors[log_id] = [
            {"ts": f"{ts:.6f}", "tid": tid, "values": [], "component": comp}
            for ts, comp, tid in sorted(entries)
        ]
    for vector in l_vectors.values():
        for e in vector:
            templates.setdefault(e["tid"], e["tid"])

    if max_traces is not None and len(l_vectors) > max_traces:
        import random

        kept = sorted(random.Random(42).sample(sorted(l_vectors), max_traces))
        l_vectors = {i: l_vectors[k] for i, k in enumerate(kept, start=1)}
    return l_vectors, templates


def list_incidents(dataset_dir: str | Path, name_filter: str | None = None) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    dirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir() and (p / "logs.csv").is_file()]
    if name_filter:
        dirs = [p for p in dirs if name_filter in p.name]
    return dirs
