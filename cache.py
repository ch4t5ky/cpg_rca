"""
cache.py
========
Unified cache system for the CPG-based RCA pipeline.

Shared between pipeline.py, rca_pipeline.py, and any other analysis scripts.
Cache is stored in a configurable directory and reused across runs.
"""

import gc
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from log2cpg2 import process_service_cpg, resolve_inter_service_edges, fast_match


class UnifiedCache:
    """Unified cache system for the entire RCA pipeline."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self.indexes_file = self.cache_dir / "cpg_indexes.pkl"
        self.edges_file = self.cache_dir / "call_edges.pkl"
        self.method_calls_file = self.cache_dir / "method_calls.pkl"
        self.lifecycle_file = self.cache_dir / "lifecycle_events.pkl"
        self.logs_file = self.cache_dir / "raw_logs.pkl"

    def get_cache_key(self, csv_path: str, services_dir: str) -> str:
        csv_mtime = Path(csv_path).stat().st_mtime
        key_str = f"{csv_path}:{csv_mtime}:{services_dir}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def exists(self) -> bool:
        return (
            self.metadata_file.exists()
            and self.indexes_file.exists()
            and self.edges_file.exists()
            and self.method_calls_file.exists()
        )

    def is_valid(self, csv_path: str, services_dir: str) -> bool:
        if not self.exists():
            return False
        try:
            with open(self.metadata_file) as f:
                metadata = json.load(f)
            return metadata.get("cache_key") == self.get_cache_key(csv_path, services_dir)
        except Exception:
            return False

    def build(self, csv_path: str, services_dir: str):
        """Build unified cache from logs CSV and CPG directory."""
        print("\n" + "=" * 70)
        print("BUILDING UNIFIED CACHE")
        print("=" * 70)

        # Load logs
        print(f"\n[1] Loading logs: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        df["container_name"] = df["container_name"].fillna("unknown").astype(str).str.strip()
        df["message"] = df["message"].fillna("").astype(str)

        print(f"  Logs: {len(df)}")
        print(f"  Services: {df['container_name'].nunique()}")
        print(
            f"  Time: {datetime.fromtimestamp(df['timestamp'].min()).strftime('%H:%M:%S')}"
            f" to {datetime.fromtimestamp(df['timestamp'].max()).strftime('%H:%M:%S')}"
        )

        # Load CPG
        print(f"\n[2] Loading CPG: {services_dir}")
        services = set(df["container_name"].unique())
        indexes = {}
        all_edges = []
        base_dir = Path(services_dir)

        for service in sorted(services):
            dot_file = base_dir / service / "export.dot"
            if not dot_file.exists():
                continue

            service_logs = df[df["container_name"] == service]
            log_rows = [
                (
                    datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                    row["message"],
                )
                for _, row in service_logs.iterrows()
            ]

            print(f"  Loading {service}...")
            _, edges, index = process_service_cpg(str(dot_file), service, log_rows[:2000])
            indexes[service] = index
            all_edges.extend(edges)

        # Inter-service edges
        if len(indexes) > 1:
            print(f"\n[3] Resolving cross-service calls...")
            inter_edges = resolve_inter_service_edges(indexes)
            all_edges.extend(inter_edges)
            print(f"  Cross-service edges: {len(inter_edges)}")

        # Map logs to methods
        print(f"\n[4] Mapping logs to methods...")
        method_calls = []
        lifecycle_events = []

        for _, row in df.iterrows():
            service = row["container_name"]
            message = str(row["message"])
            timestamp = float(row["timestamp"])
            msg_lower = message.lower()

            if service in indexes:
                result = fast_match(message, indexes[service])
                if result["matched"] and result["score"] > 0.01:
                    method_calls.append(
                        {
                            "timestamp": timestamp,
                            "service": service,
                            "method": result["function_name"],
                            "qualified_name": f"{service}::{result['function_name']}",
                            "is_error": any(
                                p in msg_lower for p in ["error", "fail", "exception"]
                            ),
                            "message": message,
                            "score": result["score"],
                        }
                    )

        print(
            f"  Mapped: {len(method_calls)}/{len(df)} logs"
            f" ({len(method_calls) / len(df) * 100:.1f}%)"
        )
        print(f"  Lifecycle events: {len(lifecycle_events)}")

        # Persist
        print(f"\n[5] Saving cache to: {self.cache_dir}")
        with open(self.indexes_file, "wb") as f:
            pickle.dump(indexes, f)
        with open(self.edges_file, "wb") as f:
            pickle.dump(all_edges, f)
        with open(self.method_calls_file, "wb") as f:
            pickle.dump(method_calls, f)
        with open(self.lifecycle_file, "wb") as f:
            pickle.dump(lifecycle_events, f)
        with open(self.logs_file, "wb") as f:
            pickle.dump(df, f)

        metadata = {
            "cache_key": self.get_cache_key(csv_path, services_dir),
            "created": datetime.now().isoformat(),
            "csv_path": csv_path,
            "services_dir": services_dir,
            "total_logs": len(df),
            "mapped_logs": len(method_calls),
            "services": list(indexes.keys()),
            "lifecycle_events": len(lifecycle_events),
            "call_edges": len(all_edges),
        }
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Cache contents:")
        print(f"    - CPG indexes: {len(indexes)} services")
        print(f"    - Call edges: {len(all_edges)}")
        print(f"    - Method calls: {len(method_calls)}")
        print(f"    - Lifecycle events: {len(lifecycle_events)}")

    def load(self) -> Dict:
        """Load all cached data. Returns dict with metadata, indexes, edges, etc."""
        print(f"\n[Cache] Loading from: {self.cache_dir}")

        with open(self.metadata_file) as f:
            metadata = json.load(f)
        with open(self.indexes_file, "rb") as f:
            indexes = pickle.load(f)
        with open(self.edges_file, "rb") as f:
            edges = pickle.load(f)
        with open(self.method_calls_file, "rb") as f:
            method_calls = pickle.load(f)
        with open(self.lifecycle_file, "rb") as f:
            lifecycle = pickle.load(f)
        with open(self.logs_file, "rb") as f:
            logs = pickle.load(f)

        print(f"  Loaded {len(method_calls)} method calls, {len(edges)} edges")

        return {
            "metadata": metadata,
            "indexes": indexes,
            "edges": edges,
            "method_calls": method_calls,
            "lifecycle": lifecycle,
            "logs": logs,
        }
