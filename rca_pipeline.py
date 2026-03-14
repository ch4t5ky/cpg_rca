#!/usr/bin/env python3
"""
pipeline.py (Enhanced with Execution Flows)
===========================================
Main CPG-based RCA Pipeline with Execution Flow Reconstruction

NEW FEATURES:
- Reconstructs execution flows by connecting method calls through:
  1. Call graph edges (caller → callee relationships from CPG)
  2. Temporal proximity (methods called close in time)
- Groups related method calls into execution flows
- Tracks flows across service boundaries
- Identifies root methods and leaf methods in each flow

This is the MAIN script that:
1. Preprocesses logs and CPG files
2. Builds UNIFIED cache (shared across all tools)
3. Reconstructs execution flows within error windows
4. Generates timeline visualization with flow connections
5. Exports data for LLM RCA

Usage:
    # Build cache and visualize with flows
    python pipeline.py --csv logs.csv --services-dir services --visualize
    
    # Force rebuild cache
    python pipeline.py --csv logs.csv --services-dir services --rebuild-cache
    
    # Adjust temporal window for flow grouping
    python pipeline.py --csv logs.csv --services-dir services --visualize --flow-window 3.0

Output:
    cache/
    ├── metadata.json              # Cache info
    ├── cpg_indexes.pkl            # CPG data (reusable)
    ├── call_edges.pkl             # Call relationships
    ├── method_calls.pkl           # All log→method mappings
    ├── lifecycle_events.pkl       # Service start/stop/crash
    └── raw_logs.pkl               # Original logs
    
    output/
    ├── timeline.png               # Visualization with flows
    ├── timeline_methods.json      # Method details
    ├── execution_flows.json       # Reconstructed flows
    └── llm_input.json             # Data for LLM RCA (with flows)
"""

import argparse
import pickle
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from log2cpg2 import process_service_cpg, resolve_inter_service_edges, fast_match


# ===========================================================================
# UNIFIED CACHE (same as before)
# ===========================================================================

class UnifiedCache:
    """Unified cache system for entire RCA pipeline."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files
        self.metadata_file = self.cache_dir / "metadata.json"
        self.indexes_file = self.cache_dir / "cpg_indexes.pkl"
        self.edges_file = self.cache_dir / "call_edges.pkl"
        self.method_calls_file = self.cache_dir / "method_calls.pkl"
        self.lifecycle_file = self.cache_dir / "lifecycle_events.pkl"
        self.logs_file = self.cache_dir / "raw_logs.pkl"
    
    def get_cache_key(self, csv_path: str, services_dir: str) -> str:
        """Generate cache key."""
        csv_mtime = Path(csv_path).stat().st_mtime
        key_str = f"{csv_path}:{csv_mtime}:{services_dir}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def exists(self) -> bool:
        """Check if cache exists."""
        return (self.metadata_file.exists() and 
                self.indexes_file.exists() and 
                self.edges_file.exists() and
                self.method_calls_file.exists())
    
    def is_valid(self, csv_path: str, services_dir: str) -> bool:
        """Check if cache is valid."""
        if not self.exists():
            return False
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            current_key = self.get_cache_key(csv_path, services_dir)
            return metadata.get('cache_key') == current_key
        except:
            return False
    
    def build(self, csv_path: str, services_dir: str):
        """Build unified cache."""
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
        print(f"  Time: {datetime.fromtimestamp(df['timestamp'].min()).strftime('%H:%M:%S')} to {datetime.fromtimestamp(df['timestamp'].max()).strftime('%H:%M:%S')}")
        
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
            log_rows = [(
                datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                row["message"]
            ) for _, row in service_logs.iterrows()]
            
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
        
        # Map all logs to methods
        print(f"\n[4] Mapping logs to methods...")
        method_calls = []
        lifecycle_events = []
        
        for _, row in df.iterrows():
            service = row["container_name"]
            message = str(row["message"])
            timestamp = float(row["timestamp"])
            msg_lower = message.lower()
            
            # Method mapping
            if service in indexes:
                result = fast_match(message, indexes[service])
                if result['matched'] and result['score'] > 0.01:
                    method_calls.append({
                        'timestamp': timestamp,
                        'service': service,
                        'method': result['function_name'],
                        'qualified_name': f"{service}::{result['function_name']}",
                        'is_error': any(p in msg_lower for p in ['error', 'fail', 'exception']),
                        'message': message,
                        'score': result['score'],
                    })
        
        print(f"  Mapped: {len(method_calls)}/{len(df)} logs ({len(method_calls)/len(df)*100:.1f}%)")
        print(f"  Lifecycle events: {len(lifecycle_events)}")
        
        # Save cache
        print(f"\n[5] Saving cache to: {self.cache_dir}")
        
        with open(self.indexes_file, 'wb') as f:
            pickle.dump(indexes, f)
        
        with open(self.edges_file, 'wb') as f:
            pickle.dump(all_edges, f)
        
        with open(self.method_calls_file, 'wb') as f:
            pickle.dump(method_calls, f)
        
        with open(self.lifecycle_file, 'wb') as f:
            pickle.dump(lifecycle_events, f)
        
        with open(self.logs_file, 'wb') as f:
            pickle.dump(df, f)
        
        # Metadata
        metadata = {
            'cache_key': self.get_cache_key(csv_path, services_dir),
            'created': datetime.now().isoformat(),
            'csv_path': csv_path,
            'services_dir': services_dir,
            'total_logs': len(df),
            'mapped_logs': len(method_calls),
            'services': list(indexes.keys()),
            'lifecycle_events': len(lifecycle_events),
            'call_edges': len(all_edges)
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Cache built: {len(method_calls)} calls, {len(all_edges)} edges")
    
    def load(self) -> Dict:
        """Load cached data."""
        print("\n[Cache] Loading cached data...")
        
        with open(self.indexes_file, 'rb') as f:
            indexes = pickle.load(f)
        
        with open(self.edges_file, 'rb') as f:
            edges = pickle.load(f)
        
        with open(self.method_calls_file, 'rb') as f:
            method_calls = pickle.load(f)
        
        with open(self.lifecycle_file, 'rb') as f:
            lifecycle = pickle.load(f)
        
        with open(self.logs_file, 'rb') as f:
            logs = pickle.load(f)
        
        print(f"  ✓ Loaded: {len(method_calls)} calls, {len(edges)} edges")
        
        return {
            'indexes': indexes,
            'edges': edges,
            'method_calls': method_calls,
            'lifecycle': lifecycle,
            'logs': logs
        }


# ===========================================================================
# EXECUTION FLOW RECONSTRUCTION
# ===========================================================================

class ExecutionFlowReconstructor:
    """
    Reconstructs execution flows from method calls using:
    1. Call graph edges (structural relationships from CPG)
    2. Temporal proximity (time-based grouping)
    
    Without trace IDs, we rely on:
    - Strong signal: Call graph edges (A calls B = same flow)
    - Weak signal: Time proximity (called within N seconds = likely same flow)
    """
    
    def __init__(self, edges: List[Tuple[str, str, str]], time_window: float = 2.0):
        """
        Args:
            edges: List of (caller, callee, kind) tuples
            time_window: Max time gap between calls in same flow (seconds)
        """
        self.time_window = time_window
        
        # Build call graph lookup
        self.call_graph = defaultdict(set)  # caller -> set of callees
        self.reverse_graph = defaultdict(set)  # callee -> set of callers
        
        for caller, callee, kind in edges:
            if kind == "CALL":
                self.call_graph[caller].add(callee)
                self.reverse_graph[callee].add(caller)
    
    def reconstruct_flows(self, method_calls: List[Dict]) -> List[Dict]:
        """
        Reconstruct execution flows from method calls.
        
        Strategy:
        1. Sort calls by timestamp
        2. Group calls that are either:
           - Connected by call graph edges (A→B means same flow)
           - Close in time (within time_window seconds)
        3. Identify root/leaf methods in each flow
        
        Returns:
            List of flow dictionaries, each containing:
            - flow_id: Unique identifier
            - method_calls: List of method calls in this flow
            - root_methods: Methods with no incoming calls in this flow
            - leaf_methods: Methods with no outgoing calls in this flow
            - services_involved: Set of services in this flow
            - start_time: First method call timestamp
            - end_time: Last method call timestamp
            - has_error: Whether flow contains any errors
        """
        if not method_calls:
            return []
        
        # Sort by timestamp
        sorted_calls = sorted(method_calls, key=lambda x: x['timestamp'])
        
        # Group using temporal proximity + call relationships
        flows = self._group_by_temporal_and_structural(sorted_calls)
        
        # Assign flow IDs
        for i, flow in enumerate(flows):
            flow['flow_id'] = f"flow_{i:04d}"
        
        return flows
    
    def _group_by_temporal_and_structural(self, method_calls: List[Dict]) -> List[Dict]:
        """
        Group method calls by temporal proximity and call relationships.
        
        Algorithm:
        - Start with first call as new flow
        - For each next call:
          - If within time window OR has call relationship with any call in current flow:
            -> Add to current flow
          - Else:
            -> Start new flow
        """
        if not method_calls:
            return []
        
        flows = []
        current_flow_calls = [method_calls[0]]
        
        for i in range(1, len(method_calls)):
            current_call = method_calls[i]
            
            # Check if should continue current flow
            should_continue = False
            
            # Strategy 1: Check temporal proximity with last call
            prev_call = method_calls[i-1]
            time_gap = current_call['timestamp'] - prev_call['timestamp']
            if time_gap <= self.time_window:
                should_continue = True
            
            # Strategy 2: Check call relationship with ANY call in current flow
            if not should_continue:
                current_qname = current_call['qualified_name']
                for flow_call in current_flow_calls:
                    flow_qname = flow_call['qualified_name']
                    
                    # Check if current calls any in flow, or any in flow calls current
                    if (current_qname in self.call_graph.get(flow_qname, set()) or
                        flow_qname in self.call_graph.get(current_qname, set()) or
                        current_qname in self.reverse_graph.get(flow_qname, set()) or
                        flow_qname in self.reverse_graph.get(current_qname, set())):
                        should_continue = True
                        break
            
            if should_continue:
                current_flow_calls.append(current_call)
            else:
                # Finish current flow and start new one
                if current_flow_calls:
                    flow = self._build_flow(current_flow_calls)
                    if flow:
                        flows.append(flow)
                current_flow_calls = [current_call]
        
        # Add last flow
        if current_flow_calls:
            flow = self._build_flow(current_flow_calls)
            if flow:
                flows.append(flow)
        
        return flows
    
    def _build_flow(self, method_calls: List[Dict]) -> Dict:
        """Build a single flow from a list of method calls."""
        if not method_calls:
            return None
        
        # Sort by timestamp
        sorted_calls = sorted(method_calls, key=lambda x: x['timestamp'])
        
        # Find root and leaf methods based on call graph
        qualified_names = {c['qualified_name'] for c in sorted_calls}
        
        root_methods = []
        leaf_methods = []
        
        for call in sorted_calls:
            qname = call['qualified_name']
            
            # Root: has no callers within this flow
            callers = self.reverse_graph.get(qname, set())
            if not any(caller in qualified_names for caller in callers):
                root_methods.append(qname)
            
            # Leaf: has no callees within this flow
            callees = self.call_graph.get(qname, set())
            if not any(callee in qualified_names for callee in callees):
                leaf_methods.append(qname)
        
        # Build flow
        return {
            'method_calls': sorted_calls,
            'root_methods': root_methods,
            'leaf_methods': leaf_methods,
            'services_involved': list(set(c['service'] for c in sorted_calls)),
            'start_time': sorted_calls[0]['timestamp'],
            'end_time': sorted_calls[-1]['timestamp'],
            'duration': sorted_calls[-1]['timestamp'] - sorted_calls[0]['timestamp'],
            'has_error': any(c['is_error'] for c in sorted_calls),
            'error_count': sum(1 for c in sorted_calls if c['is_error']),
        }


# ===========================================================================
# ERROR WINDOW
# ===========================================================================

def find_error_window(method_calls: List[Dict], window: int) -> Tuple[List[Dict], float, float]:
    """Find time window around first error."""
    errors = [c for c in method_calls if c['is_error']]
    
    if not errors:
        print("  [Warning] No errors found, using full timeline")
        sorted_calls = sorted(method_calls, key=lambda x: x['timestamp'])
        if not sorted_calls:
            return [], 0, 0
        start = sorted_calls[0]['timestamp']
        end = sorted_calls[-1]['timestamp']
        return sorted_calls, start, end - start
    
    first_error = min(e['timestamp'] for e in errors)
    start = max(first_error - window, min(c['timestamp'] for c in method_calls))
    end = first_error + window
    
    events = [c for c in method_calls if start <= c['timestamp'] <= end]
    events.sort(key=lambda x: x['timestamp'])
    
    return events, first_error, end - start


# ===========================================================================
# VISUALIZATION WITH FLOWS
# ===========================================================================

def create_visualization_with_flows(
    data: Dict,
    window: int,
    flow_window: float,
    output_dir: Path,
    dpi: int = 150
):
    """Create timeline visualization with execution flow connections."""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION WITH EXECUTION FLOWS")
    print("=" * 70)
    
    method_calls = data['method_calls']
    edges = data['edges']
    
    # Get error window
    events, first_error, time_range = find_error_window(method_calls, window)
    
    if not events:
        print("  [Error] No events in window")
        return {}
    
    # Reconstruct flows
    print(f"\n[1] Reconstructing execution flows...")
    flow_reconstructor = ExecutionFlowReconstructor(edges, time_window=flow_window)
    flows = flow_reconstructor.reconstruct_flows(events)
    
    print(f"  Found {len(flows)} execution flows")
    print(f"  Error flows: {sum(1 for f in flows if f['has_error'])}")
    
    # Build event index for quick lookup
    event_to_flow = {}
    for flow in flows:
        for call in flow['method_calls']:
            event_to_flow[id(call)] = flow['flow_id']
    
    # Create visualization
    print(f"\n[2] Creating visualization...")
    
    start_time = events[0]['timestamp']
    services = sorted(set(e['service'] for e in events))
    
    # Colors
    colors = {
        'frontend': '#4285F4',
        'userservice': '#34A853',
        'productservice': '#FBBC04',
        'orderservice': '#EA4335',
        'paymentservice': '#9C27B0',
        'shipmentservice': '#FF9800',
    }
    
    # Flow colors (for connecting lines)
    flow_colors = plt.cm.tab20(np.linspace(0, 1, len(flows)))
    flow_color_map = {f['flow_id']: flow_colors[i] for i, f in enumerate(flows)}
    
    # Setup figure
    fig_height = max(8, len(services) * 2 + 2)
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.set_xlim(-0.5, time_range + 0.5)
    ax.set_ylim(-1, len(services) * 2.5 + 1)
    ax.axis('off')
    
    # Service lanes
    y_positions = {}
    for i, service in enumerate(services):
        y = len(services) * 2.5 - i * 2.5
        y_positions[service] = y
        
        color = colors.get(service, '#78909C')
        
        ax.add_patch(plt.Rectangle(
            (0, y - 0.8), time_range, 1.6,
            facecolor=f'{color}10',
            edgecolor=f'{color}80',
            linewidth=1.5,
            zorder=1
        ))
        
        ax.text(-0.3, y, service,
               ha='right', va='center',
               fontsize=11, fontweight='bold',
               color=color)
    
    # Plot events and build position map
    positions = {}
    event_labels = {}
    method_details = {'events': [], 'flows': []}
    
    for i, evt in enumerate(events):
        rel_time = evt['timestamp'] - start_time
        x = rel_time
        y = y_positions[evt['service']]
        
        positions[i] = (x, y)
        
        # Abbreviated label
        method_name = evt['method']
        if len(method_name) > 8:
            parts = [p for p in method_name.split('_') if p]
            if len(parts) > 1:
                label = ''.join(p[0].upper() for p in parts[:2])
            else:
                label = method_name[:3].upper()
        else:
            label = method_name[:4].upper()
        
        event_labels[i] = label
        
        # Store details
        method_details['events'].append({
            'index': i,
            'service': evt['service'],
            'method': evt['method'],
            'qualified_name': evt['qualified_name'],
            'timestamp': datetime.fromtimestamp(evt['timestamp']).isoformat(),
            'relative_time': round(rel_time, 3),
            'is_error': evt['is_error'],
            'message': evt['message'],
            'flow_id': event_to_flow.get(id(evt)),
            'label': label
        })
    
    # Draw flow connections
    print(f"\n[3] Drawing flow connections...")
    drawn_connections = set()
    
    for flow in flows:
        flow_id = flow['flow_id']
        flow_color = flow_color_map[flow_id]
        
        # Get event indices for this flow
        flow_event_indices = []
        for i, evt in enumerate(events):
            if id(evt) in [id(c) for c in flow['method_calls']]:
                flow_event_indices.append(i)
        
        # Draw connections based on call graph edges
        for i in range(len(flow_event_indices)):
            for j in range(i + 1, len(flow_event_indices)):
                idx1, idx2 = flow_event_indices[i], flow_event_indices[j]
                evt1, evt2 = events[idx1], events[idx2]
                
                # Check if there's a call relationship
                has_edge = (
                    (evt1['qualified_name'], evt2['qualified_name'], 'CALL') in edges or
                    (evt2['qualified_name'], evt1['qualified_name'], 'CALL') in edges
                )
                
                if has_edge:
                    connection_key = (min(idx1, idx2), max(idx1, idx2), flow_id)
                    if connection_key in drawn_connections:
                        continue
                    drawn_connections.add(connection_key)
                    
                    x1, y1 = positions[idx1]
                    x2, y2 = positions[idx2]
                    
                    # Draw curved arrow
                    is_cross_service = evt1['service'] != evt2['service']
                    
                    arrow = FancyArrowPatch(
                        (x1, y1), (x2, y2),
                        connectionstyle=f"arc3,rad={'0.3' if is_cross_service else '0.2'}",
                        arrowstyle='->,head_width=0.3,head_length=0.3',
                        color=flow_color,
                        linewidth=2.5 if is_cross_service else 1.5,
                        alpha=0.7 if is_cross_service else 0.5,
                        zorder=10 if is_cross_service else 5
                    )
                    ax.add_patch(arrow)
    
    print(f"  Drew {len(drawn_connections)} flow connections")
    
    # Draw method circles
    for i, evt in enumerate(events):
        if i not in positions:
            continue
        
        x, y = positions[i]
        color = colors.get(evt['service'], '#78909C')
        is_error = evt['is_error']
        
        radius = 0.14 if not is_error else 0.18
        circle_color = '#FF4444' if is_error else color
        
        # Shadow
        ax.add_patch(Circle((x + 0.015, y - 0.015), radius,
                           facecolor='#00000015', edgecolor='none', zorder=18))
        
        # Circle
        ax.add_patch(Circle((x, y), radius,
                           facecolor='white' if not is_error else circle_color,
                           edgecolor=circle_color,
                           linewidth=2.5 if is_error else 1.8,
                           zorder=20))
        
        # Label
        label = event_labels[i]
        ax.text(x, y, label,
               ha='center', va='center',
               fontsize=5 if not is_error else 6,
               fontweight='bold',
               color=circle_color if not is_error else 'white',
               zorder=21)
    
    # Time axis
    n_ticks = min(30, int(time_range) + 1)
    tick_pos = np.linspace(0, time_range, n_ticks)
    tick_labels = [datetime.fromtimestamp(start_time + t).strftime('%H:%M:%S')
                   for t in tick_pos]
    
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xticks(tick_pos)
    ax_time.set_xticklabels(tick_labels, fontsize=9, rotation=45)
    ax_time.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)
    ax_time.spines['left'].set_visible(False)
    
    # Title
    title_y = len(services) * 2.5 + 1.2
    ax.text(time_range / 2, title_y, 'Service Timeline with Execution Flows',
           ha='center', fontsize=16, fontweight='bold')
    
    start_str = datetime.fromtimestamp(start_time).strftime('%H:%M:%S')
    ax.text(time_range / 2, title_y - 0.4,
           f'{start_str} | {len(events)} calls | {len(flows)} flows | {len(services)} services',
           ha='center', fontsize=10, color='#666')
    
    # Legend
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444',
               markersize=10, markeredgewidth=2.5, markeredgecolor='#FF4444', label='Error'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=8, markeredgewidth=1.8, markeredgecolor='#4285F4', label='Method'),
        Line2D([0], [0], color='gray', linewidth=2.5, label='Cross-Service Call'),
        Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.5, label='Same-Service Call'),
    ]
    
    ax.legend(handles=legend, loc='lower right', fontsize=10, framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(exist_ok=True)
    viz_file = output_dir / "timeline.png"
    fig.savefig(viz_file, bbox_inches='tight', dpi=dpi)
    plt.close()
    
    print(f"\n  ✓ Visualization: {viz_file}")
    print(f"    Flow window: {flow_window}s, Found {len(flows)} flows")
    
    # Add flow details
    for flow in flows:
        method_details['flows'].append({
            'flow_id': flow['flow_id'],
            'root_methods': flow['root_methods'],
            'leaf_methods': flow['leaf_methods'],
            'services_involved': flow['services_involved'],
            'start_time': datetime.fromtimestamp(flow['start_time']).isoformat(),
            'end_time': datetime.fromtimestamp(flow['end_time']).isoformat(),
            'duration': round(flow['duration'], 3),
            'has_error': flow['has_error'],
            'error_count': flow['error_count'],
            'method_count': len(flow['method_calls']),
        })
    
    # Save JSON
    json_file = output_dir / "timeline_methods.json"
    with open(json_file, 'w') as f:
        json.dump(method_details, f, indent=2)
    
    print(f"  ✓ Method details: {json_file}")
    
    # Save flows separately
    flows_file = output_dir / "execution_flows.json"
    with open(flows_file, 'w') as f:
        json.dump(method_details['flows'], f, indent=2)
    
    print(f"  ✓ Execution flows: {flows_file}")
    
    return method_details


# ===========================================================================
# LLM EXPORT WITH FLOWS
# ===========================================================================

def export_for_llm_with_flows(data: Dict, window: int, flow_window: float, output_dir: Path):
    """Export data for LLM RCA with execution flow information."""
    print("\n" + "=" * 70)
    print("EXPORTING FOR LLM RCA (WITH FLOWS)")
    print("=" * 70)
    
    method_calls = data['method_calls']
    edges = data['edges']
    lifecycle = data['lifecycle']
    
    # Get error window
    events, first_error, time_range = find_error_window(method_calls, window)
    
    # Reconstruct flows
    flow_reconstructor = ExecutionFlowReconstructor(edges, time_window=flow_window)
    flows = flow_reconstructor.reconstruct_flows(events)
    
    # Build LLM data
    llm_data = {
        'summary': {
            'total_method_calls': len(events),
            'error_count': sum(1 for e in events if e['is_error']),
            'services': list(set(e['service'] for e in events)),
            'time_range_seconds': time_range,
            'first_error_time': datetime.fromtimestamp(first_error).isoformat() if first_error else None,
            'total_flows': len(flows),
            'error_flows': sum(1 for f in flows if f['has_error']),
        },
        'errors': [
            {
                'service': e['service'],
                'method': e['method'],
                'timestamp': datetime.fromtimestamp(e['timestamp']).isoformat(),
                'message': e['message'],
            }
            for e in events if e['is_error']
        ],
        'execution_flows': [
            {
                'flow_id': f['flow_id'],
                'root_methods': f['root_methods'],
                'leaf_methods': f['leaf_methods'],
                'services_involved': f['services_involved'],
                'duration_seconds': round(f['duration'], 3),
                'has_error': f['has_error'],
                'error_count': f['error_count'],
                'method_sequence': [
                    {
                        'service': c['service'],
                        'method': c['method'],
                        'qualified_name': c['qualified_name'],
                        'timestamp': datetime.fromtimestamp(c['timestamp']).isoformat(),
                        'is_error': c['is_error'],
                    }
                    for c in f['method_calls']
                ],
            }
            for f in flows
        ],
        'lifecycle_events': [
            {
                'service': lc['service'],
                'type': lc['type'],
                'timestamp': datetime.fromtimestamp(lc['timestamp']).isoformat(),
                'message': lc['message']
            }
            for lc in lifecycle
        ],
        'call_relationships': [
            {'caller': caller, 'callee': callee}
            for caller, callee, kind in edges if kind == "CALL"
        ]
    }
    
    output_dir.mkdir(exist_ok=True)
    llm_file = output_dir / "llm_input.json"
    
    with open(llm_file, 'w') as f:
        json.dump(llm_data, f, indent=2)
    
    print(f"\n  ✓ LLM input: {llm_file}")
    print(f"  ✓ Errors: {len(llm_data['errors'])}")
    print(f"  ✓ Execution flows: {len(llm_data['execution_flows'])}")
    print(f"  ✓ Lifecycle events: {len(llm_data['lifecycle_events'])}")
    print(f"  ✓ Method calls: {len(events)}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CPG-based RCA Pipeline with Execution Flow Reconstruction"
    )
    
    parser.add_argument('--csv', required=True, help='Path to logs CSV')
    parser.add_argument('--services-dir', required=True, help='Path to CPG directory')
    parser.add_argument('--window', type=int, default=10, help='Time window around error (seconds)')
    parser.add_argument('--flow-window', type=float, default=2.0, 
                       help='Max time gap between calls in same flow (seconds)')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory')
    parser.add_argument('--dpi', type=int, default=150, help='Visualization DPI')
    
    # Output control
    parser.add_argument('--visualize', action='store_true', help='Generate timeline visualization')
    parser.add_argument('--no-llm-export', action='store_true', help='Disable LLM export')
    
    # Cache control
    parser.add_argument('--rebuild-cache', action='store_true', help='Force rebuild cache')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CPG-BASED RCA PIPELINE (WITH EXECUTION FLOWS)")
    print("=" * 70)
    
    cache = UnifiedCache(args.cache_dir)
    
    # Build or load cache
    if args.rebuild_cache or not cache.is_valid(args.csv, args.services_dir):
        cache.build(args.csv, args.services_dir)
        data = cache.load()
    else:
        print("\n[Cache] Valid cache found, loading...")
        data = cache.load()
    
    output_dir = Path(args.output_dir)
    
    # Visualization (with flows)
    if args.visualize:
        create_visualization_with_flows(data, args.window, args.flow_window, output_dir, args.dpi)
    else:
        print("\n[Skip] Visualization (use --visualize to enable)")
    
    # LLM export (with flows)
    if not args.no_llm_export:
        export_for_llm_with_flows(data, args.window, args.flow_window, output_dir)
    else:
        print("\n[Skip] LLM export disabled")
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nCache: {args.cache_dir}/")
    print(f"Output: {args.output_dir}/")
    print(f"\nFlow reconstruction settings:")
    print(f"  - Time window for grouping: {args.flow_window}s")
    print(f"  - Uses CPG call edges + temporal proximity")
    print(f"\nOutputs:")
    print(f"  - timeline.png                     (visualization with flow connections)")
    print(f"  - execution_flows.json             (reconstructed execution flows)")
    print(f"  - llm_input.json                   (LLM data with flow information)")


if __name__ == "__main__":
    main()