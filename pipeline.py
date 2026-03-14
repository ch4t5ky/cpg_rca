#!/usr/bin/env python3
"""
pipeline.py
===========
Main CPG-based RCA Pipeline

This is the MAIN script that:
1. Preprocesses logs and CPG files
2. Builds UNIFIED cache (shared across all tools)
3. Generates timeline visualization
4. Exports data for LLM RCA

Cache is stored in 'cache/' directory and reused by:
- This visualization tool
- LLM RCA tools
- Any other analysis scripts

Usage:
    # Build cache and visualize
    python pipeline.py --csv logs.csv --services-dir services
    
    # Force rebuild cache
    python pipeline.py --csv logs.csv --services-dir services --rebuild-cache
    
    # Use existing cache, just regenerate visualization
    python pipeline.py --csv logs.csv --services-dir services --viz-only

Output:
    cache/
    ├── metadata.json              # Cache info
    ├── cpg_indexes.pkl            # CPG data (reusable)
    ├── call_edges.pkl             # Call relationships
    ├── method_calls.pkl           # All log->method mappings
    ├── lifecycle_events.pkl       # Service start/stop/crash
    └── raw_logs.pkl               # Original logs
    
    output/
    ├── timeline.png               # Visualization
    ├── timeline_methods.json      # Method details
    └── llm_input.json             # Data for LLM RCA
"""

import argparse
import pickle
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from log2cpg2 import process_service_cpg, resolve_inter_service_edges, fast_match


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
                        'score': result['score']
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
        
        print(f"  ✓ Cache built successfully")
        print(f"\n  Cache contents:")
        print(f"    - CPG indexes: {len(indexes)} services")
        print(f"    - Call edges: {len(all_edges)}")
        print(f"    - Method calls: {len(method_calls)}")
        print(f"    - Lifecycle events: {len(lifecycle_events)}")
    
    def load(self) -> Dict:
        """Load cache data."""
        print(f"\n[Cache] Loading from: {self.cache_dir}")
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
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
        
        print(f"  ✓ Loaded {len(method_calls)} method calls, {len(edges)} edges")
        
        return {
            'metadata': metadata,
            'indexes': indexes,
            'edges': edges,
            'method_calls': method_calls,
            'lifecycle': lifecycle,
            'logs': logs
        }


def find_error_window(method_calls: List[dict], window: int = 10):
    """Find time window around first error."""
    errors = [m for m in method_calls if m['is_error']]
    
    if not errors:
        # No errors, use all
        if not method_calls:
            return [], 0, 20
        first_time = method_calls[0]['timestamp']
        last_time = method_calls[-1]['timestamp']
        return method_calls, first_time, last_time
    
    first_error_time = min(e['timestamp'] for e in errors)
    start_time = first_error_time - window
    end_time = first_error_time + window
    
    window_events = [m for m in method_calls 
                     if start_time <= m['timestamp'] <= end_time]
    
    return window_events, first_error_time, end_time - start_time


def create_visualization(data: Dict, window: int, output_dir: Path, dpi: int):
    """Create timeline visualization."""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)
    
    method_calls = data['method_calls']
    edges = data['edges']
    lifecycle = data['lifecycle']
    
    # Get window
    events, first_error, time_range = find_error_window(method_calls, window)
    start_time = first_error - window
    
    if not events:
        print("  No events to visualize")
        return
    
    print(f"\n  Events in window: {len(events)}")
    print(f"  Time range: {time_range:.1f}s")
    
    # Get services
    services = []
    seen = set()
    for e in sorted(events, key=lambda x: x['timestamp']):
        if e['service'] not in seen:
            services.append(e['service'])
            seen.add(e['service'])
    
    service_y = {s: (len(services) - 1 - i) * 2.5 for i, s in enumerate(services)}
    
    # Service short names
    service_short = {
        'frontend': 'fe', 'adservice': 'ad', 'cartservice': 'cart',
        'checkoutservice': 'chk', 'currencyservice': 'cur', 'emailservice': 'email',
        'paymentservice': 'pay', 'recommendationservice': 'rec',
        'shippingservice': 'ship', 'redis': 'rds',
    }
    
    # Assign IDs to UNIQUE methods (reuse same ID for same method)
    method_to_id = {}  # qualified_name -> label
    method_counter = defaultdict(int)
    method_details = {}
    
    # First pass: assign IDs to unique methods
    for evt in events:
        qn = evt['qualified_name']
        
        if qn not in method_to_id:
            service = evt['service']
            short = service_short.get(service, service[:4])
            method_counter[service] += 1
            number = method_counter[service]
            
            label = f"{short}-{number}"
            method_to_id[qn] = label
            
            # Store details (use first occurrence)
            method_details[label] = {
                'service': service,
                'method': evt['method'],
                'qualified_name': qn,
                'first_timestamp': evt['timestamp'],
                'first_time_str': datetime.fromtimestamp(evt['timestamp']).strftime('%H:%M:%S.%f')[:-3],
                'is_error': evt['is_error'],
                'message': evt['message'],
                'score': evt.get('score', 0.0),
                'total_calls': 0,
                'all_timestamps': []
            }
    
    # Second pass: count calls and collect timestamps
    for evt in events:
        qn = evt['qualified_name']
        label = method_to_id[qn]
        method_details[label]['total_calls'] += 1
        method_details[label]['all_timestamps'].append(
            datetime.fromtimestamp(evt['timestamp']).strftime('%H:%M:%S.%f')[:-3]
        )
    
    # Map each event to its method ID
    event_labels = {}
    for i, evt in enumerate(events):
        event_labels[i] = method_to_id[evt['qualified_name']]
    
    # Figure
    fig_width = max(30, min(70, time_range * 5 + 10))
    fig_height = max(12, min(30, len(services) * 3.5 + 4))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_xlim(-1, time_range + 1)
    ax.set_ylim(-2, len(services) * 2.5 + 1.5)
    ax.axis('off')
    
    # Colors
    colors = {
        'frontend': '#4285F4', 'adservice': '#EA4335', 'cartservice': '#FBBC04',
        'checkoutservice': '#34A853', 'currencyservice': '#9334E6', 'emailservice': '#FF6D00',
        'paymentservice': '#00BFA5', 'recommendationservice': '#D32F2F',
        'shippingservice': '#7B1FA2', 'redis': '#DC382D',
    }
    
    # Draw lanes
    for service in services:
        y = service_y[service]
        color = colors.get(service, '#78909C')
        
        ax.fill_betweenx([y - 1.0, y + 1.0], -0.5, time_range + 0.5,
                         color=color, alpha=0.05, zorder=0)
        ax.plot([0, time_range], [y, y], color=color, lw=2, alpha=0.3, zorder=1)
        
        ax.text(-1.0, y, service, ha='right', va='center', fontsize=12, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
               edgecolor=color, linewidth=2))
    
    # Position events with stacking
    bucket_size = 0.15
    service_buckets = defaultdict(lambda: defaultdict(list))
    
    for i, evt in enumerate(events):
        rel_time = evt['timestamp'] - start_time
        bucket = int(rel_time / bucket_size)
        service_buckets[evt['service']][bucket].append((i, evt))
    
    positions = {}
    
    for service in services:
        y_base = service_y[service]
        
        for bucket, evts in service_buckets[service].items():
            n = len(evts)
            
            for idx, (i, evt) in enumerate(sorted(evts, key=lambda x: x[1]['timestamp'])):
                rel_time = evt['timestamp'] - start_time
                
                if n == 1:
                    y_pos = y_base
                elif n == 2:
                    y_pos = y_base + (0.25 if idx == 0 else -0.25)
                else:
                    spread = min(0.7, (n - 1) * 0.25)
                    y_pos = y_base + spread/2 - idx * (spread / (n - 1))
                
                positions[i] = (rel_time, y_pos)
    
    # Build edge map
    edge_map = defaultdict(set)
    for caller, callee, kind in edges:
        if kind == "CALL":
            edge_map[caller].add(callee)
    
    # Draw arrows
    drawn = set()
    for i, evt in enumerate(events):
        if i not in positions:
            continue
        
        caller_qn = evt['qualified_name']
        if caller_qn not in edge_map:
            continue
        
        caller_x, caller_y = positions[i]
        
        for j in range(i + 1, min(i + 20, len(events))):
            if j not in positions:
                continue
            
            evt2 = events[j]
            if evt2['timestamp'] > evt['timestamp'] + 2:
                break
            
            if evt2['qualified_name'] not in edge_map[caller_qn]:
                continue
            
            if (i, j) in drawn:
                continue
            drawn.add((i, j))
            
            callee_x, callee_y = positions[j]
            is_cross = (evt['service'] != evt2['service'])
            
            color = '#E53935' if is_cross else colors.get(evt['service'], '#78909C')
            lw = 2.5 if is_cross else 1.5
            alpha = 0.7 if is_cross else 0.3
            
            dx = callee_x - caller_x
            dy = callee_y - caller_y
            dist = (dx**2 + dy**2)**0.5
            
            if dist < 0.1:
                continue
            
            rad = 0.2 if dy > 0 else -0.2
            arrow = FancyArrowPatch(
                (caller_x, caller_y), (callee_x, callee_y),
                arrowstyle='-|>', connectionstyle=f"arc3,rad={rad}",
                color=color, linewidth=lw, alpha=alpha,
                mutation_scale=10, zorder=8
            )
            ax.add_patch(arrow)
    
    # Draw lifecycle events
    for lc in lifecycle:
        if lc['service'] not in service_y:
            continue
        
        rel_time = lc['timestamp'] - start_time
        if rel_time < 0 or rel_time > time_range:
            continue
        
        y = service_y[lc['service']]
        
        ax.scatter(rel_time, y, color=color,
                  edgecolors='white', linewidths=2.5, zorder=30, alpha=0.9)
        
        ax.text(rel_time, y + 0.7, lc['type'].upper(),
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.3',
               facecolor='white', edgecolor=color, linewidth=2), zorder=31)
    
    # Draw circles with labels
    for i, evt in enumerate(events):
        if i not in positions:
            continue
        
        x, y = positions[i]
        color = colors.get(evt['service'], '#78909C')
        is_error = evt['is_error']
        
        radius = 0.14 if not is_error else 0.18
        circle_color = '#FF4444' if is_error else color
        
        ax.add_patch(Circle((x + 0.015, y - 0.015), radius,
                           facecolor='#00000015', edgecolor='none', zorder=18))
        
        ax.add_patch(Circle((x, y), radius,
                           facecolor='white' if not is_error else circle_color,
                           edgecolor=circle_color,
                           linewidth=2.5 if is_error else 1.8,
                           zorder=20))
        
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
    ax.text(time_range / 2, title_y, 'Service Timeline with Method Calls',
           ha='center', fontsize=16, fontweight='bold')
    
    start_str = datetime.fromtimestamp(start_time).strftime('%H:%M:%S')
    ax.text(time_range / 2, title_y - 0.4,
           f'{start_str} | {len(events)} calls | {len(services)} services',
           ha='center', fontsize=10, color='#666')
    
    # Legend
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444',
               markersize=10, markeredgewidth=2.5, markeredgecolor='#FF4444', label='Error'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=8, markeredgewidth=1.8, markeredgecolor='#4285F4', label='Method'),
        Line2D([0], [0], color='#E53935', linewidth=2.5, label='Cross-Service'),
    ]
    
    ax.legend(handles=legend, loc='lower right', fontsize=10, framealpha=0.95, ncol=3)
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(exist_ok=True)
    viz_file = output_dir / "timeline.png"
    fig.savefig(viz_file, bbox_inches='tight', dpi=dpi)
    plt.close()
    
    print(f"\n  ✓ Visualization: {viz_file}")
    
    # Save method details JSON
    json_file = output_dir / "timeline_methods.json"
    with open(json_file, 'w') as f:
        json.dump(method_details, f, indent=2)
    
    print(f"  ✓ Method details: {json_file}")
    
    return method_details


def export_for_llm(data: Dict, window: int, output_dir: Path):
    """Export data for LLM RCA."""
    print("\n" + "=" * 70)
    print("EXPORTING FOR LLM RCA")
    print("=" * 70)
    
    method_calls = data['method_calls']
    edges = data['edges']
    lifecycle = data['lifecycle']
    
    # Get error window
    events, first_error, time_range = find_error_window(method_calls, window)
    
    # Build data for LLM
    llm_data = {
        'summary': {
            'total_method_calls': len(events),
            'error_count': sum(1 for e in events if e['is_error']),
            'services': list(set(e['service'] for e in events)),
            'time_range_seconds': time_range,
            'first_error_time': datetime.fromtimestamp(first_error).isoformat()
        },
        'errors': [
            {
                'service': e['service'],
                'method': e['method'],
                'timestamp': datetime.fromtimestamp(e['timestamp']).isoformat(),
                'message': e['message']
            }
            for e in events if e['is_error']
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
        'method_sequence': [
            {
                'service': e['service'],
                'method': e['method'],
                'timestamp': datetime.fromtimestamp(e['timestamp']).isoformat(),
                'is_error': e['is_error']
            }
            for e in events
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
    print(f"  ✓ Lifecycle events: {len(llm_data['lifecycle_events'])}")
    print(f"  ✓ Method calls: {len(llm_data['method_sequence'])}")


def main():
    parser = argparse.ArgumentParser(
        description="CPG-based RCA Pipeline - Main Entry Point"
    )
    
    parser.add_argument('--csv', required=True, help='Path to logs CSV')
    parser.add_argument('--services-dir', required=True, help='Path to CPG directory')
    parser.add_argument('--window', type=int, default=10, help='Time window around error (seconds)')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory')
    parser.add_argument('--dpi', type=int, default=150, help='Visualization DPI')
    
    # Output control (LLM export ON by default, visualization OFF by default)
    parser.add_argument('--visualize', action='store_true', help='Generate timeline visualization (default: False)')
    parser.add_argument('--no-llm-export', action='store_true', help='Disable LLM export (default: enabled)')
    
    # Cache control
    parser.add_argument('--rebuild-cache', action='store_true', help='Force rebuild cache')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CPG-BASED RCA PIPELINE")
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
    
    # Visualization (optional, default False)
    if args.visualize:
        create_visualization(data, args.window, output_dir, args.dpi)
    else:
        print("\n[Skip] Visualization (use --visualize to enable)")
    
    # LLM export (default True)
    if not args.no_llm_export:
        export_for_llm(data, args.window, output_dir)
    else:
        print("\n[Skip] LLM export disabled")
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nCache: {args.cache_dir}/")
    print(f"Output: {args.output_dir}/")
    print(f"\nUsage:")
    print(f"  # With visualization:")
    print(f"  python pipeline.py --csv {args.csv} --services-dir {args.services_dir} --visualize")
    print(f"  # LLM RCA:")
    print(f"  python llm_rca.py --cache-dir {args.cache_dir}")


if __name__ == "__main__":
    main()