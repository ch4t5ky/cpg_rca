#!/usr/bin/env python3
"""
rca_pipeline.py
===============
Complete End-to-End RCA Pipeline

Complete workflow:
1. Retrieve error logs
2. Aggregate error logs  
3. Find method where error is located (using CPG)
4. Step backward through call graph
5. Visualize discovered methods
6. Output graphs for LLM analysis

Memory-efficient: Processes ONE CPG at a time.

Usage:
    python rca_pipeline.py --csv logs.csv --services-dir services/
    python rca_pipeline.py --csv logs.csv --services-dir services/ --max-errors 100

Output:
    output/
    ├── 1_errors.json              # Step 1: All error logs
    ├── 2_aggregated.json          # Step 2: Unique error patterns
    ├── 3_methods.json             # Step 3: Error → Method mappings
    ├── 4_call_paths.json          # Step 4: Backward call traces
    ├── 5_timeline.html            # Step 5: Visualization
    └── 6_llm_input.json           # Step 6: LLM analysis data
"""

import argparse
import gc
import hashlib
import json
import re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

# Import CPG utilities
from log2cpg import (
    process_service_cpg,
    resolve_inter_service_edges,
    fast_match,
    get_method_call_graph,
    MethodIndex,
)


# ===========================================================================
# STEP 1: RETRIEVE ERROR LOGS
# ===========================================================================

ERROR_PATTERNS = {
    'error': ['error', 'exception', 'failed', 'failure', 'fatal'],
    'timeout': ['timeout', 'timed out', 'deadline exceeded'],
    'crash': ['crash', 'panic', 'segfault', 'core dump', 'killed'],
    'connection': ['connection refused', 'connection reset', 'connection closed'],
    'oom': ['out of memory', 'oom', 'memory exhausted'],
}

TRACE_ID_PATTERNS = [
    r'trace[_-]?id[:\s=]+([a-zA-Z0-9-]+)',
    r'request[_-]?id[:\s=]+([a-zA-Z0-9-]+)',
    r'\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b',
]


def detect_error_type(message: str) -> str:
    """Detect error type from message."""
    msg_lower = message.lower()
    for error_type, patterns in ERROR_PATTERNS.items():
        if any(p in msg_lower for p in patterns):
            return error_type
    return 'error'


def is_error_log(message: str) -> bool:
    """Check if log message is an error."""
    msg_lower = message.lower()
    return any(
        pattern in msg_lower 
        for patterns in ERROR_PATTERNS.values() 
        for pattern in patterns
    )


def extract_trace_id(message: str) -> str:
    """Extract trace/request ID from message."""
    for pattern in TRACE_ID_PATTERNS:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def step1_retrieve_errors(csv_path: str, chunk_size: int = 10000) -> List[dict]:
    """
    STEP 1: Retrieve error logs from CSV.
    
    Processes logs in chunks for memory efficiency.
    
    Returns:
        List of error log dictionaries
    """
    print("\n" + "=" * 70)
    print("STEP 1: RETRIEVE ERROR LOGS")
    print("=" * 70)
    
    print(f"\n[1.1] Loading logs: {csv_path}")
    
    all_errors = []
    total_logs = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk.columns = [c.strip() for c in chunk.columns]
        chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
        chunk.dropna(subset=['timestamp'], inplace=True)
        chunk['container_name'] = chunk['container_name'].fillna('unknown').astype(str)
        chunk['message'] = chunk['message'].fillna('').astype(str)
        
        total_logs += len(chunk)
        
        for _, row in chunk.iterrows():
            message = row['message']
            
            if not is_error_log(message):
                continue
            
            all_errors.append({
                'timestamp': float(row['timestamp']),
                'time_str': datetime.fromtimestamp(float(row['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
                'service': row['container_name'],
                'error_type': detect_error_type(message),
                'message': message,
                'trace_id': extract_trace_id(message),
            })
        
        print(f"  Processed {total_logs:,} logs, found {len(all_errors):,} errors...")
    
    print(f"\n[1.2] Retrieval complete")
    print(f"  Total logs: {total_logs:,}")
    print(f"  Total errors: {len(all_errors):,}")
    print(f"  Error rate: {len(all_errors)/total_logs*100:.2f}%")
    
    return all_errors


# ===========================================================================
# STEP 2: AGGREGATE ERROR LOGS
# ===========================================================================

def normalize_message(message: str) -> str:
    """Normalize message by removing variable parts."""
    msg = message.lower()
    msg = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', msg)
    msg = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?', '<TIMESTAMP>', msg)
    msg = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?\b', '<IP>', msg)
    msg = re.sub(r'\b\d{4,}\b', '<NUM>', msg)
    msg = re.sub(r':\s*\d+', ':<NUM>', msg)
    msg = re.sub(r'=\s*\d+', '=<NUM>', msg)
    msg = re.sub(r'"[^"]{20,}"', '"<STRING>"', msg)
    msg = re.sub(r'\s+', ' ', msg).strip()
    return msg


def generate_signature(service: str, error_type: str, normalized_msg: str) -> str:
    """Generate unique signature for error pattern."""
    sig_str = f"{service}|{error_type}|{normalized_msg}"
    return hashlib.md5(sig_str.encode()).hexdigest()[:12]


def step2_aggregate_errors(all_errors: List[dict]) -> Dict[str, dict]:
    """
    STEP 2: Aggregate error logs to unique patterns.
    
    Deduplicates similar errors by normalizing messages.
    
    Returns:
        Dictionary of unique error patterns
    """
    print("\n" + "=" * 70)
    print("STEP 2: AGGREGATE ERROR LOGS")
    print("=" * 70)
    
    error_patterns = defaultdict(lambda: {
        'count': 0,
        'services': set(),
        'first_seen': None,
        'last_seen': None,
        'samples': [],
        'trace_ids': set(),
    })
    
    for error in all_errors:
        normalized = normalize_message(error['message'])
        signature = generate_signature(error['service'], error['error_type'], normalized)
        
        pattern = error_patterns[signature]
        pattern['count'] += 1
        pattern['services'].add(error['service'])
        pattern['error_type'] = error['error_type']
        pattern['normalized_message'] = normalized
        
        if pattern['first_seen'] is None or error['timestamp'] < pattern['first_seen']:
            pattern['first_seen'] = error['timestamp']
        if pattern['last_seen'] is None or error['timestamp'] > pattern['last_seen']:
            pattern['last_seen'] = error['timestamp']
        
        if len(pattern['samples']) < 3:
            pattern['samples'].append(error['message'])
        
        if error['trace_id']:
            pattern['trace_ids'].add(error['trace_id'])
    
    print(f"\n[2.1] Aggregation complete")
    print(f"  Total errors: {len(all_errors):,}")
    print(f"  Unique patterns: {len(error_patterns):,}")
    print(f"  Deduplication: {(1 - len(error_patterns)/len(all_errors)):.1%}")
    
    # Show top patterns
    print(f"\n[2.2] Top error patterns:")
    sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1]['count'], reverse=True)
    for i, (sig, pattern) in enumerate(sorted_patterns[:5], 1):
        pct = pattern['count'] / len(all_errors) * 100
        print(f"  {i}. [{pattern['error_type'].upper()}] {pattern['count']:,} ({pct:.1f}%)")
        print(f"     Services: {', '.join(list(pattern['services'])[:3])}")
    
    return error_patterns


# ===========================================================================
# STEP 3: FIND METHODS WHERE ERRORS ARE LOCATED (using CPG)
# ===========================================================================

def step3_find_error_methods(
    all_errors: List[dict],
    services_dir: str,
    max_errors: int = None
) -> Tuple[Dict[str, MethodIndex], List[dict], List[tuple]]:
    """
    STEP 3: Map errors to source methods using CPG.
    
    Memory-efficient: Processes ONE CPG at a time.
    
    Returns:
        (indexes, error_methods, all_edges)
    """
    print("\n" + "=" * 70)
    print("STEP 3: FIND METHODS WHERE ERRORS ARE LOCATED")
    print("=" * 70)
    
    # Limit if requested
    if max_errors:
        all_errors = all_errors[:max_errors]
        print(f"\n[3.1] Limited to {max_errors} errors for processing")
    
    # Group errors by service
    errors_by_service = defaultdict(list)
    for error in all_errors:
        errors_by_service[error['service']].append(error)
    
    print(f"\n[3.1] Processing CPG (one service at a time)...")
    print(f"  Services with errors: {len(errors_by_service)}")
    
    base_dir = Path(services_dir)
    all_indexes = {}
    all_edges = []
    error_methods = []
    
    # Process each service ONE AT A TIME
    for service in sorted(errors_by_service.keys()):
        cpg_file = base_dir / service / "export.dot"
        
        if not cpg_file.exists():
            print(f"\n  ⚠ Skipping {service}: CPG not found")
            continue
        
        print(f"\n  Processing {service}...")
        print(f"    Errors: {len(errors_by_service[service])}")
        
        # Prepare log rows
        log_rows = [(e['time_str'], e['message']) for e in errors_by_service[service][:2000]]
        
        # Process CPG (loads → processes → FREES memory)
        print(f"    Loading CPG: {cpg_file}")
        _, edges, index = process_service_cpg(
            dot_path=str(cpg_file),
            service_name=service,
            log_rows=log_rows
        )
        
        # Map errors to methods
        for error in errors_by_service[service]:
            result = fast_match(error['message'], index)
            
            if result['matched'] and result['score'] > 0.01:
                error_methods.append({
                    'timestamp': error['timestamp'],
                    'time_str': error['time_str'],
                    'service': service,
                    'error_type': error['error_type'],
                    'message': error['message'],
                    'trace_id': error.get('trace_id'),
                    'method': result['function_name'],
                    'qualified_name': f"{service}::{result['function_name']}",
                    'full_name': result.get('full_name', ''),
                    'line': result.get('line', ''),
                    'score': result['score'],
                })
        
        print(f"    ✓ Mapped {len([m for m in error_methods if m['service'] == service])}/{len(errors_by_service[service])} to methods")
        
        # Store results
        all_indexes[service] = index
        all_edges.extend(edges)
        
        # Force garbage collection
        gc.collect()
        print(f"    ✓ Memory freed")
    
    # Resolve inter-service edges
    print(f"\n[3.2] Resolving cross-service calls...")
    if len(all_indexes) > 1:
        inter_edges = resolve_inter_service_edges(all_indexes)
        all_edges.extend(inter_edges)
        print(f"  ✓ Found {len(inter_edges)} cross-service edges")
    
    print(f"\n[3.3] Method mapping complete")
    print(f"  Mapped: {len(error_methods)}/{len(all_errors)} ({len(error_methods)/len(all_errors):.1%})")
    print(f"  Services processed: {len(all_indexes)}")
    
    return all_indexes, error_methods, all_edges


# ===========================================================================
# STEP 4: STEP BACKWARD THROUGH CALL GRAPH
# ===========================================================================

def step4_backward_trace(
    error_methods: List[dict],
    all_edges: List[tuple],
    max_depth: int = 3
) -> Dict[str, dict]:
    """
    STEP 4: Trace backward through call graph to find root causes.
    
    For each error method, finds:
    - Who called it (backward trace)
    - What it was calling (forward trace)
    - All services involved
    
    Returns:
        Dictionary of call paths keyed by qualified method name
    """
    print("\n" + "=" * 70)
    print("STEP 4: STEP BACKWARD THROUGH CALL GRAPH")
    print("=" * 70)
    
    call_paths = {}
    unique_methods = {}
    
    # Get unique error methods
    for method in error_methods:
        qualified = method['qualified_name']
        if qualified not in unique_methods:
            unique_methods[qualified] = method
    
    print(f"\n[4.1] Building call graphs...")
    print(f"  Unique error methods: {len(unique_methods)}")
    
    for i, (qualified, method) in enumerate(unique_methods.items()):
        service, method_name = qualified.split("::", 1)
        
        try:
            # Get call graph (backward + forward)
            graph = get_method_call_graph(
                method_name=method_name,
                service_name=service,
                all_edges=all_edges,
                max_depth=max_depth
            )
            
            call_paths[qualified] = {
                'error_method': qualified,
                'error_type': method['error_type'],
                'error_message': method['message'],
                'trace_id': method.get('trace_id'),
                'timestamp': method['time_str'],
                
                # Backward trace (who called this)
                'internal_calls': graph['internal_calls'],      # Same service
                
                # Forward trace (what this called)
                'outbound_calls': graph['outbound_calls'],      # To external services
                'external_chain_calls': graph['external_chain_calls'],  # External → external
                
                # Summary
                'services_involved': list(graph['services']),
                'total_reachable': len(graph['all_reachable']),
                'call_depth': max(graph['depth_map'].values()) if graph['depth_map'] else 0,
            }
        except Exception as e:
            print(f"  ⚠ Failed to build graph for {qualified}: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(unique_methods)} methods...")
    
    print(f"\n[4.2] Backward trace complete")
    print(f"  Call paths generated: {len(call_paths)}")
    
    # Show summary
    total_internal = sum(len(p['internal_calls']) for p in call_paths.values())
    total_outbound = sum(len(p['outbound_calls']) for p in call_paths.values())
    
    print(f"  Internal calls found: {total_internal}")
    print(f"  Cross-service calls found: {total_outbound}")
    
    return call_paths


# ===========================================================================
# STEP 5: VISUALIZE DISCOVERED METHODS
# ===========================================================================

def step5_visualize(
    all_errors: List[dict],
    error_methods: List[dict],
    call_paths: Dict[str, dict],
    output_path: str,
    dpi: int = 300
) -> None:
    """
    STEP 5: Create timeline visualization showing reconstructed flows.
    
    Style matches original pipeline.py:
    - Service lanes (horizontal)
    - Method circles with short labels (fe-1, ad-2, etc.)
    - Arrows showing call flows from Step 4
    - Color-coded by service and error type
    
    Args:
        all_errors: All error logs
        error_methods: Methods mapped from errors  
        call_paths: Call graph traces (from Step 4)
        output_path: Path to save figure (without extension)
        dpi: DPI for output (300 for publication quality)
    """
    print("\n" + "=" * 70)
    print("STEP 5: VISUALIZE RECONSTRUCTED FLOWS")
    print("=" * 70)
    
    print(f"\n[5.1] Generating timeline with call flows...")
    
    if not error_methods:
        print("  No methods to visualize")
        return
    
    # Get time range
    timestamps = [m['timestamp'] for m in error_methods]
    start_time = min(timestamps)
    end_time = max(timestamps)
    time_range = max(end_time - start_time, 1)
    
    # Get services in order
    services = []
    seen = set()
    for m in sorted(error_methods, key=lambda x: x['timestamp']):
        if m['service'] not in seen:
            services.append(m['service'])
            seen.add(m['service'])
    
    service_y = {s: (len(services) - 1 - i) * 2.5 for i, s in enumerate(services)}
    
    # Service colors
    colors = {
        'frontend': '#4285F4', 'adservice': '#EA4335', 'cartservice': '#FBBC04',
        'checkoutservice': '#34A853', 'currencyservice': '#9334E6', 'emailservice': '#FF6D00',
        'paymentservice': '#00BFA5', 'recommendationservice': '#D32F2F',
        'shippingservice': '#7B1FA2', 'redis': '#DC382D',
    }
    
    # Short names
    service_short = {
        'frontend': 'fe', 'adservice': 'ad', 'cartservice': 'cart',
        'checkoutservice': 'chk', 'currencyservice': 'cur', 'emailservice': 'email',
        'paymentservice': 'pay', 'recommendationservice': 'rec',
        'shippingservice': 'ship', 'redis': 'rds',
    }
    
    # Assign IDs to unique methods
    method_to_id = {}
    method_counter = defaultdict(int)
    method_details = {}
    
    for evt in error_methods:
        qn = evt['qualified_name']
        
        if qn not in method_to_id:
            service = evt['service']
            short = service_short.get(service, service[:4])
            method_counter[service] += 1
            number = method_counter[service]
            
            label = f"{short}-{number}"
            method_to_id[qn] = label
            
            method_details[label] = {
                'service': service,
                'method': evt['method'],
                'qualified_name': qn,
                'is_error': evt.get('error_type') is not None,
                'error_type': evt.get('error_type', ''),
            }
    
    # Map events to labels
    event_labels = {i: method_to_id[evt['qualified_name']] for i, evt in enumerate(error_methods)}
    
    # Create figure
    fig_width = max(30, min(70, time_range * 5 + 10))
    fig_height = max(12, min(30, len(services) * 3.5 + 4))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_xlim(-1, time_range + 1)
    ax.set_ylim(-2, len(services) * 2.5 + 1.5)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Draw service lanes
    for service in services:
        y = service_y[service]
        color = colors.get(service, '#78909C')
        
        # Lane background
        ax.fill_betweenx([y - 1.0, y + 1.0], -0.5, time_range + 0.5,
                         color=color, alpha=0.05, zorder=0)
        ax.plot([0, time_range], [y, y], color=color, lw=2, alpha=0.3, zorder=1)
        
        # Service label
        ax.text(-1.0, y, service, ha='right', va='center', fontsize=12, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
               edgecolor=color, linewidth=2))
    
    # Position events with stacking
    bucket_size = 0.15
    service_buckets = defaultdict(lambda: defaultdict(list))
    
    for i, evt in enumerate(error_methods):
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
    
    # Build edge map from call_paths (Step 4 results)
    edge_map = defaultdict(set)
    
    for qualified, path in call_paths.items():
        # Internal calls (backward trace)
        for caller, callee in path['internal_calls']:
            edge_map[caller].add(callee)
        
        # Outbound calls (forward trace)
        for caller, callee in path['outbound_calls']:
            edge_map[caller].add(callee)
        
        # External chains
        for caller, callee in path['external_chain_calls']:
            edge_map[caller].add(callee)
    
    # Draw arrows showing reconstructed flows
    drawn = set()
    for i, evt in enumerate(error_methods):
        if i not in positions:
            continue
        
        caller_qn = evt['qualified_name']
        if caller_qn not in edge_map:
            continue
        
        caller_x, caller_y = positions[i]
        
        # Find matching callees in nearby time window
        for j in range(i + 1, min(i + 20, len(error_methods))):
            if j not in positions:
                continue
            
            evt2 = error_methods[j]
            if evt2['timestamp'] > evt['timestamp'] + 2:
                break
            
            if evt2['qualified_name'] not in edge_map[caller_qn]:
                continue
            
            if (i, j) in drawn:
                continue
            drawn.add((i, j))
            
            callee_x, callee_y = positions[j]
            is_cross = (evt['service'] != evt2['service'])
            
            # Arrow styling
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
    
    # Draw method circles with labels
    for i, evt in enumerate(error_methods):
        if i not in positions:
            continue
        
        x, y = positions[i]
        color = colors.get(evt['service'], '#78909C')
        is_error = evt.get('error_type') is not None
        
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
    ax.text(time_range / 2, title_y, 'Service Timeline with Method Calls',
           ha='center', fontsize=16, fontweight='bold')
    
    start_str = datetime.fromtimestamp(start_time).strftime('%H:%M:%S')
    ax.text(time_range / 2, title_y - 0.4,
           f'{start_str} | {len(error_methods)} calls | {len(services)} services',
           ha='center', fontsize=10, color='#666')
    
    # Legend
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444',
               markersize=10, markeredgewidth=2.5, markeredgecolor='#FF4444', label='Error'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=8, markeredgewidth=1.8, markeredgecolor='#4285F4', label='Method'),
        Line2D([0], [0], color='#E53935', linewidth=2.5, label='Cross-Service Call'),
    ]
    
    ax.legend(handles=legend, loc='lower right', fontsize=10, framealpha=0.95, ncol=3)
    
    plt.tight_layout()
    
    # Save
    png_path = f"{output_path}.png"
    pdf_path = f"{output_path}.pdf"
    
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Timeline with reconstructed flows saved:")
    print(f"    PNG (raster): {png_path}")
    print(f"    PDF (vector): {pdf_path}")
    print(f"  Methods shown: {len(positions)}")
    print(f"  Call flows drawn: {len(drawn)}")
    print(f"  Services: {len(services)}")
    print(f"  DPI: {dpi} (publication quality)")


# ===========================================================================
# STEP 6: OUTPUT GRAPHS FOR LLM ANALYSIS
# ===========================================================================

def step6_llm_output(
    all_errors: List[dict],
    error_patterns: Dict[str, dict],
    error_methods: List[dict],
    call_paths: Dict[str, dict]
) -> dict:
    """
    STEP 6: Prepare structured output for LLM analysis.
    
    Returns:
        Dictionary ready for LLM consumption
    """
    print("\n" + "=" * 70)
    print("STEP 6: OUTPUT GRAPHS FOR LLM ANALYSIS")
    print("=" * 70)
    
    print(f"\n[6.1] Building LLM context...")
    
    # Convert patterns
    patterns_list = []
    for sig, pattern in sorted(error_patterns.items(), key=lambda x: x[1]['count'], reverse=True):
        patterns_list.append({
            'signature': sig,
            'error_type': pattern['error_type'],
            'count': pattern['count'],
            'percentage': pattern['count'] / len(all_errors) * 100,
            'services': list(pattern['services']),
            'normalized_message': pattern['normalized_message'],
            'first_seen': datetime.fromtimestamp(pattern['first_seen']).isoformat(),
            'last_seen': datetime.fromtimestamp(pattern['last_seen']).isoformat(),
            'sample_messages': pattern['samples'],
        })
    
    # Build LLM context
    llm_context = {
        'workflow_steps': {
            'step1': 'Retrieved error logs',
            'step2': 'Aggregated to unique patterns',
            'step3': 'Mapped to source methods using CPG',
            'step4': 'Traced backward through call graph',
            'step5': 'Generated visualization',
            'step6': 'Prepared this output'
        },
        
        'summary': {
            'total_errors': len(all_errors),
            'unique_patterns': len(error_patterns),
            'deduplication_ratio': 1 - len(error_patterns) / len(all_errors),
            'mapped_to_methods': len(error_methods),
            'mapping_rate': len(error_methods) / len(all_errors),
            'call_paths_traced': len(call_paths),
            'services_affected': len(set(e['service'] for e in all_errors)),
        },
        
        'error_patterns': patterns_list[:10],  # Top 10
        
        'error_methods': {
            'total': len(error_methods),
            'by_service': dict(Counter(m['service'] for m in error_methods).most_common()),
            'by_error_type': dict(Counter(m['error_type'] for m in error_methods).most_common()),
            'top_methods': [
                {'method': method, 'count': count}
                for method, count in Counter(m['qualified_name'] for m in error_methods).most_common(10)
            ],
        },
        
        'call_graph_analysis': {
            'total_paths': len(call_paths),
            'root_causes': [
                {
                    'method': qualified,
                    'error_type': path['error_type'],
                    'reason': 'No incoming calls (potential root cause)',
                    'outbound_calls': len(path['outbound_calls']),
                    'services_involved': path['services_involved'],
                }
                for qualified, path in call_paths.items()
                if not path['internal_calls']
            ][:10],
            'complex_paths': [
                {
                    'method': qualified,
                    'error_type': path['error_type'],
                    'total_reachable': path['total_reachable'],
                    'call_depth': path['call_depth'],
                    'services_involved': path['services_involved'],
                }
                for qualified, path in sorted(
                    call_paths.items(),
                    key=lambda x: x[1]['total_reachable'],
                    reverse=True
                )[:10]
            ],
        },
        
        'recommendations': []
    }
    
    # Auto-generate recommendations
    if patterns_list and patterns_list[0]['count'] > len(all_errors) * 0.5:
        llm_context['recommendations'].append({
            'priority': 'CRITICAL',
            'finding': f"One error pattern accounts for {patterns_list[0]['percentage']:.0f}% of all errors",
            'pattern': patterns_list[0]['normalized_message'],
            'action': 'Focus investigation on this specific error pattern'
        })
    
    timeout_count = sum(1 for e in all_errors if e['error_type'] == 'timeout')
    if timeout_count > len(all_errors) * 0.3:
        llm_context['recommendations'].append({
            'priority': 'HIGH',
            'finding': f"Timeouts represent {timeout_count/len(all_errors):.0%} of errors",
            'action': 'Check service response times, connection pools, and network latency'
        })
    
    root_causes = llm_context['call_graph_analysis']['root_causes']
    if len(root_causes) <= 3:
        llm_context['recommendations'].append({
            'priority': 'HIGH',
            'finding': f"Only {len(root_causes)} root cause methods identified",
            'methods': [rc['method'] for rc in root_causes],
            'action': 'Investigate these methods - they have no incoming calls'
        })
    
    print(f"\n[6.2] LLM context complete")
    print(f"  Total errors: {llm_context['summary']['total_errors']}")
    print(f"  Unique patterns: {llm_context['summary']['unique_patterns']}")
    print(f"  Mapped to methods: {llm_context['summary']['mapped_to_methods']}")
    print(f"  Root causes found: {len(root_causes)}")
    print(f"  Recommendations: {len(llm_context['recommendations'])}")
    
    return llm_context


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete End-to-End RCA Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete Workflow:
  1. Retrieve error logs
  2. Aggregate error logs  
  3. Find method where error is located (using CPG)
  4. Step backward through call graph
  5. Visualize discovered methods (matplotlib - publication quality)
  6. Output graphs for LLM analysis

Usage:
  python rca_pipeline.py --csv logs.csv --services-dir services/
  python rca_pipeline.py --csv logs.csv --services-dir services/ --max-errors 100

Output Files:
  output/1_errors.json           - All error logs (Step 1)
  output/2_aggregated.json       - Unique patterns (Step 2)
  output/3_methods.json          - Method mappings (Step 3)
  output/4_call_paths.json       - Call traces (Step 4)
  output/5_timeline.png          - Matplotlib figure 300 DPI (Step 5)
  output/5_timeline.pdf          - Vector format for LaTeX (Step 5)
  output/6_llm_input.json        - LLM analysis (Step 6)
        """
    )
    
    parser.add_argument('--csv', required=True, help='Path to CSV log file')
    parser.add_argument('--services-dir', required=True, help='Path to CPG services directory')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--max-errors', type=int, help='Limit number of errors to process')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Log processing chunk size')
    parser.add_argument('--max-depth', type=int, default=3, help='Max call graph depth')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("COMPLETE END-TO-END RCA PIPELINE")
    print("=" * 70)
    print("\nWorkflow:")
    print("  1. Retrieve error logs")
    print("  2. Aggregate error logs")
    print("  3. Find methods where errors are located (CPG)")
    print("  4. Step backward through call graph")
    print("  5. Visualize discovered methods")
    print("  6. Output graphs for LLM analysis")
    
    # STEP 1: Retrieve error logs
    all_errors = step1_retrieve_errors(args.csv, args.chunk_size)
    
    with open(output_dir / "1_errors.json", 'w') as f:
        json.dump({'total': len(all_errors), 'errors': all_errors}, f, indent=2)
    print(f"\n  ✓ Saved: {output_dir / '1_errors.json'}")
    
    # STEP 2: Aggregate errors
    error_patterns = step2_aggregate_errors(all_errors)
    
    # Convert for JSON
    patterns_json = {
        sig: {
            **{k: v for k, v in pattern.items() if k not in ['services', 'trace_ids']},
            'services': list(pattern['services']),
            'trace_ids': list(pattern['trace_ids']),
        }
        for sig, pattern in error_patterns.items()
    }
    
    with open(output_dir / "2_aggregated.json", 'w') as f:
        json.dump({'total': len(error_patterns), 'patterns': patterns_json}, f, indent=2)
    print(f"\n  ✓ Saved: {output_dir / '2_aggregated.json'}")
    
    # STEP 3: Find error methods
    indexes, error_methods, all_edges = step3_find_error_methods(
        all_errors,
        args.services_dir,
        args.max_errors
    )
    
    with open(output_dir / "3_methods.json", 'w') as f:
        json.dump({'total': len(error_methods), 'methods': error_methods}, f, indent=2)
    print(f"\n  ✓ Saved: {output_dir / '3_methods.json'}")
    
    # STEP 4: Backward trace
    call_paths = step4_backward_trace(error_methods, all_edges, args.max_depth)
    
    with open(output_dir / "4_call_paths.json", 'w') as f:
        json.dump({'total': len(call_paths), 'paths': call_paths}, f, indent=2)
    print(f"\n  ✓ Saved: {output_dir / '4_call_paths.json'}")
    
    # STEP 5: Visualize
    step5_visualize(all_errors, error_methods, call_paths, 
                   str(output_dir / "5_timeline"), dpi=300)
    print(f"\n  ✓ Saved: {output_dir / '5_timeline.png'} and .pdf")
    
    # STEP 6: LLM output
    llm_context = step6_llm_output(all_errors, error_patterns, error_methods, call_paths)
    
    with open(output_dir / "6_llm_input.json", 'w') as f:
        json.dump(llm_context, f, indent=2)
    print(f"\n  ✓ Saved: {output_dir / '6_llm_input.json'}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults in: {output_dir}/")
    print(f"  1_errors.json       - {len(all_errors)} error logs")
    print(f"  2_aggregated.json   - {len(error_patterns)} unique patterns")
    print(f"  3_methods.json      - {len(error_methods)} method mappings")
    print(f"  4_call_paths.json   - {len(call_paths)} call traces")
    print(f"  5_timeline.png      - Matplotlib visualization (300 DPI)")
    print(f"  5_timeline.pdf      - Vector format for papers")
    print(f"  6_llm_input.json    - LLM analysis ready")
    print(f"\nNext steps:")
    print(f"  1. Use {output_dir / '5_timeline.pdf'} in your research paper")
    print(f"  2. Feed {output_dir / '6_llm_input.json'} to Claude/GPT")
    print(f"  3. Review root causes in 4_call_paths.json")


if __name__ == "__main__":
    main()