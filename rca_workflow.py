#!/usr/bin/env python3
"""
iterative_rca.py
================
Iterative Root Cause Analysis System with LLM Integration

Complete workflow:
1. Retrieve error logs
2. Aggregate error logs  
3. Find method where error is located (using CPG)
4. Step forward/backward through call graph
5. Visualize discovered methods
6. Output graphs for LLM analysis
7. Query LLM for RCA if not found manually

Usage:
    # Basic RCA
    python iterative_rca.py --csv logs.csv --services-dir services
    
    # With LLM integration
    python iterative_rca.py --csv logs.csv --services-dir services --use-llm
    
    # Custom steps
    python iterative_rca.py --csv logs.csv --services-dir services --max-steps 5
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# Import modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from log2cpg import (
    MethodIndex,
    process_service_cpg,
    resolve_inter_service_edges,
    fast_match,
)

# Try to import find_errors and aggregate_errors
try:
    from find_errors import find_errors_in_logs
    HAS_FIND_ERRORS = True
except:
    HAS_FIND_ERRORS = False

try:
    from aggregate_errors import aggregate_errors
    HAS_AGGREGATE = True
except:
    HAS_AGGREGATE = False


# ===========================================================================
# STEP 1: Retrieve Error Logs
# ===========================================================================

def step1_retrieve_error_logs(csv_path: str) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Step 1: Retrieve error logs from CSV.
    
    Returns: (all_logs_df, error_dicts)
    """
    print("\n" + "=" * 70)
    print("STEP 1: RETRIEVE ERROR LOGS")
    print("=" * 70)
    
    # Load all logs
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df["container_name"] = df["container_name"].fillna("unknown").astype(str).str.strip()
    df["message"] = df["message"].fillna("").astype(str)
    
    print(f"✓ Loaded {len(df)} total logs")
    
    # Find errors
    if HAS_FIND_ERRORS:
        print(f"✓ Using find_errors module...")
        errors, summary = find_errors_in_logs(csv_path)
        error_dicts = [e.to_dict() for e in errors]
        print(f"✓ Found {len(error_dicts)} errors")
    else:
        print(f"✓ Using basic pattern matching...")
        # Simple pattern matching
        error_mask = df["message"].str.contains("error|fail|exception", case=False, regex=True)
        error_df = df[error_mask]
        error_dicts = error_df.to_dict('records')
        print(f"✓ Found {len(error_dicts)} errors")
    
    return df, error_dicts


# ===========================================================================
# STEP 2: Aggregate Error Logs
# ===========================================================================

def step2_aggregate_errors(error_dicts: List[dict]) -> List[dict]:
    """
    Step 2: Aggregate error logs to find unique patterns.
    
    Returns: List of unique error patterns
    """
    print("\n" + "=" * 70)
    print("STEP 2: AGGREGATE ERROR LOGS")
    print("=" * 70)
    
    if HAS_AGGREGATE:
        print(f"✓ Using aggregate_errors module...")
        patterns_dict, stats = aggregate_errors(error_dicts, strategy='signature')
        patterns = [p.to_dict() for p in patterns_dict.values()]
        print(f"✓ Aggregated to {len(patterns)} unique patterns")
        print(f"  Deduplication ratio: {stats.deduplication_ratio:.1%}")
    else:
        print(f"✓ Using simple aggregation...")
        # Simple aggregation by service + message
        pattern_map = {}
        for error in error_dicts:
            key = f"{error.get('service', 'unknown')}::{error.get('message', '')[:50]}"
            if key not in pattern_map:
                pattern_map[key] = {
                    'pattern_id': f"pattern_{len(pattern_map)}",
                    'service': error.get('service', 'unknown'),
                    'message_template': error.get('message', ''),
                    'count': 0,
                    'representative_message': error.get('message', ''),
                }
            pattern_map[key]['count'] += 1
        
        patterns = list(pattern_map.values())
        print(f"✓ Aggregated to {len(patterns)} unique patterns")
    
    # Show top patterns
    sorted_patterns = sorted(patterns, key=lambda p: p.get('count', 0), reverse=True)
    for i, p in enumerate(sorted_patterns[:5], 1):
        print(f"  {i}. {p.get('service', 'unknown')}: {p.get('message_template', '')[:60]}... ({p.get('count', 0)}×)")
    
    return patterns


# ===========================================================================
# STEP 3: Find Methods Using CPG
# ===========================================================================

def step3_find_error_methods(
    patterns: List[dict],
    df: pd.DataFrame,
    services_dir: str
) -> Tuple[Dict[str, MethodIndex], List[Tuple[str, str, str]], Dict[str, dict]]:
    """
    Step 3: Find methods where errors are located using CPG.
    
    Returns: (indexes, all_call_edges, pattern_methods)
    """
    print("\n" + "=" * 70)
    print("STEP 3: FIND ERROR METHODS USING CPG")
    print("=" * 70)
    
    # Get services involved
    services = set(p.get('service', 'unknown') for p in patterns)
    print(f"✓ Services with errors: {', '.join(sorted(services))}")
    
    # Load CPG indexes
    print(f"\n[CPG] Loading indexes...")
    indexes = {}
    all_edges = []
    
    base_dir = Path(services_dir)
    
    for service in sorted(services):
        dot_file = base_dir / service / "export.dot"
        
        if not dot_file.exists():
            print(f"  ⚠ {service}: No CPG file")
            continue
        
        # Get logs for this service
        service_df = df[df["container_name"] == service]
        log_rows = [(
            datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            row["message"]
        ) for _, row in service_df.iterrows()]
        
        # Process CPG
        activity, edges, index = process_service_cpg(
            str(dot_file),
            service,
            log_rows[:1000]  # Limit for performance
        )
        
        indexes[service] = index
        all_edges.extend(edges)
    
    # Resolve inter-service edges
    if len(indexes) > 1:
        print(f"\n[CPG] Resolving inter-service edges...")
        inter_edges = resolve_inter_service_edges(indexes)
        all_edges.extend(inter_edges)
        print(f"  ✓ Added {len(inter_edges)} cross-service edges")
    
    print(f"\n[CPG] Total call edges: {len(all_edges)}")
    
    # Map each pattern to a method
    print(f"\n[Mapping] Mapping error patterns to methods...")
    pattern_methods = {}
    
    for pattern in patterns:
        service = pattern.get('service', 'unknown')
        message = pattern.get('representative_message', pattern.get('message_template', ''))
        
        if service not in indexes:
            print(f"  ⚠ {pattern.get('pattern_id')}: No CPG for {service}")
            pattern_methods[pattern['pattern_id']] = {
                'qualified_name': f"{service}::<unknown>",
                'method': '<unknown>',
                'score': 0.0
            }
            continue
        
        # Match to method
        result = fast_match(message, indexes[service])
        
        if result['matched']:
            qualified_name = f"{service}::{result['function_name']}"
            print(f"  ✓ {pattern.get('pattern_id')}: {qualified_name} (score: {result['score']:.3f})")
            pattern_methods[pattern['pattern_id']] = {
                'qualified_name': qualified_name,
                'method': result['function_name'],
                'full_name': result['full_name'],
                'line': result['line'],
                'score': result['score']
            }
        else:
            print(f"  ⚠ {pattern.get('pattern_id')}: No method match")
            pattern_methods[pattern['pattern_id']] = {
                'qualified_name': f"{service}::<unknown>",
                'method': '<unknown>',
                'score': 0.0
            }
    
    return indexes, all_edges, pattern_methods


# ===========================================================================
# STEP 4: Iterative Forward/Backward Stepping
# ===========================================================================

def step4_iterative_stepping(
    pattern_methods: Dict[str, dict],
    all_call_edges: List[Tuple[str, str, str]],
    max_steps: int = 3
) -> Dict[str, dict]:
    """
    Step 4: Step forward and backward through call graph to find root cause.
    
    Returns: exploration_results
    """
    print("\n" + "=" * 70)
    print("STEP 4: ITERATIVE FORWARD/BACKWARD STEPPING")
    print("=" * 70)
    
    # Build call graph
    G = nx.DiGraph()
    for caller, callee, kind in all_call_edges:
        if kind == "CALL":
            G.add_edge(caller, callee)
    
    print(f"✓ Call graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Explore from each error method
    exploration = {}
    
    for pattern_id, method_info in pattern_methods.items():
        error_method = method_info['qualified_name']
        
        if error_method not in G:
            print(f"\n⚠ {pattern_id}: Method not in call graph")
            exploration[pattern_id] = {
                'error_method': error_method,
                'backward_methods': [],
                'forward_methods': [],
                'all_discovered': set([error_method])
            }
            continue
        
        print(f"\n✓ {pattern_id}: Exploring from {error_method}")
        
        # Step BACKWARD (find callers - potential root causes)
        backward_methods = []
        visited_back = set()
        queue = deque([(error_method, 0)])
        
        while queue:
            node, depth = queue.popleft()
            
            if depth > max_steps or node in visited_back:
                continue
            
            visited_back.add(node)
            
            if node != error_method:
                backward_methods.append({
                    'qualified_name': node,
                    'depth': depth,
                    'service': node.split('::')[0],
                    'method': node.split('::')[1] if '::' in node else node
                })
            
            # Get predecessors (methods that call this)
            for pred in G.predecessors(node):
                if pred not in visited_back:
                    queue.append((pred, depth + 1))
        
        print(f"  Backward: Found {len(backward_methods)} callers")
        
        # Step FORWARD (find callees - what error method tries to call)
        forward_methods = []
        visited_fwd = set()
        queue = deque([(error_method, 0)])
        
        while queue:
            node, depth = queue.popleft()
            
            if depth > max_steps or node in visited_fwd:
                continue
            
            visited_fwd.add(node)
            
            if node != error_method:
                forward_methods.append({
                    'qualified_name': node,
                    'depth': depth,
                    'service': node.split('::')[0],
                    'method': node.split('::')[1] if '::' in node else node
                })
            
            # Get successors (methods this calls)
            for succ in G.successors(node):
                if succ not in visited_fwd:
                    queue.append((succ, depth + 1))
        
        print(f"  Forward: Found {len(forward_methods)} callees")
        
        # Store results
        exploration[pattern_id] = {
            'error_method': error_method,
            'backward_methods': backward_methods,
            'forward_methods': forward_methods,
            'all_discovered': visited_back | visited_fwd,
            'call_graph': G
        }
        
        # Show cross-service calls (likely root causes)
        error_service = error_method.split('::')[0]
        cross_service_callers = [
            m for m in backward_methods 
            if m['service'] != error_service
        ]
        
        if cross_service_callers:
            print(f"  → Found {len(cross_service_callers)} cross-service callers (potential root causes):")
            for m in cross_service_callers[:5]:
                print(f"     • {m['qualified_name']} (depth: {m['depth']})")
    
    return exploration


# ===========================================================================
# STEP 5: Visualize Discovered Methods
# ===========================================================================

def step5_visualize_methods(
    patterns: List[dict],
    pattern_methods: Dict[str, dict],
    exploration: Dict[str, dict],
    output_dir: str = "rca_output"
) -> List[str]:
    """
    Step 5: Visualize discovered methods for each error pattern.
    
    Returns: List of output file paths
    """
    print("\n" + "=" * 70)
    print("STEP 5: VISUALIZE DISCOVERED METHODS")
    print("=" * 70)
    
    Path(output_dir).mkdir(exist_ok=True)
    output_files = []
    
    for pattern in patterns:
        pattern_id = pattern['pattern_id']
        
        if pattern_id not in exploration:
            continue
        
        exp = exploration[pattern_id]
        error_method = exp['error_method']
        
        print(f"\n✓ Visualizing {pattern_id}: {error_method}")
        
        # Create visualization
        output_file = f"{output_dir}/{pattern_id}_graph.png"
        
        _create_method_graph_viz(
            error_method=error_method,
            backward_methods=exp['backward_methods'],
            forward_methods=exp['forward_methods'],
            call_graph=exp.get('call_graph'),
            error_message=pattern.get('message_template', ''),
            output_file=output_file
        )
        
        output_files.append(output_file)
        print(f"  Saved: {output_file}")
    
    return output_files


def _create_method_graph_viz(
    error_method: str,
    backward_methods: List[dict],
    forward_methods: List[dict],
    call_graph: nx.DiGraph,
    error_message: str,
    output_file: str,
    dpi: int = 200
):
    """Create method graph visualization."""
    
    # Collect all nodes
    all_nodes = {error_method}
    all_nodes.update(m['qualified_name'] for m in backward_methods)
    all_nodes.update(m['qualified_name'] for m in forward_methods)
    
    # Build subgraph
    if call_graph:
        subG = call_graph.subgraph(all_nodes).copy()
    else:
        subG = nx.DiGraph()
        subG.add_node(error_method)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12), dpi=dpi)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)
    ax.axis('off')
    
    # Position nodes
    pos = {}
    
    # Error at center
    pos[error_method] = (0, 0)
    
    # Backward (callers) on left
    back_by_depth = defaultdict(list)
    for m in backward_methods:
        back_by_depth[m['depth']].append(m['qualified_name'])
    
    for depth, nodes in sorted(back_by_depth.items()):
        n = len(nodes)
        y_vals = [i - (n-1)/2 for i in range(n)]
        for node, y in zip(nodes, y_vals):
            pos[node] = (-2 * depth, y * 1.5)
    
    # Forward (callees) on right
    fwd_by_depth = defaultdict(list)
    for m in forward_methods:
        fwd_by_depth[m['depth']].append(m['qualified_name'])
    
    for depth, nodes in sorted(fwd_by_depth.items()):
        n = len(nodes)
        y_vals = [i - (n-1)/2 for i in range(n)]
        for node, y in zip(nodes, y_vals):
            pos[node] = (2 * depth, y * 1.5)
    
    # Service colors
    service_colors = {
        'frontend': '#4285F4',
        'adservice': '#EA4335',
        'cartservice': '#FBBC04',
        'checkoutservice': '#34A853',
        'currencyservice': '#9334E6',
        'emailservice': '#FF6D00',
        'paymentservice': '#00BFA5',
        'recommendationservice': '#D32F2F',
        'shippingservice': '#7B1FA2',
    }
    
    # Draw edges
    if subG:
        for u, v in subG.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                
                # Check if cross-service
                u_service = u.split('::')[0]
                v_service = v.split('::')[0]
                is_cross = (u_service != v_service)
                
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle='-|>',
                    connectionstyle='arc3,rad=0.1',
                    color='#E53935' if is_cross else '#999',
                    linewidth=2.5 if is_cross else 1.5,
                    mutation_scale=15,
                    zorder=5 if is_cross else 3
                )
                ax.add_patch(arrow)
    
    # Draw nodes
    for node, (x, y) in pos.items():
        service = node.split('::')[0]
        method = node.split('::')[1] if '::' in node else node
        
        is_error = (node == error_method)
        color = service_colors.get(service, '#78909C')
        
        # Box
        if is_error:
            box_color = '#FF4444'
            width, height = 1.5, 0.8
            edge_width = 3
        else:
            box_color = color
            width, height = 1.2, 0.6
            edge_width = 2
        
        box = FancyBboxPatch(
            (x - width/2, y - height/2),
            width, height,
            boxstyle='round,pad=0.1',
            facecolor='white' if not is_error else box_color,
            edgecolor=box_color,
            linewidth=edge_width,
            zorder=10
        )
        ax.add_patch(box)
        
        # Service label
        ax.text(x, y + height/2 + 0.15, service,
               ha='center', va='bottom',
               fontsize=7, color=color,
               fontweight='bold')
        
        # Method name
        display_method = method[:20] + '...' if len(method) > 20 else method
        ax.text(x, y, display_method,
               ha='center', va='center',
               fontsize=8 if not is_error else 9,
               fontweight='bold' if is_error else 'normal',
               color='white' if is_error else '#333')
    
    # Title
    ax.text(0, 4.5, f'Method Call Graph',
           ha='center', va='bottom',
           fontsize=16, fontweight='bold')
    
    ax.text(0, 4.2, f'Error: {error_message[:80]}...',
           ha='center', va='bottom',
           fontsize=10, color='#666')
    
    # Legend
    ax.text(-5.5, -4.5, '← BACKWARD (Callers)',
           ha='left', va='bottom',
           fontsize=10, fontweight='bold',
           color='#4CAF50')
    
    ax.text(5.5, -4.5, 'FORWARD (Callees) →',
           ha='right', va='bottom',
           fontsize=10, fontweight='bold',
           color='#2196F3')
    
    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=dpi)
    plt.close()


# ===========================================================================
# STEP 6: Export for LLM Analysis
# ===========================================================================

def step6_export_for_llm(
    patterns: List[dict],
    pattern_methods: Dict[str, dict],
    exploration: Dict[str, dict],
    visualization_files: List[str],
    output_dir: str = "rca_output"
) -> str:
    """
    Step 6: Export all data in LLM-friendly format.
    
    Returns: Path to LLM input file
    """
    print("\n" + "=" * 70)
    print("STEP 6: EXPORT FOR LLM ANALYSIS")
    print("=" * 70)
    
    llm_data = {
        'analysis_type': 'root_cause_analysis',
        'timestamp': datetime.now().isoformat(),
        'error_patterns': [],
        'visualization_files': visualization_files
    }
    
    for pattern in patterns:
        pattern_id = pattern['pattern_id']
        
        if pattern_id not in exploration:
            continue
        
        exp = exploration[pattern_id]
        method_info = pattern_methods.get(pattern_id, {})
        
        # Build LLM-friendly structure
        llm_pattern = {
            'pattern_id': pattern_id,
            'error_info': {
                'service': pattern.get('service'),
                'message': pattern.get('message_template'),
                'count': pattern.get('count'),
                'error_method': {
                    'qualified_name': method_info.get('qualified_name'),
                    'method': method_info.get('method'),
                    'full_name': method_info.get('full_name'),
                    'line': method_info.get('line'),
                    'matching_score': method_info.get('score')
                }
            },
            'call_analysis': {
                'backward_calls': {
                    'description': 'Methods that call the error method (potential root causes)',
                    'total_count': len(exp['backward_methods']),
                    'methods': [
                        {
                            'qualified_name': m['qualified_name'],
                            'service': m['service'],
                            'method': m['method'],
                            'depth': m['depth']
                        }
                        for m in exp['backward_methods']
                    ]
                },
                'forward_calls': {
                    'description': 'Methods called by the error method (what failed)',
                    'total_count': len(exp['forward_methods']),
                    'methods': [
                        {
                            'qualified_name': m['qualified_name'],
                            'service': m['service'],
                            'method': m['method'],
                            'depth': m['depth']
                        }
                        for m in exp['forward_methods']
                    ]
                },
                'cross_service_calls': {
                    'description': 'Calls between different services (likely root causes)',
                    'methods': [
                        m for m in exp['backward_methods']
                        if m['service'] != pattern.get('service')
                    ] + [
                        m for m in exp['forward_methods']
                        if m['service'] != pattern.get('service')
                    ]
                }
            },
            'visualization': f"{pattern_id}_graph.png"
        }
        
        llm_data['error_patterns'].append(llm_pattern)
    
    # Save to JSON
    output_file = f"{output_dir}/llm_input.json"
    with open(output_file, 'w') as f:
        json.dump(llm_data, f, indent=2)
    
    print(f"✓ Saved LLM input: {output_file}")
    print(f"  Patterns: {len(llm_data['error_patterns'])}")
    print(f"  Visualizations: {len(visualization_files)}")
    
    return output_file


# ===========================================================================
# STEP 7: Query LLM for RCA (Optional)
# ===========================================================================

def step7_query_llm(
    llm_input_file: str,
    api_key: Optional[str] = None
) -> Optional[dict]:
    """
    Step 7: Query LLM (Claude) for root cause analysis.
    
    Returns: LLM response
    """
    print("\n" + "=" * 70)
    print("STEP 7: QUERY LLM FOR ROOT CAUSE ANALYSIS")
    print("=" * 70)
    
    # Check for API key
    if not api_key:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("⚠ No API key provided. Skipping LLM analysis.")
        print("  Set ANTHROPIC_API_KEY environment variable to enable.")
        return None
    
    try:
        import anthropic
    except ImportError:
        print("⚠ anthropic package not installed. Run: pip install anthropic")
        return None
    
    # Load LLM input
    with open(llm_input_file, 'r') as f:
        llm_data = json.load(f)
    
    print(f"✓ Loaded analysis data for {len(llm_data['error_patterns'])} patterns")
    
    # Create prompt
    prompt = _create_llm_prompt(llm_data)
    
    print(f"✓ Querying Claude API...")
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result = {
            'response': response.content[0].text,
            'model': "claude-sonnet-4-20250514",
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✓ Received LLM analysis")
        
        # Save response
        output_file = llm_input_file.replace('llm_input.json', 'llm_response.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Saved LLM response: {output_file}")
        
        # Print summary
        print(f"\n" + "-" * 70)
        print("LLM ROOT CAUSE ANALYSIS:")
        print("-" * 70)
        print(result['response'][:1000])
        if len(result['response']) > 1000:
            print("... (truncated, see full response in file)")
        print("-" * 70)
        
        return result
        
    except Exception as e:
        print(f"✗ Error querying LLM: {e}")
        return None


def _create_llm_prompt(llm_data: dict) -> str:
    """Create prompt for LLM analysis."""
    
    prompt = """You are an expert in root cause analysis for microservices systems. 
I have analyzed error logs and traced method calls through the codebase using Code Property Graphs (CPG).

Please analyze the following error patterns and their call graphs to identify the root cause:

"""
    
    for i, pattern in enumerate(llm_data['error_patterns'], 1):
        error_info = pattern['error_info']
        call_analysis = pattern['call_analysis']
        
        prompt += f"""
## Error Pattern {i}: {pattern['pattern_id']}

**Error Details:**
- Service: {error_info['service']}
- Message: {error_info['message']}
- Occurrences: {error_info['count']}
- Error Method: {error_info['error_method']['qualified_name']}
- Matching Score: {error_info['error_method']['matching_score']:.2f}

**Backward Analysis (Who calls this method):**
Found {call_analysis['backward_calls']['total_count']} callers.
Top callers:
"""
        
        for m in call_analysis['backward_calls']['methods'][:10]:
            prompt += f"  - {m['qualified_name']} (depth: {m['depth']})\n"
        
        prompt += f"""
**Forward Analysis (What this method calls):**
Found {call_analysis['forward_calls']['total_count']} callees.
Top callees:
"""
        
        for m in call_analysis['forward_calls']['methods'][:10]:
            prompt += f"  - {m['qualified_name']} (depth: {m['depth']})\n"
        
        prompt += f"""
**Cross-Service Calls (Likely root causes):**
{len(call_analysis['cross_service_calls']['methods'])} cross-service interactions found.
"""
        
        for m in call_analysis['cross_service_calls']['methods'][:5]:
            prompt += f"  - {m['qualified_name']}\n"
        
        prompt += "\n---\n"
    
    prompt += """
Based on this analysis:

1. **Identify the most likely root cause** for each error pattern
2. **Rank services** by their likelihood of being the root cause (1 = most likely)
3. **Provide reasoning** for each ranking
4. **Suggest investigation steps** for confirming the root cause
5. **Recommend fixes** if the root cause is clear

Please provide your analysis in a structured format.
"""
    
    return prompt


# ===========================================================================
# Main Pipeline
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Iterative Root Cause Analysis with LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--csv', required=True,
                       help='Path to logs CSV')
    parser.add_argument('--services-dir', required=True,
                       help='Directory with CPG files')
    parser.add_argument('--output-dir', default='rca_output',
                       help='Output directory for results')
    parser.add_argument('--max-steps', type=int, default=3,
                       help='Maximum steps for forward/backward exploration')
    parser.add_argument('--use-llm', action='store_true',
                       help='Query LLM for RCA (requires ANTHROPIC_API_KEY)')
    parser.add_argument('--api-key',
                       help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("ITERATIVE ROOT CAUSE ANALYSIS SYSTEM")
    print("=" * 70)
    
    # Step 1: Retrieve error logs
    df, error_dicts = step1_retrieve_error_logs(args.csv)
    
    if not error_dicts:
        print("\n✗ No errors found. Exiting.")
        return
    
    # Step 2: Aggregate errors
    patterns = step2_aggregate_errors(error_dicts)
    
    # Step 3: Find methods using CPG
    indexes, all_call_edges, pattern_methods = step3_find_error_methods(
        patterns, df, args.services_dir
    )
    
    # Step 4: Iterative stepping
    exploration = step4_iterative_stepping(
        pattern_methods, all_call_edges, args.max_steps
    )
    
    # Step 5: Visualize
    visualization_files = step5_visualize_methods(
        patterns, pattern_methods, exploration, args.output_dir
    )
    
    # Step 6: Export for LLM
    llm_input_file = step6_export_for_llm(
        patterns, pattern_methods, exploration, 
        visualization_files, args.output_dir
    )


if __name__ == "__main__":
    main()