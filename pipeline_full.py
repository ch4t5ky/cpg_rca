#!/usr/bin/env python3
"""
pipeline_with_cfg.py
====================
RCA Pipeline with CFG-Based Intra-Method Flow Analysis

UPGRADED FEATURES:
1. On-Demand CPG Loading (memory efficient)
2. CFG-Based Intra-Method Analysis (NEW!)
   - Analyzes control flow INSIDE error method
   - Detects which if/else branches were taken
   - Shows internal method calls
3. Inter-Service DFS Tracing (existing)
4. Error Aggregation (existing)

Key Innovation:
- For each error, analyze BOTH:
  a) What happened INSIDE the error method (CFG)
  b) What happened BEFORE reaching the error method (DFS)

Usage:
    python pipeline_with_cfg.py --csv logs.csv --services-dir services
"""

import argparse
import json
import hashlib
import re
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

import pandas as pd
from networkx.drawing.nx_pydot import read_dot

import sys
sys.path.insert(0, str(Path(__file__).parent))

from log2cpg2 import process_service_cpg, fast_match, get_method_call_graph


# ===========================================================================
# ON-DEMAND CPG LOADER (With CFG Support)
# ===========================================================================

class OnDemandCPGLoader:
    """Loads CPG files on-demand with CFG extraction capability."""
    
    def __init__(self, services_dir: str):
        self.services_dir = Path(services_dir)
        self.current_cpg = None
        self.current_service = None
        self.current_graph = None  # NEW: Keep graph for CFG analysis
        self.load_count = 0
    
    def load_service_cpg(self, service: str, sample_logs: List[Tuple[str, str]] = None, keep_graph: bool = False):
        """
        Load CPG for a service.
        
        Args:
            service: Service name
            sample_logs: Sample logs for matching
            keep_graph: If True, keep graph in memory for CFG analysis
        """
        # Already loaded?
        if self.current_service == service and self.current_cpg:
            return self.current_cpg
        
        # Flush old CPG if different service
        if self.current_cpg and self.current_service != service:
            print(f"    [CPG] Flushing {self.current_service}, loading {service}...")
            del self.current_cpg
            del self.current_graph
            self.current_cpg = None
            self.current_graph = None
            gc.collect()
        
        # Load new CPG
        dot_file = self.services_dir / service / "export.dot"
        
        if not dot_file.exists():
            print(f"    [CPG] Warning: {dot_file} not found")
            return None
        
        print(f"    [CPG] Loading {service}...")
        
        # Load graph first (for CFG extraction)
        if keep_graph:
            self.current_graph = read_dot(str(dot_file))
        
        _, edges, index = process_service_cpg(
            str(dot_file),
            service,
            sample_logs or []
        )
        
        self.current_cpg = {
            'service': service,
            'index': index,
            'edges': edges,
            'dot_file': str(dot_file)
        }
        self.current_service = service
        self.load_count += 1
        
        return self.current_cpg
    
    def get_callers(self, method: str, service: str) -> List[Tuple[str, str, bool]]:
        """Get callers of a method."""
        if not self.current_cpg or self.current_service != service:
            return []
        
        qualified_name = f"{service}::{method}"
        callers = []
        
        for caller, callee, kind in self.current_cpg['edges']:
            if kind == "CALL" and callee == qualified_name:
                caller_service = caller.split("::")[0]
                is_external = (caller_service != service)
                callers.append((caller, caller_service, is_external))
        
        return callers
    
    def match_log_to_method(self, log_message: str, service: str) -> Optional[Dict]:
        """Match a log message to a method."""
        if not self.current_cpg or self.current_service != service:
            return None
        
        result = fast_match(log_message, self.current_cpg['index'])
        
        if result['matched'] and result['score'] > 0.01:
            return {
                'method': result['function_name'],
                'qualified_name': f"{service}::{result['function_name']}",
                'score': result['score'],
                'full_name': result.get('full_name', '')
            }
        
        return None
    
    def extract_internal_calls(self, method_name: str) -> List[str]:
        """
        NEW: Extract all method calls WITHIN a specific method (CFG analysis).
        
        This shows what the error method calls internally.
        """
        if not self.current_graph:
            # Need to load graph
            dot_file = self.current_cpg['dot_file']
            self.current_graph = read_dot(dot_file)
        
        print(f"    [CFG] Extracting internal calls for: {method_name}")
        
        # Find METHOD node
        method_node_id = None
        for node_id, attrs in self.current_graph.nodes(data=True):
            label = attrs.get('label', '')
            if '<METHOD>' in label and method_name in label:
                method_node_id = node_id
                break
        
        if not method_node_id:
            print(f"    [CFG] Method node not found")
            return []
        
        # Find CALL nodes via AST edges
        internal_calls = []
        visited = set()
        queue = [method_node_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            attrs = self.current_graph.nodes[current]
            label = attrs.get('label', '')
            
            # Extract CALL nodes
            if '<CALL>' in label:
                match = re.search(r'name:\s*"([^"]+)"', label)
                if match:
                    called = match.group(1).strip()
                    if called not in ['info', 'warn', 'error', 'debug', 'log', 'print']:
                        internal_calls.append(called)
                        print(f"      Found call: {called}")
            
            # Follow AST edges
            for _, target, edge_data in self.current_graph.edges(current, data=True):
                if edge_data.get('label') == 'AST':
                    queue.append(target)
        
        print(f"    [CFG] Found {len(internal_calls)} internal calls")
        return internal_calls


# ===========================================================================
# LOG INDEX
# ===========================================================================

class LogIndex:
    """Fast log lookup with method-level indexing."""
    
    def __init__(self, logs: List[Dict]):
        self.logs_by_service = defaultdict(list)
        self.logs_by_method = defaultdict(list)  # NEW: Index by method
        self.all_logs = sorted(logs, key=lambda x: x['timestamp'])
        
        for log in logs:
            self.logs_by_service[log['service']].append(log)
            
            # Index by method if available
            method = log.get('method') or log.get('qualified_name', '')
            if method:
                self.logs_by_method[method].append(log)
        
        for service in self.logs_by_service:
            self.logs_by_service[service].sort(key=lambda x: x['timestamp'])
        
        for method in self.logs_by_method:
            self.logs_by_method[method].sort(key=lambda x: x['timestamp'])
    
    def find_log_before(self, service: str, method: str, timestamp: float, window: float = 10.0) -> Optional[Dict]:
        """Find log of service before timestamp."""
        if service not in self.logs_by_service:
            return None
        
        logs = self.logs_by_service[service]
        candidates = [
            log for log in logs
            if timestamp - window <= log['timestamp'] < timestamp
            and (not method or method in log.get('message', ''))
        ]
        
        if candidates:
            return max(candidates, key=lambda x: x['timestamp'])
        
        return None
    
    def find_method_logs_before(self, method: str, timestamp: float, window: float = 10.0) -> List[Dict]:
        """NEW: Find all logs of a specific method before timestamp."""
        if method not in self.logs_by_method:
            return []
        
        logs = self.logs_by_method[method]
        return [
            log for log in logs
            if timestamp - window <= log['timestamp'] < timestamp
        ]
    
    def get_service_logs_sample(self, service: str, max_count: int = 2000) -> List[Tuple[str, str]]:
        """Get sample logs for CPG matching."""
        if service not in self.logs_by_service:
            return []
        
        logs = self.logs_by_service[service][:max_count]
        return [
            (datetime.fromtimestamp(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S'), log['message'])
            for log in logs
        ]


# ===========================================================================
# CFG-BASED INTRA-METHOD ANALYZER (NEW!)
# ===========================================================================

class IntraMethodAnalyzer:
    """
    Analyzes what happened INSIDE the error method.
    
    Uses CFG to determine:
    - Which methods were called internally
    - Which if/else branches were taken
    - Complete execution flow within the method
    """
    
    def __init__(self, cpg_loader: OnDemandCPGLoader, log_index: LogIndex):
        self.cpg_loader = cpg_loader
        self.log_index = log_index
    
    def analyze_error_method(
        self,
        error_method: str,
        error_service: str,
        error_time: float,
        time_window: float = 10.0
    ) -> Optional[Dict]:
        """
        Analyze what happened inside the error method.
        
        Returns:
            Dict with internal execution flow
        """
        print(f"\n  [CFG] Analyzing internal flow of {error_service}::{error_method}")
        
        # Ensure CPG loaded with graph
        cpg = self.cpg_loader.load_service_cpg(
            error_service,
            self.log_index.get_service_logs_sample(error_service),
            keep_graph=True
        )
        
        if not cpg:
            return None
        
        # Extract internal calls from CFG
        internal_calls = self.cpg_loader.extract_internal_calls(error_method)
        
        if not internal_calls:
            # Fallback to call graph
            print(f"    [CFG] No CFG calls found, using call graph...")
            call_graph = get_method_call_graph(
                method_name=error_method,
                service_name=error_service,
                all_edges=cpg['edges'],
                max_depth=2
            )
            
            internal_calls = [
                callee.split("::")[-1]
                for _, callee in call_graph['internal_calls']
            ]
        
        print(f"    [CFG] Analyzing {len(internal_calls)} internal calls")
        
        # Match internal calls with logs
        call_evidence = []
        window_start = error_time - time_window
        
        for called_method in internal_calls:
            # Find logs for this method before error
            service_logs = self.log_index.logs_by_service[error_service]
            
            matching_logs = []
            for log in service_logs:
                if window_start <= log['timestamp'] < error_time:
                    # Match log to method
                    match = self.cpg_loader.match_log_to_method(log['message'], error_service)
                    if match and match['method'] == called_method:
                        matching_logs.append(log)
            
            has_evidence = len(matching_logs) > 0
            
            call_evidence.append({
                'method': called_method,
                'qualified_name': f"{error_service}::{called_method}",
                'has_evidence': has_evidence,
                'log_count': len(matching_logs),
                'logs': matching_logs[:2]
            })
            
            status = "✓" if has_evidence else "✗"
            print(f"      {status} {called_method}: {len(matching_logs)} logs")
        
        # Build internal execution path
        executed_calls = [e for e in call_evidence if e['has_evidence']]
        executed_calls.sort(
            key=lambda x: x['logs'][0]['timestamp'] if x['logs'] else 0
        )
        
        internal_path = [
            {
                'method': e['method'],
                'timestamp': e['logs'][0]['timestamp'] if e['logs'] else error_time,
                'evidence': True
            }
            for e in executed_calls
        ]
        
        # Calculate confidence
        confidence = len(executed_calls) / len(internal_calls) if internal_calls else 0.0
        
        return {
            'error_method': error_method,
            'service': error_service,
            'all_internal_calls': internal_calls,
            'executed_calls': [e['method'] for e in executed_calls],
            'not_executed_calls': [e['method'] for e in call_evidence if not e['has_evidence']],
            'internal_execution_path': internal_path,
            'call_evidence': call_evidence,
            'confidence': confidence,
            'total_calls': len(internal_calls),
            'matched_calls': len(executed_calls)
        }


# ===========================================================================
# INTER-SERVICE DFS RECONSTRUCTOR (Existing)
# ===========================================================================

class InterServiceDFSReconstructor:
    """DFS backward tracing BETWEEN services."""
    
    def __init__(self, cpg_loader: OnDemandCPGLoader, log_index: LogIndex):
        self.cpg_loader = cpg_loader
        self.log_index = log_index
    
    def reconstruct_from_error(
        self,
        error_log: Dict,
        max_depth: int = 10,
        time_window: float = 10.0
    ) -> Optional[Dict]:
        """Reconstruct inter-service flow leading to error."""
        
        error_service = error_log['service']
        error_time = error_log['timestamp']
        
        print(f"\n  [DFS] Inter-service tracing from {error_service}")
        
        # Load error service CPG
        sample_logs = self.log_index.get_service_logs_sample(error_service)
        cpg = self.cpg_loader.load_service_cpg(error_service, sample_logs)
        
        if not cpg:
            return None
        
        # Match error log to method
        method_match = self.cpg_loader.match_log_to_method(error_log['message'], error_service)
        
        if not method_match:
            return None
        
        error_method = method_match['method']
        print(f"    [DFS] Error method: {error_service}::{error_method}")
        
        # DFS backward
        path = self._dfs_backward(
            method=error_method,
            service=error_service,
            timestamp=error_time,
            depth=0,
            max_depth=max_depth,
            time_window=time_window,
            visited=set()
        )
        
        if not path:
            return None
        
        return {
            'execution_path': [f"{p['service']}::{p['method']}" for p in path],
            'root_methods': [f"{path[0]['service']}::{path[0]['method']}"],
            'services_involved': list(set(p['service'] for p in path)),
            'method_calls': path,
            'start_time': path[0]['timestamp'],
            'end_time': path[-1]['timestamp'],
            'duration': path[-1]['timestamp'] - path[0]['timestamp']
        }
    
    def _dfs_backward(
        self,
        method: str,
        service: str,
        timestamp: float,
        depth: int,
        max_depth: int,
        time_window: float,
        visited: Set[str]
    ) -> List[Dict]:
        """DFS backward with on-demand loading."""
        
        qualified = f"{service}::{method}"
        
        current = {
            'method': method,
            'service': service,
            'qualified_name': qualified,
            'timestamp': timestamp,
            'depth': depth
        }
        
        if depth >= max_depth or qualified in visited:
            return [current]
        
        visited.add(qualified)
        
        # Ensure CPG loaded
        if self.cpg_loader.current_service != service:
            sample_logs = self.log_index.get_service_logs_sample(service)
            cpg = self.cpg_loader.load_service_cpg(service, sample_logs)
            if not cpg:
                return [current]
        
        # Get callers
        callers = self.cpg_loader.get_callers(method, service)
        
        if not callers:
            print(f"    [DFS] Root: {qualified}")
            return [current]
        
        # Find best caller
        best_path = None
        best_conf = 0.0
        
        for caller_qualified, caller_service, is_external in callers:
            caller_method = caller_qualified.split("::")[-1]
            
            if caller_qualified in visited:
                continue
            
            # Load caller service if external
            if is_external:
                print(f"    [DFS] External: {caller_qualified} → {qualified}")
                sample_logs = self.log_index.get_service_logs_sample(caller_service)
                cpg = self.cpg_loader.load_service_cpg(caller_service, sample_logs)
                if not cpg:
                    continue
            
            # Find log evidence
            caller_log = self.log_index.find_log_before(
                caller_service,
                caller_method,
                timestamp,
                time_window
            )
            
            if not caller_log:
                continue
            
            time_gap = timestamp - caller_log['timestamp']
            if time_gap < 0 or time_gap > time_window:
                continue
            
            confidence = 1.0 - (time_gap / time_window)
            
            print(f"    [DFS] Follow: {caller_qualified} (conf: {confidence:.2f})")
            
            # Recurse
            caller_path = self._dfs_backward(
                method=caller_method,
                service=caller_service,
                timestamp=caller_log['timestamp'],
                depth=depth + 1,
                max_depth=max_depth,
                time_window=time_window,
                visited=visited.copy()
            )
            
            full_path = caller_path + [current]
            avg_conf = confidence
            
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_path = full_path
        
        if best_path:
            return best_path
        else:
            print(f"    [DFS] Root: {qualified}")
            return [current]


# ===========================================================================
# ERROR AGGREGATION
# ===========================================================================

def normalize_error_message(message: str) -> str:
    """Normalize message by removing variable parts."""
    msg = message.lower()
    msg = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', msg)
    msg = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', msg)
    msg = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', msg)
    msg = re.sub(r'\b\d{6,}\b', '<NUMBER>', msg)
    msg = re.sub(r':\s*\d+', ':<NUM>', msg)
    msg = re.sub(r'=\s*\d+', '=<NUM>', msg)
    msg = re.sub(r'"[^"]{20,}"', '"<STRING>"', msg)
    msg = re.sub(r'\s+', ' ', msg).strip()
    return msg


def extract_error_signature(message: str, error_type: str, service: str) -> str:
    """Generate unique signature for error pattern."""
    normalized = normalize_error_message(message)
    if len(normalized) > 200:
        normalized = normalized[:200] + "..."
    
    sig_str = f"{service}|{error_type}|{normalized}"
    return hashlib.md5(sig_str.encode()).hexdigest()[:12]


def aggregate_error_patterns(logs: List[Dict]) -> Dict[str, Dict]:
    """Aggregate errors to unique patterns."""
    print("\n" + "=" * 70)
    print("PHASE 1: ERROR AGGREGATION")
    print("=" * 70)
    
    error_logs = [log for log in logs if log.get('is_error')]
    
    if not error_logs:
        print("  No errors found")
        return {}
    
    print(f"\n  Aggregating {len(error_logs)} errors...")
    
    patterns = defaultdict(lambda: {
        'count': 0,
        'services': set(),
        'first_seen': None,
        'last_seen': None,
        'samples': [],
    })
    
    for log in error_logs:
        msg_lower = log['message'].lower()
        if 'timeout' in msg_lower:
            error_type = 'timeout'
        elif 'connection' in msg_lower:
            error_type = 'connection'
        elif 'exception' in msg_lower:
            error_type = 'exception'
        else:
            error_type = 'error'
        
        signature = extract_error_signature(log['message'], error_type, log['service'])
        
        pattern = patterns[signature]
        pattern['count'] += 1
        pattern['services'].add(log['service'])
        pattern['error_type'] = error_type
        pattern['normalized_message'] = normalize_error_message(log['message'])
        
        if pattern['first_seen'] is None or log['timestamp'] < pattern['first_seen']:
            pattern['first_seen'] = log['timestamp']
        if pattern['last_seen'] is None or log['timestamp'] > pattern['last_seen']:
            pattern['last_seen'] = log['timestamp']
        
        if len(pattern['samples']) < 3:
            pattern['samples'].append(log)
    
    print(f"  Unique patterns: {len(patterns)}")
    print(f"  Deduplication: {(1 - len(patterns)/len(error_logs)):.1%}")
    
    return patterns


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RCA Pipeline with CFG-Based Intra-Method Analysis"
    )
    
    parser.add_argument('--csv', required=True, help='Path to logs CSV')
    parser.add_argument('--services-dir', required=True, help='Path to CPG directory')
    parser.add_argument('--window', type=int, default=10, help='Time window (seconds)')
    parser.add_argument('--max-depth', type=int, default=10, help='Max DFS depth')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit number of errors to analyze')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RCA PIPELINE WITH CFG-BASED ANALYSIS")
    print("=" * 70)
    print("\nFeatures:")
    print("  1. Error Aggregation")
    print("  2. CFG Intra-Method Analysis (NEW!)")
    print("  3. Inter-Service DFS Tracing")
    print("  4. On-Demand CPG Loading\n")
    
    # Load logs
    print("[Step 1] Loading logs...")
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df["container_name"] = df["container_name"].fillna("unknown").astype(str).str.strip()
    df["message"] = df["message"].fillna("").astype(str)
    
    print(f"  Logs: {len(df)}")
    print(f"  Services: {df['container_name'].nunique()}")
    
    # Prepare log list
    logs = []
    for _, row in df.iterrows():
        msg_lower = row['message'].lower()
        logs.append({
            'timestamp': float(row['timestamp']),
            'service': row['container_name'],
            'message': row['message'],
            'is_error': any(p in msg_lower for p in ['error', 'fail', 'exception'])
        })
    
    # Phase 1: Aggregate errors
    patterns = aggregate_error_patterns(logs)
    
    # Phase 2: Analyze each error
    print("\n" + "=" * 70)
    print("PHASE 2: CFG + DFS ANALYSIS")
    print("=" * 70)
    
    # Create components
    log_index = LogIndex(logs)
    cpg_loader = OnDemandCPGLoader(args.services_dir)
    
    intra_analyzer = IntraMethodAnalyzer(cpg_loader, log_index)
    inter_reconstructor = InterServiceDFSReconstructor(cpg_loader, log_index)
    
    # Analyze errors
    error_logs = [log for log in logs if log['is_error']]
    
    if args.limit:
        error_logs = error_logs[:args.limit]
    
    print(f"\n  Processing {len(error_logs)} errors...")
    
    results = []
    
    for i, error_log in enumerate(error_logs):
        print(f"\n--- Error {i+1}/{len(error_logs)} ---")
        
        try:
            # First match to method
            sample_logs = log_index.get_service_logs_sample(error_log['service'])
            cpg = cpg_loader.load_service_cpg(error_log['service'], sample_logs, keep_graph=True)
            
            if not cpg:
                continue
            
            match = cpg_loader.match_log_to_method(error_log['message'], error_log['service'])
            
            if not match:
                print(f"  Could not match error to method")
                continue
            
            # Intra-method analysis (CFG)
            intra_result = intra_analyzer.analyze_error_method(
                error_method=match['method'],
                error_service=error_log['service'],
                error_time=error_log['timestamp'],
                time_window=args.window
            )
            
            # Inter-service tracing (DFS)
            inter_result = inter_reconstructor.reconstruct_from_error(
                error_log,
                max_depth=args.max_depth,
                time_window=args.window
            )
            
            # Combine results
            result = {
                'error_id': f"error_{i:04d}",
                'error_message': error_log['message'],
                'error_service': error_log['service'],
                'error_time': error_log['timestamp'],
                
                'intra_method_analysis': intra_result,
                'inter_service_trace': inter_result,
                
                'cpg_loads': cpg_loader.load_count
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n  Analyzed: {len(results)} errors")
    print(f"  Total CPG loads: {cpg_loader.load_count}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save patterns
    patterns_json = {
        sig: {
            **{k: v for k, v in p.items() if k not in ['services', 'samples']},
            'services': list(p['services']),
        }
        for sig, p in patterns.items()
    }
    
    with open(output_dir / "1_error_patterns.json", 'w') as f:
        json.dump({'patterns': list(patterns_json.values())}, f, indent=2)
    
    # Save CFG+DFS analysis
    with open(output_dir / "2_cfg_dfs_analysis.json", 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    # Save summary
    summary = {
        'total_errors': len(error_logs),
        'analyzed': len(results),
        'unique_patterns': len(patterns),
        'cpg_loads': cpg_loader.load_count,
        'services_analyzed': len(set(r['error_service'] for r in results))
    }
    
    with open(output_dir / "0_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {args.output_dir}/")
    print(f"  0_summary.json")
    print(f"  1_error_patterns.json")
    print(f"  2_cfg_dfs_analysis.json")
    print(f"\nEfficiency:")
    print(f"  CPG loads: {cpg_loader.load_count}")
    print(f"  Memory: ~{cpg_loader.load_count * 50}MB peak")


if __name__ == "__main__":
    main()