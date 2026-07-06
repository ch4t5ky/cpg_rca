"""
Log-to-Handler Path Tracer

This module provides functionality to trace from a single log message back to its
complete execution path and HTTP handler. It combines log template matching with
reverse call graph traversal to reconstruct the full execution context.

Key Features:
- Traces from log message to originating method
- Finds the HTTP endpoint handler that initiated the execution
- Reconstructs the complete call path with control flow context
- Supports multiple path scenarios and branching logic

Usage:
    tracer = LogToHandlerTracer(G)
    result = tracer.trace_log_to_handler("request started for user 123")
"""

from __future__ import annotations

import html
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from cpg.endpoint import EndpointDetector, Endpoint
from cpg.flow import EndpointFlow, EndpointFlowResult, MethodEntry
from cpg.method import MethodConstructor, MethodGraph
import cpg_trie_parser as parser

__all__ = [
    "LogTraceResult",
    "ExecutionPath", 
    "CallStep",
    "LogToHandlerTracer",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CallStep:
    """A single step in the execution path from handler to log."""
    method_name: str
    method_full_name: str
    method_node_id: str
    call_node_id: Optional[str] = None
    call_code: Optional[str] = None
    line_number: Optional[str] = None
    depth: int = 0
    branch_condition: Optional[str] = None


@dataclass
class ExecutionPath:
    """Complete execution path from handler to log statement."""
    handler_endpoint: Optional[Endpoint] = None
    call_chain: List[CallStep] = field(default_factory=list)
    log_method: Optional[str] = None
    log_template: Optional[str] = None
    confidence_score: float = 0.0
    
    def summary(self) -> str:
        """Generate a human-readable summary of the execution path."""
        lines = []
        if self.handler_endpoint:
            lines.append(f"Handler: {self.handler_endpoint.name} ({self.handler_endpoint.full_name})")
        else:
            lines.append("Handler: Unknown")
            
        lines.append(f"Call Chain ({len(self.call_chain)} steps):")
        for i, step in enumerate(self.call_chain):
            indent = "  " + "  " * step.depth
            branch_info = f" [{step.branch_condition}]" if step.branch_condition else ""
            lines.append(f"{indent}{i+1}. {step.method_name}(){branch_info}")
            if step.call_code:
                lines.append(f"{indent}   Code: {step.call_code}")
                
        if self.log_method:
            lines.append(f"Log Method: {self.log_method}")
        if self.log_template:
            lines.append(f"Log Template: {self.log_template}")
        lines.append(f"Confidence: {self.confidence_score:.2f}")
        
        return "\n".join(lines)


@dataclass
class LogTraceResult:
    """Result of tracing a log message to its handler."""
    original_message: str
    matched_template: Optional[str] = None
    possible_paths: List[ExecutionPath] = field(default_factory=list)
    best_path: Optional[ExecutionPath] = None
    
    def summary(self) -> str:
        """Generate a summary of the trace result."""
        lines = [f"Log Message: {self.original_message}"]
        if self.matched_template:
            lines.append(f"Matched Template: {self.matched_template}")
        else:
            lines.append("No template match found")
            
        lines.append(f"Possible Paths: {len(self.possible_paths)}")
        
        if self.best_path:
            lines.append("\nBest Path:")
            lines.append(self.best_path.summary())
            
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Tracer Class
# ─────────────────────────────────────────────────────────────────────────────


class LogToHandlerTracer:
    """
    Traces log messages back to their originating HTTP handlers.
    
    This class combines several analysis techniques:
    1. Template matching to find the method that generated the log
    2. Reverse call graph traversal to find potential handlers
    3. Forward flow analysis to validate and rank possible paths
    """
    
    def __init__(self, G: nx.MultiDiGraph, max_depth: int = 10, max_paths: int = 50):
        """
        Initialize the tracer with a Code Property Graph.
        
        Args:
            G: NetworkX MultiDiGraph representing the CPG
            max_depth: Maximum depth for call graph traversal
            max_paths: Maximum number of paths to consider
        """
        self.G = G
        self.max_depth = max_depth
        self.max_paths = max_paths
        
        # Initialize components
        self.endpoint_detector = EndpointDetector(G)
        self.method_constructor = MethodConstructor(G)
        self.flow_analyzer = EndpointFlow(G, max_depth=max_depth, max_paths=max_paths)
        
        # Build indexes
        self._build_indexes()
        
    def _build_indexes(self):
        """Build internal indexes for efficient lookup."""
        # Build method name to node ID mapping
        self.method_index = {}
        self.reverse_call_graph = defaultdict(set)
        
        for nid, data in self.G.nodes(data=True):
            label = self._clean(data.get("label", "")).upper()
            if label == "METHOD":
                full_name = self._clean(data.get("FULL_NAME", ""))
                name = self._clean(data.get("NAME", ""))
                if full_name:
                    self.method_index[full_name] = nid
                if name:
                    self.method_index[name] = nid
                    
        # Build reverse call graph (callee -> callers)
        for nid, data in self.G.nodes(data=True):
            label = self._clean(data.get("label", "")).upper()
            if label == "CALL":
                # Find the method this call belongs to
                caller_method = self._find_containing_method(nid)
                # Find what this call invokes
                callee_method = self._find_called_method(nid)
                
                if caller_method and callee_method:
                    self.reverse_call_graph[callee_method].add(caller_method)
    
    def trace_log_to_handler(
        self, 
        log_message: str,
        service_name: Optional[str] = None,
        timestamp: Optional[int] = None
    ) -> LogTraceResult:
        """
        Trace a log message back to its originating handler.
        
        Args:
            log_message: The log message to trace
            service_name: Optional service name for context
            timestamp: Optional timestamp for context
            
        Returns:
            LogTraceResult containing all possible paths and the best match
        """
        result = LogTraceResult(original_message=log_message)
        
        # Step 1: Find the method that generated this log
        log_mapping = self._match_log_to_method(log_message)
        if not log_mapping or not log_mapping.matched:
            return result
            
        result.matched_template = log_mapping.template
        
        # Step 2: Find all possible handlers that could reach this method
        possible_handlers = self._find_possible_handlers(log_mapping.method_node_id)
        
        # Step 3: For each handler, try to build the execution path
        for handler in possible_handlers:
            paths = self._build_execution_paths(handler, log_mapping)
            result.possible_paths.extend(paths)
            
        # Step 4: Rank paths and select the best one
        if result.possible_paths:
            result.possible_paths.sort(key=lambda p: p.confidence_score, reverse=True)
            result.best_path = result.possible_paths[0]
            
        return result
    
    def _match_log_to_method(self, log_message: str) -> Optional[parser.LogMapping]:
        """Match a log message to a method using template matching."""
        # Build templates from CPG
        templates = parser.build_templates_from_cpg(self.G, max_ddg_depth=5)
        if not templates:
            return None
            
        # Build trie for matching
        root = parser.build_trie(templates)
        
        # Create log entry for matching
        log_rows = [("trace", log_message)]
        
        # Perform matching
        mappings = parser.map_logs(log_rows, root, templates, min_static=1)
        
        return mappings[0] if mappings else None
    
    def _find_possible_handlers(self, method_node_id: str) -> List[Endpoint]:
        """Find all HTTP handlers that could potentially call the given method."""
        # Get all endpoints
        endpoints = self.endpoint_detector.detect()
        
        # For each endpoint, check if it can reach the target method
        reachable_endpoints = []
        
        for endpoint in endpoints:
            if self._can_reach_method(endpoint.node_id, method_node_id):
                reachable_endpoints.append(endpoint)
                
        return reachable_endpoints
    
    def _can_reach_method(self, handler_node_id: str, target_method_id: str) -> bool:
        """Check if a handler can reach a target method through call graph traversal."""
        visited = set()
        queue = deque([handler_node_id])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            if current == target_method_id:
                return True
                
            # Find methods called by current method
            called_methods = self._find_called_methods(current)
            for called in called_methods:
                if called not in visited:
                    queue.append(called)
                    
        return False
    
    def _build_execution_paths(
        self, 
        handler: Endpoint, 
        log_mapping: parser.LogMapping
    ) -> List[ExecutionPath]:
        """Build detailed execution paths from handler to log method."""
        paths = []
        
        try:
            # Use flow analyzer to get detailed execution flow
            flow_result = self.flow_analyzer.build(handler.node_id)
            
            # Find paths that include our target method
            target_method_name = log_mapping.method_name
            
            for method_full_name, method_paths in flow_result.method_paths.items():
                if target_method_name in method_full_name or method_full_name.endswith(f".{target_method_name}"):
                    # Build execution path from flow result
                    exec_path = self._build_path_from_flow(flow_result, method_full_name, log_mapping)
                    if exec_path:
                        paths.append(exec_path)
                        
        except Exception as e:
            # Fallback to simpler path construction
            simple_path = self._build_simple_path(handler, log_mapping)
            if simple_path:
                paths.append(simple_path)
                
        return paths
    
    def _build_path_from_flow(
        self, 
        flow_result: EndpointFlowResult, 
        target_method: str,
        log_mapping: parser.LogMapping
    ) -> Optional[ExecutionPath]:
        """Build an execution path from flow analysis result."""
        path = ExecutionPath()
        path.handler_endpoint = Endpoint(
            node_id=flow_result.endpoint_node_id,
            name=flow_result.endpoint_name,
            full_name=flow_result.endpoint_full_name
        )
        path.log_method = log_mapping.method_name
        path.log_template = log_mapping.template
        
        # Build call chain from sequence
        for i, entry in enumerate(flow_result.sequence):
            step = CallStep(
                method_name=entry.method_graph.name,
                method_full_name=entry.method_graph.full_name,
                method_node_id=entry.method_graph.node_id,
                depth=entry.depth
            )
            
            # Add branch condition if available
            if entry.via_path and entry.via_path.segments:
                for segment in entry.via_path.segments:
                    if segment.condition:
                        step.branch_condition = segment.condition
                        break
                        
            path.call_chain.append(step)
            
            # Stop if we reached our target method
            if target_method in entry.method_graph.full_name:
                break
                
        # Calculate confidence score
        path.confidence_score = self._calculate_confidence(path, log_mapping)
        
        return path
    
    def _build_simple_path(
        self, 
        handler: Endpoint, 
        log_mapping: parser.LogMapping
    ) -> Optional[ExecutionPath]:
        """Build a simple execution path using basic call graph traversal."""
        path = ExecutionPath()
        path.handler_endpoint = handler
        path.log_method = log_mapping.method_name
        path.log_template = log_mapping.template
        
        # Simple BFS to find path from handler to log method
        visited = set()
        queue = deque([(handler.node_id, [])])
        
        while queue:
            current_method, call_chain = queue.popleft()
            
            if current_method in visited:
                continue
            visited.add(current_method)
            
            # Create step for current method
            method_data = self.G.nodes.get(current_method, {})
            step = CallStep(
                method_name=self._clean(method_data.get("NAME", "")),
                method_full_name=self._clean(method_data.get("FULL_NAME", "")),
                method_node_id=current_method,
                depth=len(call_chain)
            )
            
            new_chain = call_chain + [step]
            
            # Check if we reached the target
            if current_method == log_mapping.method_node_id:
                path.call_chain = new_chain
                path.confidence_score = self._calculate_confidence(path, log_mapping)
                return path
                
            # Continue search
            if len(new_chain) < self.max_depth:
                called_methods = self._find_called_methods(current_method)
                for called in called_methods:
                    if called not in visited:
                        queue.append((called, new_chain))
                        
        return None
    
    def _calculate_confidence(self, path: ExecutionPath, log_mapping: parser.LogMapping) -> float:
        """Calculate confidence score for an execution path."""
        score = 0.0
        
        # Base score from template matching
        score += log_mapping.score * 0.3
        
        # Bonus for shorter paths (more direct)
        if path.call_chain:
            path_length = len(path.call_chain)
            score += max(0, (10 - path_length) * 0.1)
            
        # Bonus for having a clear handler
        if path.handler_endpoint:
            score += 0.2
            
        # Bonus for branch conditions (more specific)
        branch_count = sum(1 for step in path.call_chain if step.branch_condition)
        score += branch_count * 0.05
        
        return min(1.0, score)
    
    def _find_containing_method(self, node_id: str) -> Optional[str]:
        """Find the METHOD node that contains the given node."""
        visited = set()
        queue = deque([node_id])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            # Check if current node is a METHOD
            data = self.G.nodes.get(current, {})
            if self._clean(data.get("label", "")).upper() == "METHOD":
                return current
                
            # Check parents
            for parent, _, edge_data in self.G.in_edges(current, data=True):
                edge_label = self._clean(edge_data.get("label", "")).upper()
                if edge_label in ["AST", "CFG", "CONTAINS"]:
                    queue.append(parent)
                    
        return None
    
    def _find_called_method(self, call_node_id: str) -> Optional[str]:
        """Find the METHOD node that is called by the given CALL node."""
        # Look for outgoing edges that indicate method calls
        for _, target, edge_data in self.G.out_edges(call_node_id, data=True):
            target_data = self.G.nodes.get(target, {})
            if self._clean(target_data.get("label", "")).upper() == "METHOD":
                return target
                
        # Alternative: look for method reference by name
        call_data = self.G.nodes.get(call_node_id, {})
        method_name = self._clean(call_data.get("METHOD_FULL_NAME", ""))
        if not method_name:
            method_name = self._clean(call_data.get("NAME", ""))
            
        if method_name and method_name in self.method_index:
            return self.method_index[method_name]
            
        return None
    
    def _find_called_methods(self, method_node_id: str) -> List[str]:
        """Find all methods called by the given method."""
        called_methods = []
        
        # Find all CALL nodes within this method
        method_graph = self.method_constructor.build(method_node_id)
        
        for cfg_node in method_graph.cfg_nodes:
            if cfg_node.label.upper() == "CALL":
                called_method = self._find_called_method(cfg_node.node_id)
                if called_method:
                    called_methods.append(called_method)
                    
        return called_methods
    
    def _clean(self, value) -> str:
        """Clean and normalize string values from CPG."""
        s = html.unescape(str(value)).strip()
        if len(s) >= 2 and s[0] == s[-1] == '"':
            s = s[1:-1]
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def trace_single_log(
    G: nx.MultiDiGraph, 
    log_message: str,
    max_depth: int = 10,
    max_paths: int = 50
) -> LogTraceResult:
    """
    Convenience function to trace a single log message.
    
    Args:
        G: Code Property Graph
        log_message: Log message to trace
        max_depth: Maximum traversal depth
        max_paths: Maximum paths to consider
        
    Returns:
        LogTraceResult with trace information
    """
    tracer = LogToHandlerTracer(G, max_depth=max_depth, max_paths=max_paths)
    return tracer.trace_log_to_handler(log_message)


def print_trace_result(result: LogTraceResult, show_all_paths: bool = False):
    """
    Print a formatted trace result.
    
    Args:
        result: The trace result to print
        show_all_paths: Whether to show all possible paths or just the best one
    """
    print("=" * 80)
    print("LOG-TO-HANDLER TRACE RESULT")
    print("=" * 80)
    print(result.summary())
    
    if show_all_paths and len(result.possible_paths) > 1:
        print("\n" + "─" * 80)
        print("ALL POSSIBLE PATHS:")
        print("─" * 80)
        for i, path in enumerate(result.possible_paths):
            print(f"\nPath {i+1} (confidence: {path.confidence_score:.2f}):")
            print(path.summary())