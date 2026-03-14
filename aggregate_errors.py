#!/usr/bin/env python3
"""
aggregate_errors.py
===================
Aggregate and deduplicate errors to find unique error patterns.

Features:
- Multiple aggregation strategies (message similarity, error signature, stack trace)
- Configurable similarity thresholds
- Representative error selection
- Frequency analysis
- Export unique errors with statistics

Usage:
    python aggregate_errors.py --csv logs.csv
    python aggregate_errors.py --json errors.json
    python aggregate_errors.py --csv logs.csv --similarity 0.8
    python aggregate_errors.py --csv logs.csv --group-by signature
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class ErrorPattern:
    """Represents a unique error pattern."""
    pattern_id: str
    error_type: str
    service: str
    
    # Signature and grouping
    signature: str
    message_template: str
    
    # Statistics
    count: int = 0
    services_affected: Set[str] = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0
    
    # Representative examples
    representative_message: str = ""
    sample_messages: List[str] = field(default_factory=list)
    
    # Traceback info
    has_traceback: bool = False
    common_traceback_lines: List[str] = field(default_factory=list)
    exception_types: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            'pattern_id': self.pattern_id,
            'error_type': self.error_type,
            'service': self.service,
            'signature': self.signature,
            'message_template': self.message_template,
            'count': self.count,
            'services_affected': list(self.services_affected),
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'representative_message': self.representative_message,
            'sample_messages': self.sample_messages[:5],
            'has_traceback': self.has_traceback,
            'common_traceback_lines': self.common_traceback_lines,
            'exception_types': list(self.exception_types),
        }


@dataclass
class AggregationStats:
    """Statistics about the aggregation process."""
    total_errors: int = 0
    unique_patterns: int = 0
    services_affected: int = 0
    deduplication_ratio: float = 0.0
    top_patterns: List[Tuple[str, int]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'total_errors': self.total_errors,
            'unique_patterns': self.unique_patterns,
            'services_affected': self.services_affected,
            'deduplication_ratio': round(self.deduplication_ratio, 2),
            'top_patterns': self.top_patterns[:10],
        }


# ===========================================================================
# Signature Generation
# ===========================================================================

def normalize_message(message: str) -> str:
    """
    Normalize message by removing variable parts.
    This helps group similar errors together.
    """
    msg = message.lower()
    
    # Remove UUIDs
    msg = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 
                 '<UUID>', msg)
    
    # Remove timestamps
    msg = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', msg)
    msg = re.sub(r'\d{10,13}', '<TIMESTAMP>', msg)
    
    # Remove IP addresses
    msg = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', msg)
    
    # Remove email addresses
    msg = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w+\b', '<EMAIL>', msg)
    
    # Remove URLs
    msg = re.sub(r'https?://[^\s]+', '<URL>', msg)
    
    # Remove hex values
    msg = re.sub(r'0x[0-9a-f]+', '<HEX>', msg)
    
    # Remove large numbers
    msg = re.sub(r'\b\d{6,}\b', '<NUMBER>', msg)
    
    # Remove version numbers
    msg = re.sub(r'\bv?\d+\.\d+(\.\d+)?([.-]\w+)?\b', '<VERSION>', msg)
    
    # Remove file paths
    msg = re.sub(r'(?:/[\w.-]+)+/?', '<PATH>', msg)
    msg = re.sub(r'(?:[A-Z]:)?(?:\\[\w.-]+)+\\?', '<PATH>', msg)
    
    # Remove memory addresses
    msg = re.sub(r'\b[0-9a-f]{8,16}\b', '<ADDR>', msg)
    
    # Remove user IDs / session IDs
    msg = re.sub(r'\b(user|session|request|trace)[-_]?id[=:]?\s*\w+', 
                 r'\1_id=<ID>', msg)
    
    # Remove specific numeric values in JSON-like structures
    msg = re.sub(r':\s*\d+', ':<NUM>', msg)
    msg = re.sub(r'=\s*\d+', '=<NUM>', msg)
    
    # Remove quotes content but keep structure
    msg = re.sub(r'"[^"]{20,}"', '"<STRING>"', msg)
    msg = re.sub(r"'[^']{20,}'", "'<STRING>'", msg)
    
    # Normalize whitespace
    msg = re.sub(r'\s+', ' ', msg)
    msg = msg.strip()
    
    return msg


def extract_error_signature(
    message: str,
    error_type: str,
    service: str,
    exception_type: Optional[str] = None,
) -> str:
    """
    Extract a signature that uniquely identifies this error pattern.
    Similar errors should have the same signature.
    """
    # Start with normalized message
    normalized = normalize_message(message)
    
    # Truncate very long messages
    if len(normalized) > 200:
        normalized = normalized[:200] + "..."
    
    # Build signature components
    components = [
        f"service:{service}",
        f"type:{error_type}",
    ]
    
    if exception_type:
        components.append(f"exception:{exception_type}")
    
    components.append(f"msg:{normalized}")
    
    # Create hash for signature
    signature_str = "|".join(components)
    sig_hash = hashlib.md5(signature_str.encode()).hexdigest()[:12]
    
    return sig_hash


def create_message_template(messages: List[str]) -> str:
    """
    Create a template from multiple similar messages.
    Shows the common structure with placeholders for variable parts.
    """
    if not messages:
        return ""
    
    if len(messages) == 1:
        return normalize_message(messages[0])
    
    # For simplicity, use the normalized version of the most common message
    normalized = [normalize_message(m) for m in messages]
    
    # Find most common
    from collections import Counter
    counter = Counter(normalized)
    template = counter.most_common(1)[0][0]
    
    return template


# ===========================================================================
# Aggregation Strategies
# ===========================================================================

def aggregate_by_signature(errors: List[dict]) -> Dict[str, ErrorPattern]:
    """
    Aggregate errors by signature (service + error_type + normalized message).
    This is the most accurate deduplication method.
    """
    patterns: Dict[str, ErrorPattern] = {}
    
    for error in errors:
        # Extract signature
        signature = extract_error_signature(
            message=error['message'],
            error_type=error['error_type'],
            service=error['service'],
            exception_type=error.get('exception_type'),
        )
        
        # Get or create pattern
        if signature not in patterns:
            patterns[signature] = ErrorPattern(
                pattern_id=signature,
                error_type=error['error_type'],
                service=error['service'],
                signature=signature,
                message_template="",
                representative_message=error['message'],
                first_seen=error['timestamp'],
                last_seen=error['timestamp'],
            )
        
        pattern = patterns[signature]
        
        # Update statistics
        pattern.count += 1
        pattern.services_affected.add(error['service'])
        pattern.last_seen = max(pattern.last_seen, error['timestamp'])
        pattern.first_seen = min(pattern.first_seen, error['timestamp'])
        
        # Store sample messages (up to 5)
        if len(pattern.sample_messages) < 5:
            pattern.sample_messages.append(error['message'])
        
        # Traceback info
        if error.get('has_traceback'):
            pattern.has_traceback = True
            if error.get('traceback_lines'):
                pattern.common_traceback_lines.extend(error['traceback_lines'][:3])
        
        if error.get('exception_type'):
            pattern.exception_types.add(error['exception_type'])
    
    # Generate message templates
    for pattern in patterns.values():
        pattern.message_template = create_message_template(pattern.sample_messages)
    
    return patterns


def aggregate_by_service_and_type(errors: List[dict]) -> Dict[str, ErrorPattern]:
    """
    Aggregate errors by service and error type only.
    This is more aggressive - groups all errors of the same type in a service.
    """
    patterns: Dict[str, ErrorPattern] = {}
    
    for error in errors:
        # Simple key: service + error_type
        key = f"{error['service']}:{error['error_type']}"
        
        if key not in patterns:
            patterns[key] = ErrorPattern(
                pattern_id=key,
                error_type=error['error_type'],
                service=error['service'],
                signature=key,
                message_template="",
                representative_message=error['message'],
                first_seen=error['timestamp'],
                last_seen=error['timestamp'],
            )
        
        pattern = patterns[key]
        pattern.count += 1
        pattern.services_affected.add(error['service'])
        pattern.last_seen = max(pattern.last_seen, error['timestamp'])
        pattern.first_seen = min(pattern.first_seen, error['timestamp'])
        
        if len(pattern.sample_messages) < 10:
            pattern.sample_messages.append(error['message'])
        
        if error.get('has_traceback'):
            pattern.has_traceback = True
        if error.get('exception_type'):
            pattern.exception_types.add(error['exception_type'])
    
    for pattern in patterns.values():
        pattern.message_template = create_message_template(pattern.sample_messages)
    
    return patterns


def aggregate_by_message_similarity(
    errors: List[dict],
    threshold: float = 0.8
) -> Dict[str, ErrorPattern]:
    """
    Aggregate errors by message similarity using token-based matching.
    Groups messages that are at least `threshold` similar.
    """
    def tokenize(message: str) -> Set[str]:
        """Extract tokens from message."""
        msg = normalize_message(message)
        tokens = re.findall(r'\b\w+\b', msg.lower())
        return set(tokens)
    
    def similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
        """Calculate Jaccard similarity."""
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0
    
    patterns: Dict[str, ErrorPattern] = {}
    pattern_tokens: Dict[str, Set[str]] = {}
    
    for error in errors:
        error_tokens = tokenize(error['message'])
        
        # Find matching pattern
        matched_key = None
        best_similarity = 0.0
        
        for key, pattern in patterns.items():
            # Only match within same service and error type
            if (pattern.service == error['service'] and 
                pattern.error_type == error['error_type']):
                
                sim = similarity(error_tokens, pattern_tokens[key])
                if sim >= threshold and sim > best_similarity:
                    best_similarity = sim
                    matched_key = key
        
        # Create new pattern or update existing
        if matched_key is None:
            # New pattern
            key = f"{error['service']}:{error['error_type']}:{len(patterns)}"
            patterns[key] = ErrorPattern(
                pattern_id=key,
                error_type=error['error_type'],
                service=error['service'],
                signature=key,
                message_template="",
                representative_message=error['message'],
                first_seen=error['timestamp'],
                last_seen=error['timestamp'],
            )
            pattern_tokens[key] = error_tokens
        else:
            key = matched_key
        
        # Update pattern
        pattern = patterns[key]
        pattern.count += 1
        pattern.services_affected.add(error['service'])
        pattern.last_seen = max(pattern.last_seen, error['timestamp'])
        pattern.first_seen = min(pattern.first_seen, error['timestamp'])
        
        if len(pattern.sample_messages) < 5:
            pattern.sample_messages.append(error['message'])
        
        if error.get('has_traceback'):
            pattern.has_traceback = True
        if error.get('exception_type'):
            pattern.exception_types.add(error['exception_type'])
    
    for pattern in patterns.values():
        pattern.message_template = create_message_template(pattern.sample_messages)
    
    return patterns


# ===========================================================================
# Main Processing
# ===========================================================================

def load_errors_from_csv(csv_path: str) -> List[dict]:
    """Load errors from CSV and detect them."""
    print(f"[1] Loading errors from CSV: {csv_path}")
    
    # Import find_errors module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from find_errors import find_errors_in_logs
    
    errors, summary = find_errors_in_logs(csv_path)
    
    print(f"    Found {len(errors)} errors")
    
    # Convert to dicts
    return [e.to_dict() for e in errors]


def load_errors_from_json(json_path: str) -> List[dict]:
    """Load errors from JSON file (output of find_errors.py)."""
    print(f"[1] Loading errors from JSON: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    errors = data.get('errors', [])
    print(f"    Loaded {len(errors)} errors")
    
    return errors


def aggregate_errors(
    errors: List[dict],
    strategy: str = 'signature',
    similarity_threshold: float = 0.8,
) -> Tuple[Dict[str, ErrorPattern], AggregationStats]:
    """
    Aggregate errors to find unique patterns.
    
    Args:
        errors: List of error dictionaries
        strategy: Aggregation strategy ('signature', 'service-type', 'similarity')
        similarity_threshold: Threshold for similarity-based aggregation
    
    Returns:
        (patterns_dict, stats)
    """
    print(f"\n[2] Aggregating errors using strategy: {strategy}")
    
    # Apply aggregation strategy
    if strategy == 'signature':
        patterns = aggregate_by_signature(errors)
    elif strategy == 'service-type':
        patterns = aggregate_by_service_and_type(errors)
    elif strategy == 'similarity':
        patterns = aggregate_by_message_similarity(errors, similarity_threshold)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Calculate statistics
    stats = AggregationStats(
        total_errors=len(errors),
        unique_patterns=len(patterns),
        services_affected=len(set(e['service'] for e in errors)),
        deduplication_ratio=(len(errors) - len(patterns)) / len(errors) if errors else 0,
    )
    
    # Top patterns by frequency
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1].count, reverse=True)
    stats.top_patterns = [(p.pattern_id, p.count) for _, p in sorted_patterns[:10]]
    
    print(f"    Unique patterns: {stats.unique_patterns}")
    print(f"    Deduplication: {stats.deduplication_ratio:.1%}")
    
    return patterns, stats


# ===========================================================================
# Export Functions
# ===========================================================================

def export_to_json(
    patterns: Dict[str, ErrorPattern],
    stats: AggregationStats,
    output_path: str
):
    """Export unique patterns to JSON."""
    # Sort by frequency
    sorted_patterns = sorted(patterns.values(), key=lambda p: p.count, reverse=True)
    
    data = {
        'statistics': stats.to_dict(),
        'unique_patterns': [p.to_dict() for p in sorted_patterns],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[Export] JSON saved to: {output_path}")


def export_to_csv(patterns: Dict[str, ErrorPattern], output_path: str):
    """Export unique patterns to CSV."""
    rows = []
    
    for pattern in sorted(patterns.values(), key=lambda p: p.count, reverse=True):
        rows.append({
            'pattern_id': pattern.pattern_id,
            'error_type': pattern.error_type,
            'service': pattern.service,
            'count': pattern.count,
            'services_affected': ','.join(pattern.services_affected),
            'message_template': pattern.message_template[:200],
            'has_traceback': pattern.has_traceback,
            'exception_types': ','.join(pattern.exception_types) if pattern.exception_types else '',
            'first_seen': pattern.first_seen,
            'last_seen': pattern.last_seen,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"[Export] CSV saved to: {output_path}")


def print_summary(patterns: Dict[str, ErrorPattern], stats: AggregationStats):
    """Print summary to console."""
    print("\n" + "=" * 80)
    print("UNIQUE ERROR PATTERNS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Errors Analyzed: {stats.total_errors}")
    print(f"Unique Error Patterns: {stats.unique_patterns}")
    print(f"Services Affected: {stats.services_affected}")
    print(f"Deduplication Ratio: {stats.deduplication_ratio:.1%}")
    print(f"  (Reduced {stats.total_errors} errors to {stats.unique_patterns} unique patterns)")
    
    # Top patterns
    print("\n" + "-" * 80)
    print("TOP ERROR PATTERNS (by frequency):")
    
    sorted_patterns = sorted(patterns.values(), key=lambda p: p.count, reverse=True)
    
    for i, pattern in enumerate(sorted_patterns[:15], 1):
        pct = (pattern.count / stats.total_errors) * 100
        bar = "█" * int(pct / 2)
        
        print(f"\n{i}. {pattern.error_type} in {pattern.service}")
        print(f"   Count: {pattern.count} ({pct:.1f}%) {bar}")
        print(f"   Pattern ID: {pattern.pattern_id}")
        print(f"   Template: {pattern.message_template[:100]}...")
        
        if pattern.exception_types:
            print(f"   Exceptions: {', '.join(list(pattern.exception_types)[:3])}")
        if pattern.has_traceback:
            print(f"   Has traceback: Yes")
    
    print("\n" + "=" * 80 + "\n")


# ===========================================================================
# CLI Interface
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and deduplicate errors to find unique patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Aggregation Strategies:
  signature     - Group by normalized message signature (most accurate)
  service-type  - Group by service and error type (most aggressive)
  similarity    - Group by message similarity (adjustable threshold)

Examples:
  # Aggregate from CSV
  python aggregate_errors.py --csv logs.csv
  
  # Aggregate from JSON (output of find_errors.py)
  python aggregate_errors.py --json errors.json
  
  # Use similarity-based aggregation
  python aggregate_errors.py --csv logs.csv --strategy similarity --threshold 0.8
  
  # Export to all formats
  python aggregate_errors.py --csv logs.csv --export-all
        """
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv', help='Path to CSV log file')
    input_group.add_argument('--json', help='Path to JSON error file')
    
    # Aggregation options
    parser.add_argument('--strategy', 
                       choices=['signature', 'service-type', 'similarity'],
                       default='signature',
                       help='Aggregation strategy (default: signature)')
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='Similarity threshold for similarity strategy (default: 0.8)')
    
    # Output
    parser.add_argument('--export-json', 
                       help='Export unique patterns to JSON')
    parser.add_argument('--export-csv',
                       help='Export unique patterns to CSV')
    parser.add_argument('--export-all', action='store_true',
                       help='Export to all formats (unique_errors.json, unique_errors.csv)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    # Load errors
    if args.csv:
        errors = load_errors_from_csv(args.csv)
    else:
        errors = load_errors_from_json(args.json)
    
    if not errors:
        print("No errors found to aggregate.")
        return
    
    # Aggregate
    patterns, stats = aggregate_errors(
        errors,
        strategy=args.strategy,
        similarity_threshold=args.threshold,
    )
    
    # Print summary
    if not args.quiet:
        print_summary(patterns, stats)
    
    # Export
    if args.export_all:
        export_to_json(patterns, stats, "unique_errors.json")
        export_to_csv(patterns, "unique_errors.csv")
    else:
        if args.export_json:
            export_to_json(patterns, stats, args.export_json)
        if args.export_csv:
            export_to_csv(patterns, args.export_csv)


if __name__ == "__main__":
    main()