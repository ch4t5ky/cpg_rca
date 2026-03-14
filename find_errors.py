#!/usr/bin/env python3
"""
find_errors.py
==============
Find and extract logs with errors, exceptions, and tracebacks.

Features:
- Multi-pattern error detection
- Stack trace extraction
- Exception grouping
- Export to multiple formats (JSON, CSV, text)
- Real-time filtering

Usage:
    python find_errors.py --csv logs.csv
    python find_errors.py --csv logs.csv --error-type exception
    python find_errors.py --csv logs.csv --service frontend
    python find_errors.py --csv logs.csv --export-json errors.json
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ===========================================================================
# Error Pattern Definitions
# ===========================================================================

ERROR_PATTERNS = {
    # Critical errors
    'EXCEPTION': [
        r'exception',
        r'Exception:',
        r'Error:',
        r'Traceback \(most recent call last\)',
    ],
    'ERROR': [
        r'\berror\b',
        r'ERROR:',
        r'\[error\]',
        r'\[ERROR\]',
    ],
    'FAILED': [
        r'\bfailed\b',
        r'\bfailure\b',
        r'has failed',
        r'failed to',
    ],
    'FATAL': [
        r'\bfatal\b',
        r'FATAL:',
        r'\[FATAL\]',
        r'critical error',
    ],
    'CRASH': [
        r'\bcrash(ed)?\b',
        r'segmentation fault',
        r'core dumped',
    ],
    
    # Connection/Network errors
    'TIMEOUT': [
        r'\btimeout\b',
        r'timed out',
        r'time out',
        r'deadline exceeded',
    ],
    'CONNECTION_ERROR': [
        r'connection (refused|reset|closed)',
        r'unable to connect',
        r'connection error',
        r'network unreachable',
    ],
    
    # Access/Permission errors
    'DENIED': [
        r'(access |permission )?denied',
        r'forbidden',
        r'unauthorized',
        r'authentication failed',
    ],
    
    # Resource errors
    'OUT_OF_MEMORY': [
        r'out of memory',
        r'memory error',
        r'OOM',
        r'cannot allocate memory',
    ],
    'DISK_FULL': [
        r'disk full',
        r'no space left',
        r'quota exceeded',
    ],
    
    # Application errors
    'NULL_POINTER': [
        r'null pointer',
        r'NullPointerException',
        r'nil pointer',
        r'null reference',
    ],
    'INVALID': [
        r'\binvalid\b',
        r'validation failed',
        r'invalid input',
    ],
    'NOT_FOUND': [
        r'not found',
        r'does not exist',
        r'no such file',
        r'(?<![\d:])\b404\b(?![\d"])',  # 404 not surrounded by digits or quotes
    ],
    
    # Warnings (lower severity)
    'WARNING': [
        r'\bwarning\b',
        r'WARN:',
        r'\[WARN\]',
        r'\[WARNING\]',
    ],
    'DEPRECATED': [
        r'deprecated',
        r'deprecation warning',
    ],
}

# Compile patterns for performance
COMPILED_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in ERROR_PATTERNS.items()
}

# Stack trace patterns
TRACEBACK_PATTERNS = [
    re.compile(r'Traceback \(most recent call last\):', re.IGNORECASE),
    re.compile(r'^\s+at\s+\w+', re.MULTILINE),  # Java-style
    re.compile(r'^\s+File\s+"[^"]+",\s+line\s+\d+', re.MULTILINE),  # Python
    re.compile(r'^\s+\w+\.\w+\([^)]*\)', re.MULTILINE),  # General
]


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class ErrorLog:
    """Represents a single error log entry."""
    timestamp: float
    timestamp_str: str
    service: str
    message: str
    error_type: str
    severity: int = 5  # 1-10 scale
    has_traceback: bool = False
    traceback_lines: List[str] = field(default_factory=list)
    exception_type: Optional[str] = None
    error_location: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'timestamp_str': self.timestamp_str,
            'service': self.service,
            'message': self.message,
            'error_type': self.error_type,
            'severity': self.severity,
            'has_traceback': self.has_traceback,
            'traceback_lines': self.traceback_lines,
            'exception_type': self.exception_type,
            'error_location': self.error_location,
        }


@dataclass
class ErrorSummary:
    """Summary statistics for error analysis."""
    total_errors: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_service: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_with_traceback: int = 0
    unique_exceptions: Set[str] = field(default_factory=set)
    time_range: Tuple[float, float] = (0, 0)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_errors': self.total_errors,
            'errors_by_type': dict(self.errors_by_type),
            'errors_by_service': dict(self.errors_by_service),
            'errors_with_traceback': self.errors_with_traceback,
            'unique_exceptions': list(self.unique_exceptions),
            'time_range': {
                'start': self.time_range[0],
                'end': self.time_range[1],
                'duration_seconds': self.time_range[1] - self.time_range[0],
            }
        }


# ===========================================================================
# Error Detection Functions
# ===========================================================================

def detect_error_type(message: str) -> Tuple[Optional[str], int]:
    """
    Detect error type from message.
    
    Returns:
        (error_type, severity) or (None, 0) if no error detected
    """
    message_lower = message.lower()
    
    # Check each category in priority order
    severity_map = {
        'FATAL': 10,
        'CRASH': 10,
        'EXCEPTION': 9,
        'OUT_OF_MEMORY': 9,
        'ERROR': 8,
        'FAILED': 7,
        'TIMEOUT': 7,
        'CONNECTION_ERROR': 7,
        'DENIED': 7,
        'DISK_FULL': 8,
        'NULL_POINTER': 8,
        'INVALID': 6,
        'NOT_FOUND': 5,
        'WARNING': 3,
        'DEPRECATED': 2,
    }
    
    for category, patterns in COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(message):
                return category, severity_map.get(category, 5)
    
    return None, 0


def extract_exception_type(message: str) -> Optional[str]:
    """Extract exception type from error message."""
    # Common exception patterns
    patterns = [
        r'(\w+Exception):',
        r'(\w+Error):',
        r'raise (\w+Exception)',
        r'throw new (\w+Exception)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    
    return None


def extract_error_location(message: str) -> Optional[str]:
    """Extract file/line information from error message."""
    # Python-style: File "path/file.py", line 123
    match = re.search(r'File\s+"([^"]+)",\s+line\s+(\d+)', message)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    
    # Java-style: at package.Class.method(File.java:123)
    match = re.search(r'at\s+[\w.]+\((\w+\.java):(\d+)\)', message)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    
    # Generic function/method reference
    match = re.search(r'in\s+(\w+)\s+at\s+line\s+(\d+)', message)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    
    return None


def has_traceback(message: str) -> bool:
    """Check if message contains a stack trace."""
    for pattern in TRACEBACK_PATTERNS:
        if pattern.search(message):
            return True
    return False


def extract_traceback_lines(message: str) -> List[str]:
    """Extract individual lines from a stack trace."""
    if not has_traceback(message):
        return []
    
    lines = message.split('\n')
    traceback_lines = []
    
    in_traceback = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Start of traceback
        if any(p.search(line) for p in TRACEBACK_PATTERNS[:1]):
            in_traceback = True
            traceback_lines.append(line)
            continue
        
        # Traceback continuation
        if in_traceback:
            # Python File/line pattern
            if re.match(r'File\s+"[^"]+",\s+line\s+\d+', line):
                traceback_lines.append(line)
            # Java at pattern
            elif re.match(r'at\s+\w+', line):
                traceback_lines.append(line)
            # Indented code line
            elif line.startswith(' ') or line.startswith('\t'):
                traceback_lines.append(line)
            # Exception line
            elif ':' in line and ('Exception' in line or 'Error' in line):
                traceback_lines.append(line)
                in_traceback = False  # End of traceback
    
    return traceback_lines[:50]  # Limit to 50 lines


# ===========================================================================
# Main Processing Functions
# ===========================================================================

def find_errors_in_logs(
    csv_path: str,
    service_filter: Optional[str] = None,
    error_type_filter: Optional[str] = None,
    min_severity: int = 1,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> Tuple[List[ErrorLog], ErrorSummary]:
    """
    Scan logs and extract all error entries.
    
    Args:
        csv_path: Path to CSV log file
        service_filter: Only include specific service (optional)
        error_type_filter: Only include specific error type (optional)
        min_severity: Minimum severity threshold (1-10)
        start_ts: Start timestamp filter
        end_ts: End timestamp filter
    
    Returns:
        (error_logs, summary)
    """
    print(f"[1] Loading logs from: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Data validation and conversion
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df["container_name"] = df["container_name"].fillna("unknown").astype(str).str.strip()
    df["message"] = df["message"].fillna("").astype(str)
    
    # Apply filters
    if start_ts:
        df = df[df["timestamp"] >= start_ts]
    if end_ts:
        df = df[df["timestamp"] <= end_ts]
    if service_filter:
        df = df[df["container_name"] == service_filter]
    
    print(f"    Scanning {len(df)} log entries...")
    
    # Process logs
    error_logs: List[ErrorLog] = []
    summary = ErrorSummary()
    
    for idx, row in df.iterrows():
        message = str(row['message'])
        service = str(row['container_name'])
        timestamp = float(row['timestamp'])
        
        # Detect error
        error_type, severity = detect_error_type(message)
        
        if error_type is None:
            continue  # Not an error
        
        if severity < min_severity:
            continue  # Below threshold
        
        if error_type_filter and error_type != error_type_filter.upper():
            continue  # Doesn't match filter
        
        # Extract additional info
        has_trace = has_traceback(message)
        traceback = extract_traceback_lines(message) if has_trace else []
        exception_type = extract_exception_type(message)
        location = extract_error_location(message)
        
        # Create error log entry
        error_log = ErrorLog(
            timestamp=timestamp,
            timestamp_str=datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            service=service,
            message=message,
            error_type=error_type,
            severity=severity,
            has_traceback=has_trace,
            traceback_lines=traceback,
            exception_type=exception_type,
            error_location=location,
        )
        
        error_logs.append(error_log)
        
        # Update summary
        summary.total_errors += 1
        summary.errors_by_type[error_type] += 1
        summary.errors_by_service[service] += 1
        if has_trace:
            summary.errors_with_traceback += 1
        if exception_type:
            summary.unique_exceptions.add(exception_type)
    
    # Finalize summary
    if error_logs:
        summary.time_range = (
            min(e.timestamp for e in error_logs),
            max(e.timestamp for e in error_logs),
        )
    
    print(f"[2] Found {len(error_logs)} error entries")
    print(f"    With traceback: {summary.errors_with_traceback}")
    print(f"    Unique exceptions: {len(summary.unique_exceptions)}")
    
    return error_logs, summary


# ===========================================================================
# Export Functions
# ===========================================================================

def export_to_json(errors: List[ErrorLog], summary: ErrorSummary, output_path: str):
    """Export errors to JSON format."""
    data = {
        'summary': summary.to_dict(),
        'errors': [e.to_dict() for e in errors],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[Export] JSON saved to: {output_path}")


def export_to_csv(errors: List[ErrorLog], output_path: str):
    """Export errors to CSV format."""
    # Flatten for CSV
    rows = []
    for error in errors:
        rows.append({
            'timestamp': error.timestamp,
            'timestamp_str': error.timestamp_str,
            'service': error.service,
            'error_type': error.error_type,
            'severity': error.severity,
            'has_traceback': error.has_traceback,
            'exception_type': error.exception_type or '',
            'error_location': error.error_location or '',
            'message': error.message[:200],  # Truncate for CSV
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"[Export] CSV saved to: {output_path}")


def export_to_text(errors: List[ErrorLog], summary: ErrorSummary, output_path: str):
    """Export errors to human-readable text format."""
    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("ERROR LOG REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Errors: {summary.total_errors}\n")
        f.write(f"Errors with Traceback: {summary.errors_with_traceback}\n")
        f.write(f"Unique Exception Types: {len(summary.unique_exceptions)}\n")
        
        if summary.time_range[0] > 0:
            start = datetime.fromtimestamp(summary.time_range[0]).strftime('%Y-%m-%d %H:%M:%S')
            end = datetime.fromtimestamp(summary.time_range[1]).strftime('%Y-%m-%d %H:%M:%S')
            duration = summary.time_range[1] - summary.time_range[0]
            f.write(f"Time Range: {start} to {end} ({duration:.0f} seconds)\n")
        
        f.write("\n")
        
        # Errors by type
        f.write("ERRORS BY TYPE\n")
        f.write("-" * 80 + "\n")
        for error_type, count in sorted(summary.errors_by_type.items(), 
                                       key=lambda x: x[1], reverse=True):
            pct = (count / summary.total_errors) * 100
            f.write(f"  {error_type:20s} : {count:5d} ({pct:5.1f}%)\n")
        f.write("\n")
        
        # Errors by service
        f.write("ERRORS BY SERVICE\n")
        f.write("-" * 80 + "\n")
        for service, count in sorted(summary.errors_by_service.items(),
                                    key=lambda x: x[1], reverse=True):
            pct = (count / summary.total_errors) * 100
            f.write(f"  {service:20s} : {count:5d} ({pct:5.1f}%)\n")
        f.write("\n")
        
        # Exception types
        if summary.unique_exceptions:
            f.write("UNIQUE EXCEPTION TYPES\n")
            f.write("-" * 80 + "\n")
            for exc in sorted(summary.unique_exceptions):
                f.write(f"  - {exc}\n")
            f.write("\n")
        
        # Detailed error list
        f.write("DETAILED ERROR LOG\n")
        f.write("=" * 80 + "\n\n")
        
        for i, error in enumerate(errors, 1):
            f.write(f"Error #{i}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Time:     {error.timestamp_str}\n")
            f.write(f"Service:  {error.service}\n")
            f.write(f"Type:     {error.error_type} (severity: {error.severity}/10)\n")
            
            if error.exception_type:
                f.write(f"Exception: {error.exception_type}\n")
            if error.error_location:
                f.write(f"Location: {error.error_location}\n")
            
            f.write(f"\nMessage:\n{error.message}\n")
            
            if error.has_traceback and error.traceback_lines:
                f.write(f"\nTraceback ({len(error.traceback_lines)} lines):\n")
                for line in error.traceback_lines:
                    f.write(f"  {line}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"[Export] Text report saved to: {output_path}")


def print_summary(errors: List[ErrorLog], summary: ErrorSummary):
    """Print summary to console."""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Errors Found: {summary.total_errors}")
    print(f"Errors with Traceback: {summary.errors_with_traceback}")
    print(f"Unique Exception Types: {len(summary.unique_exceptions)}")
    
    if summary.time_range[0] > 0:
        start = datetime.fromtimestamp(summary.time_range[0]).strftime('%Y-%m-%d %H:%M:%S')
        end = datetime.fromtimestamp(summary.time_range[1]).strftime('%Y-%m-%d %H:%M:%S')
        duration = summary.time_range[1] - summary.time_range[0]
        print(f"Time Range: {start} to {end}")
        print(f"Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
    
    # Top errors by type
    print("\n" + "-" * 80)
    print("TOP ERROR TYPES:")
    for error_type, count in sorted(summary.errors_by_type.items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
        pct = (count / summary.total_errors) * 100
        bar = "█" * int(pct / 2)
        print(f"  {error_type:20s} : {count:5d} ({pct:5.1f}%) {bar}")
    
    # Top services with errors
    print("\n" + "-" * 80)
    print("TOP SERVICES WITH ERRORS:")
    for service, count in sorted(summary.errors_by_service.items(),
                                key=lambda x: x[1], reverse=True)[:10]:
        pct = (count / summary.total_errors) * 100
        bar = "█" * int(pct / 2)
        print(f"  {service:20s} : {count:5d} ({pct:5.1f}%) {bar}")
    
    # Sample errors with tracebacks
    errors_with_tb = [e for e in errors if e.has_traceback]
    if errors_with_tb:
        print("\n" + "-" * 80)
        print("SAMPLE ERRORS WITH TRACEBACK:")
        for i, error in enumerate(errors_with_tb[:3], 1):
            print(f"\n{i}. [{error.error_type}] {error.service} at {error.timestamp_str}")
            print(f"   {error.message[:100]}...")
            if error.exception_type:
                print(f"   Exception: {error.exception_type}")
            if error.traceback_lines:
                print(f"   Traceback: {len(error.traceback_lines)} lines")
    
    print("\n" + "=" * 80 + "\n")


# ===========================================================================
# CLI Interface
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Find and extract error logs with tracebacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find all errors
  python find_errors.py --csv logs.csv
  
  # Find only exceptions with tracebacks
  python find_errors.py --csv logs.csv --error-type exception --with-traceback
  
  # Find errors in specific service
  python find_errors.py --csv logs.csv --service frontend
  
  # Export to JSON
  python find_errors.py --csv logs.csv --export-json errors.json
  
  # Export to all formats
  python find_errors.py --csv logs.csv --export-all
  
  # Time-bounded search
  python find_errors.py --csv logs.csv --start-ts 1731903975 --end-ts 1731904000
        """
    )
    
    # Input
    parser.add_argument('--csv', required=True,
                       help='Path to CSV log file')
    
    # Filters
    parser.add_argument('--service',
                       help='Filter by service name')
    parser.add_argument('--error-type',
                       help='Filter by error type (exception, error, failed, etc.)')
    parser.add_argument('--with-traceback', action='store_true',
                       help='Only show errors with stack traces')
    parser.add_argument('--min-severity', type=int, default=1,
                       help='Minimum severity (1-10, default: 1)')
    parser.add_argument('--start-ts', type=float,
                       help='Start timestamp (Unix epoch)')
    parser.add_argument('--end-ts', type=float,
                       help='End timestamp (Unix epoch)')
    
    # Output
    parser.add_argument('--export-json',
                       help='Export to JSON file')
    parser.add_argument('--export-csv',
                       help='Export to CSV file')
    parser.add_argument('--export-text',
                       help='Export to text report')
    parser.add_argument('--export-all', action='store_true',
                       help='Export to all formats (errors.json, errors.csv, errors.txt)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    # Find errors
    errors, summary = find_errors_in_logs(
        csv_path=args.csv,
        service_filter=args.service,
        error_type_filter=args.error_type,
        min_severity=args.min_severity,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
    )
    
    # Additional filter for traceback
    if args.with_traceback:
        errors = [e for e in errors if e.has_traceback]
        print(f"[Filter] Filtered to {len(errors)} errors with traceback")
    
    if not errors:
        print("\nNo errors found matching the criteria.")
        return
    
    # Print summary
    if not args.quiet:
        print_summary(errors, summary)
    
    # Export
    if args.export_all:
        export_to_json(errors, summary, "errors.json")
        export_to_csv(errors, "errors.csv")
        export_to_text(errors, summary, "errors.txt")
    else:
        if args.export_json:
            export_to_json(errors, summary, args.export_json)
        if args.export_csv:
            export_to_csv(errors, args.export_csv)
        if args.export_text:
            export_to_text(errors, summary, args.export_text)


if __name__ == "__main__":
    main()