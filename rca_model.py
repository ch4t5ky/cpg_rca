#!/usr/bin/env python3
"""
local_llm_rca.py
================
Root Cause Analysis using LOCAL LLM (no external API)

Uses a local transformer model (loaded with torch) to analyze the graph
and rank services by failure likelihood.

Supports:
- Hugging Face models (LLaMA, Mistral, etc.)
- Custom fine-tuned models
- Offline inference (no API calls)

Usage:
    # Download model first (one-time)
    python local_llm_rca.py --download-model --model-name mistralai/Mistral-7B-Instruct-v0.2
    
    # Run RCA with local model
    python local_llm_rca.py --cache-dir cache --model-path ./models/mistral-7b
    
    # Use custom model
    python local_llm_rca.py --cache-dir cache --model-path ./my_custom_rca_model

Output:
    Ranked services by failure likelihood (100% local, no API)
"""

import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)


def load_cache(cache_dir: str) -> Dict:
    """Load unified cache."""
    cache_path = Path(cache_dir)
    
    print(f"[1] Loading cache from: {cache_dir}")
    
    with open(cache_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    with open(cache_path / "method_calls.pkl", 'rb') as f:
        method_calls = pickle.load(f)
    
    with open(cache_path / "call_edges.pkl", 'rb') as f:
        edges = pickle.load(f)
    
    with open(cache_path / "lifecycle_events.pkl", 'rb') as f:
        lifecycle = pickle.load(f)
    
    print(f"  ✓ Method calls: {len(method_calls)}")
    print(f"  ✓ Call edges: {len(edges)}")
    print(f"  ✓ Lifecycle events: {len(lifecycle)}")
    
    return {
        'metadata': metadata,
        'method_calls': method_calls,
        'edges': edges,
        'lifecycle': lifecycle
    }


def build_graph_structure(data: Dict, window: int = 10) -> Dict:
    """Build graph structure for analysis."""
    print(f"\n[2] Building graph structure...")
    
    method_calls = data['method_calls']
    edges = data['edges']
    lifecycle = data['lifecycle']
    
    # Find errors
    errors = [m for m in method_calls if m['is_error']]
    
    if not errors:
        print("  No errors found")
        return None
    
    # Focus on time window around first error
    first_error_time = min(e['timestamp'] for e in errors)
    start_time = first_error_time - window
    end_time = first_error_time + window
    
    window_calls = [m for m in method_calls 
                    if start_time <= m['timestamp'] <= end_time]
    
    print(f"  ✓ Window: {len(window_calls)} calls in {window*2}s")
    
    # Build nodes (unique methods)
    nodes = {}
    for call in window_calls:
        qn = call['qualified_name']
        if qn not in nodes:
            nodes[qn] = {
                'service': call['service'],
                'method': call['method'],
                'call_count': 0,
                'error_count': 0,
                'first_seen': call['timestamp'],
                'last_seen': call['timestamp']
            }
        
        nodes[qn]['call_count'] += 1
        if call['is_error']:
            nodes[qn]['error_count'] += 1
        nodes[qn]['last_seen'] = max(nodes[qn]['last_seen'], call['timestamp'])
    
    # Build edges
    edge_map = defaultdict(lambda: {'count': 0, 'cross_service': False})
    
    for caller, callee, kind in edges:
        if kind == "CALL" and caller in nodes and callee in nodes:
            edge_key = (caller, callee)
            edge_map[edge_key]['count'] += 1
            
            caller_svc = caller.split('::')[0]
            callee_svc = callee.split('::')[0]
            if caller_svc != callee_svc:
                edge_map[edge_key]['cross_service'] = True
    
    # Lifecycle in window
    lifecycle_in_window = [
        lc for lc in lifecycle
        if start_time <= lc['timestamp'] <= end_time
    ]
    
    # Get services
    services = sorted(set(node['service'] for node in nodes.values()))
    
    graph = {
        'time_range': {
            'start': datetime.fromtimestamp(start_time).isoformat(),
            'end': datetime.fromtimestamp(end_time).isoformat(),
            'duration_seconds': end_time - start_time
        },
        'services': services,
        'nodes': nodes,
        'edges': [
            {
                'caller': caller,
                'callee': callee,
                'count': data['count'],
                'cross_service': data['cross_service']
            }
            for (caller, callee), data in edge_map.items()
        ],
        'lifecycle_events': [
            {
                'time': datetime.fromtimestamp(lc['timestamp']).isoformat(),
                'service': lc['service'],
                'type': lc['type'],
                'message': lc['message']
            }
            for lc in lifecycle_in_window
        ],
        'error_summary': {
            'total_errors': sum(1 for n in nodes.values() if n['error_count'] > 0),
            'affected_services': list(set(n['service'] for n in nodes.values() if n['error_count'] > 0)),
        }
    }
    
    print(f"  ✓ Graph: {len(nodes)} nodes, {len(edge_map)} edges")
    print(f"  ✓ Services: {len(services)}")
    
    return graph


def create_prompt(graph: Dict) -> str:
    """Create prompt for local LLM."""
    
    # Build compact summary
    services = graph['services']
    
    prompt = f"""<|system|>
You are a microservices root cause analysis expert. Analyze the failure and rank services by likelihood of being the root cause.
<|end|>
<|user|>
Analyze this microservices failure:

TIME: {graph['time_range']['start']} to {graph['time_range']['end']} ({graph['time_range']['duration_seconds']:.1f}s)
SERVICES: {', '.join(services)}
ERRORS: {graph['error_summary']['total_errors']} methods with errors
AFFECTED: {', '.join(graph['error_summary']['affected_services'])}

"""
    
    # Add lifecycle events (most important signal)
    if graph['lifecycle_events']:
        prompt += "\nLIFECYCLE EVENTS:\n"
        for lc in graph['lifecycle_events']:
            prompt += f"- {lc['time']}: {lc['service']} {lc['type'].upper()} - {lc['message']}\n"
    
    # Add error summary per service
    prompt += "\nSERVICE DETAILS:\n"
    service_stats = defaultdict(lambda: {'calls': 0, 'errors': 0})
    
    for node in graph['nodes'].values():
        svc = node['service']
        service_stats[svc]['calls'] += node['call_count']
        service_stats[svc]['errors'] += node['error_count']
    
    for svc in services:
        stats = service_stats[svc]
        error_rate = stats['errors'] / stats['calls'] if stats['calls'] > 0 else 0
        prompt += f"- {svc}: {stats['calls']} calls, {stats['errors']} errors ({error_rate:.1%})\n"
    
    # Add cross-service calls
    cross_service_edges = [e for e in graph['edges'] if e['cross_service']]
    if cross_service_edges:
        prompt += "\nCROSS-SERVICE CALLS:\n"
        for edge in cross_service_edges[:10]:  # Limit to top 10
            caller_svc = edge['caller'].split('::')[0]
            callee_svc = edge['callee'].split('::')[0]
            prompt += f"- {caller_svc} → {callee_svc} ({edge['count']} calls)\n"
    
    # Add instruction
    prompt += f"""
Task: Rank ALL services ({', '.join(services)}) by likelihood of being the root cause.

Provide ONLY this format (no other text):

RANKING:
1. [service] - [confidence: HIGH/MEDIUM/LOW] - [reason]
2. [service] - [confidence: HIGH/MEDIUM/LOW] - [reason]
...

ROOT_CAUSE: [service_name]
EXPLANATION: [1-2 sentences why]
<|end|>
<|assistant|>
"""
    
    return prompt


def load_local_model(model_path: str, use_4bit: bool = True):
    """Load local LLM model."""
    print(f"\n[3] Loading local model: {model_path}")
    
    # Check if path exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"  ❌ Model not found at: {model_path}")
        print(f"  Run with --download-model first")
        return None, None
    
    # Quantization config for lower memory usage
    if use_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print(f"  Using 4-bit quantization")
    else:
        quantization_config = None
        print(f"  Using full precision")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    print(f"  ✓ Model loaded")
    print(f"  Device: {model.device}")
    
    return model, tokenizer


def analyze_with_local_llm(graph: Dict, model, tokenizer, max_length: int = 512) -> str:
    """Analyze graph with local LLM."""
    
    print(f"\n[4] Analyzing with local LLM...")
    
    # Create prompt
    prompt = create_prompt(graph)
    
    print(f"  Prompt length: {len(prompt)} chars")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"  Generating response...")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (after the prompt)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    print(f"  ✓ Generated {len(response)} chars")
    
    return response


def download_model(model_name: str, output_dir: str = "./models"):
    """Download model from Hugging Face."""
    print(f"Downloading model: {model_name}")
    
    output_path = Path(output_dir) / model_name.split('/')[-1]
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download tokenizer
    print("  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))
    
    # Download model
    print("  Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.save_pretrained(str(output_path))
    
    print(f"  ✓ Model saved to: {output_path}")
    print(f"\n  Use with: --model-path {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Root Cause Analysis with Local LLM (no API)"
    )
    
    parser.add_argument('--cache-dir', default='cache', help='Cache directory')
    parser.add_argument('--model-path', help='Path to local model directory')
    parser.add_argument('--window', type=int, default=10, help='Time window (seconds)')
    parser.add_argument('--output', help='Save analysis to file')
    parser.add_argument('--max-length', type=int, default=512, help='Max response tokens')
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')
    
    # Model download
    parser.add_argument('--download-model', action='store_true', help='Download model from HF')
    parser.add_argument('--model-name', default='mistralai/Mistral-7B-Instruct-v0.2',
                       help='HuggingFace model name')
    parser.add_argument('--models-dir', default='./models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Download mode
    if args.download_model:
        download_model(args.model_name, args.models_dir)
        return
    
    # Analysis mode
    if not args.model_path:
        print("Error: Provide --model-path or use --download-model first")
        return
    
    print("=" * 70)
    print("LOCAL LLM ROOT CAUSE ANALYSIS")
    print("=" * 70)
    
    # Load cache
    data = load_cache(args.cache_dir)
    
    # Build graph
    graph = build_graph_structure(data, args.window)
    
    if not graph:
        print("\n❌ Could not build graph")
        return
    
    # Load model
    model, tokenizer = load_local_model(args.model_path, use_4bit=not args.no_4bit)
    
    if model is None:
        return
    
    # Analyze
    analysis = analyze_with_local_llm(graph, model, tokenizer, args.max_length)
    
    # Display
    print("\n" + "=" * 70)
    print("ANALYSIS RESULT")
    print("=" * 70)
    print(analysis)
    print("=" * 70)
    
    # Save if requested
    if args.output:
        result = {
            'timestamp': datetime.now().isoformat(),
            'cache_dir': args.cache_dir,
            'model_path': args.model_path,
            'graph': graph,
            'analysis': analysis
        }
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Saved to: {args.output}")


if __name__ == "__main__":
    main()