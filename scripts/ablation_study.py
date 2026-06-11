#!/usr/bin/env python3
"""
Ablation Studies Framework for LSTM Language Model.

Tests the contribution of different architectural components:
- Attention mechanism (none vs bahdanau vs luong)
- Pretrained embeddings (random init vs GloVe)
- Weight tying
- Number of layers
- Hidden size
"""

import argparse
import json
import sys
import os
import subprocess
from typing import List, Dict, Any
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    # Model overrides
    attention_type: str = "bahdanau"
    use_glove: bool = False
    tie_weights: bool = False
    num_layers: int = 2
    hidden_size: int = 512


def get_ablation_variants() -> List[AblationConfig]:
    """Define standard ablation experiments."""
    variants = [
        AblationConfig(
            name="full_model",
            description="Full model with all features",
            attention_type="bahdanau",
            use_glove=True,
            tie_weights=True,
        ),
        AblationConfig(
            name="no_attention",
            description="Without attention mechanism",
            attention_type="none",
            use_glove=True,
            tie_weights=True,
        ),
        AblationConfig(
            name="no_glove",
            description="Without pretrained embeddings",
            attention_type="bahdanau",
            use_glove=False,
            tie_weights=True,
        ),
        AblationConfig(
            name="no_tied_weights",
            description="Without weight tying",
            attention_type="bahdanau",
            use_glove=True,
            tie_weights=False,
        ),
        AblationConfig(
            name="luong_attention",
            description="With Luong attention instead of Bahdanau",
            attention_type="luong",
            use_glove=True,
            tie_weights=True,
        ),
        AblationConfig(
            name="single_layer",
            description="Single layer LSTM",
            attention_type="bahdanau",
            use_glove=True,
            tie_weights=True,
            num_layers=1,
        ),
        AblationConfig(
            name="small_hidden",
            description="Smaller hidden size (256)",
            attention_type="bahdanau",
            use_glove=True,
            tie_weights=True,
            hidden_size=256,
        ),
        AblationConfig(
            name="baseline",
            description="Simple baseline (2-layer, 256 hidden, no attention, no GloVe)",
            attention_type="none",
            use_glove=False,
            tie_weights=False,
            hidden_size=256,
        ),
    ]
    return variants


def run_experiment(config: AblationConfig, train_path: str, valid_path: str, 
                   epochs: int, batch_size: int, limit_lines: int | None) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    
    output_dir = f"artifacts/ablation/{config.name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*60}")
    
    # Build training command
    cmd = [
        sys.executable, "scripts/train_lstm.py",
        "--train-path", train_path,
        "--valid-path", valid_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--hidden-size", str(config.hidden_size),
        "--num-layers", str(config.num_layers),
        "--attention-type", config.attention_type,
    ]
    
    if config.use_glove:
        cmd.append("--use-glove")
    if config.tie_weights:
        cmd.append("--tie-weights")
    if limit_lines:
        cmd.extend(["--limit-lines", str(limit_lines)])
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Load results
        results = {
            "name": config.name,
            "description": config.description,
            "config": {
                "attention_type": config.attention_type,
                "use_glove": config.use_glove,
                "tie_weights": config.tie_weights,
                "num_layers": config.num_layers,
                "hidden_size": config.hidden_size,
            },
            "success": True,
            "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
        }
        
        # Try to load training history
        history_path = "artifacts/lstm/training_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                results["best_val_ppl"] = min(history.get("val_ppl", [float('inf')]))
                results["final_val_ppl"] = history.get("val_ppl", [])[-1] if history.get("val_ppl") else None
        
        return results
        
    except subprocess.CalledProcessError as e:
        return {
            "name": config.name,
            "description": config.description,
            "config": {
                "attention_type": config.attention_type,
                "use_glove": config.use_glove,
                "tie_weights": config.tie_weights,
                "num_layers": config.num_layers,
                "hidden_size": config.hidden_size,
            },
            "success": False,
            "error": str(e),
            "stderr": e.stderr[-500:] if len(e.stderr) > 500 else e.stderr,
        }


def main():
    parser = argparse.ArgumentParser(description="Ablation Studies for LSTM")
    parser.add_argument("--train-path", default="data/wikitext-103/wiki.train.tokens")
    parser.add_argument("--valid-path", default="data/wikitext-103/wiki.valid.tokens")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit-lines", type=int, default=None)
    parser.add_argument("--variants", default="all", 
                       help="Comma-separated list of variant names, or 'all'")
    parser.add_argument("--out", default="artifacts/ablation/results.json")
    args = parser.parse_args()
    
    print("="*60)
    print("ABLATION STUDIES")
    print("="*60)
    print(f"Train: {args.train_path}")
    print(f"Valid: {args.valid_path}")
    print(f"Epochs per variant: {args.epochs}")
    
    # Get variants to run
    all_variants = get_ablation_variants()
    if args.variants == "all":
        variants = all_variants
    else:
        selected = set(args.variants.split(","))
        variants = [v for v in all_variants if v.name in selected]
    
    print(f"Variants: {[v.name for v in variants]}")
    print(f"Total experiments: {len(variants)}")
    
    # Run experiments
    results = []
    for variant in variants:
        result = run_experiment(
            variant, 
            args.train_path, 
            args.valid_path,
            args.epochs,
            args.batch_size,
            args.limit_lines
        )
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ {variant.name}: Best PPL = {result.get('best_val_ppl', 'N/A'):.2f}")
        else:
            print(f"  ✗ {variant.name}: Failed")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r["success"]]
    if successful:
        sorted_results = sorted(successful, key=lambda x: x.get("best_val_ppl", float('inf')))
        print("\nRanking by validation perplexity:")
        for i, r in enumerate(sorted_results, 1):
            ppl = r.get("best_val_ppl", float('inf'))
            print(f"  {i}. {r['name']}: {ppl:.2f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            "experiments": results,
            "summary": {
                "total": len(variants),
                "successful": len(successful),
                "failed": len(variants) - len(successful),
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
