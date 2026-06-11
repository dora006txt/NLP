#!/usr/bin/env python3
"""
Statistical Significance Testing for Model Comparison.

Implements:
- Paired t-test for comparing two models on same test set
- McNemar's test for binary outcomes
- Effect size calculation (Cohen's d)
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from scipy import stats
from scipy.stats import ttest_rel


def paired_t_test(model_a_scores: List[float], model_b_scores: List[float]) -> Dict[str, Any]:
    """
    Perform paired t-test to compare two models.
    
    Args:
        model_a_scores: Per-sample scores from model A
        model_b_scores: Per-sample scores from model B
    
    Returns:
        Dictionary with test statistics
    """
    if len(model_a_scores) != len(model_b_scores):
        raise ValueError("Score lists must have same length")
    
    # Paired t-test
    t_stat, p_value = ttest_rel(model_a_scores, model_b_scores)
    
    # Effect size (Cohen's d for paired samples)
    differences = np.array(model_a_scores) - np.array(model_b_scores)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    # Confidence interval for mean difference
    n = len(differences)
    se = std_diff / np.sqrt(n)
    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se
    
    return {
        "test": "paired_t_test",
        "n_samples": n,
        "model_a_mean": float(np.mean(model_a_scores)),
        "model_b_mean": float(np.mean(model_b_scores)),
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_95": [float(ci_low), float(ci_high)],
        "significant_at_05": bool(p_value < 0.05),
        "significant_at_01": bool(p_value < 0.01),
    }


def mcnemar_test(model_a_correct: List[bool], model_b_correct: List[bool]) -> Dict[str, Any]:
    """
    McNemar's test for comparing binary classification outcomes.
    
    Tests whether two models have different error rates.
    """
    # Contingency table
    # b: A correct, B incorrect
    # c: A incorrect, B correct
    b = sum(a and not b for a, b in zip(model_a_correct, model_b_correct))
    c = sum(not a and b for a, b in zip(model_a_correct, model_b_correct))
    
    # McNemar's test statistic
    if b + c == 0:
        chi2 = 0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # With continuity correction
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return {
        "test": "mcnemar",
        "n_samples": len(model_a_correct),
        "both_correct": sum(a and b for a, b in zip(model_a_correct, model_b_correct)),
        "both_incorrect": sum(not a and not b for a, b in zip(model_a_correct, model_b_correct)),
        "a_correct_b_incorrect": b,
        "a_incorrect_b_correct": c,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
    }


def evaluate_model_on_samples(model, test_sequences: List[List[str]], 
                               top_k: int = 5) -> Tuple[List[float], List[bool]]:
    """
    Evaluate model on test sequences, returning per-sample scores.
    
    Returns:
        (perplexities, top1_correct_flags)
    """
    perplexities = []
    top1_correct = []
    
    for seq in test_sequences:
        if len(seq) < 2:
            continue
        
        # Predict next word for each position
        ctx = seq[:-1]
        target = seq[-1]
        
        # Get prediction
        if hasattr(model, 'predict_next'):
            preds = model.predict_next(ctx, top_k=top_k)
            if preds:
                top1_correct.append(preds[0][0] == target)
            else:
                top1_correct.append(False)
        
        # Calculate perplexity contribution
        if hasattr(model, 'prob'):
            p = model.prob(target, ctx)
            if p > 0:
                perplexities.append(-np.log(p))
            else:
                perplexities.append(20.0)  # Cap at ~exp(20)
    
    return perplexities, top1_correct


def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Testing")
    parser.add_argument("--model-a", required=True, help="Path to model A results or checkpoint")
    parser.add_argument("--model-b", required=True, help="Path to model B results or checkpoint")
    parser.add_argument("--test-path", default="data/wikitext-103/wiki.test.tokens")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--out", default="artifacts/significance_test.json")
    parser.add_argument("--metric", default="perplexity", choices=["perplexity", "accuracy"])
    args = parser.parse_args()
    
    print("=== Statistical Significance Testing ===")
    print(f"Comparing: {args.model_a}")
    print(f"With:      {args.model_b}")
    print(f"Metric:    {args.metric}")
    
    # Placeholder: In practice, you would load models and evaluate
    # For now, we assume results are already computed and stored in JSON
    
    # Try to load pre-computed results
    try:
        with open(args.model_a, 'r') as f:
            results_a = json.load(f)
        with open(args.model_b, 'r') as f:
            results_b = json.load(f)
        
        # Extract per-sample scores
        if args.metric == "perplexity":
            scores_a = results_a.get("per_sample_perplexities", [])
            scores_b = results_b.get("per_sample_perplexities", [])
        else:
            scores_a = results_a.get("per_sample_correct", [])
            scores_b = results_b.get("per_sample_correct", [])
        
        # Run appropriate test
        if args.metric == "perplexity":
            results = paired_t_test(scores_a, scores_b)
        else:
            results = mcnemar_test([s == 1 for s in scores_a], [s == 1 for s in scores_b])
        
        print(f"\nResults:")
        print(f"  p-value: {results['p_value']:.6f}")
        print(f"  Significant at α=0.05: {results['significant_at_05']}")
        print(f"  Significant at α=0.01: {results['significant_at_01']}")
        
        # Save results
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.out}")
        
    except FileNotFoundError:
        print("\nNote: Model result files not found.")
        print("This script requires pre-computed per-sample results.")
        print("Use --model-a and --model-b with JSON files containing per_sample_* fields.")


if __name__ == "__main__":
    main()
