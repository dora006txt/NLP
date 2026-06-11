import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _model_row(name: str, benchmark_entry: Dict[str, Any]) -> Dict[str, Any]:
    result = benchmark_entry.get("result")
    stdout = benchmark_entry.get("stdout", "")
    row: Dict[str, Any] = {
        "model": name,
        "ok": bool(result),
        "perplexity": None,
        "accuracy_top1": None,
        "accuracy_topk": None,
        "topk": None,
        "mrr": None,
        "tokens": None,
        "script_elapsed_ms": benchmark_entry.get("elapsed_ms"),
        "eval_elapsed_ms": None,
        "model_size_bytes": None,
        "error": None,
    }
    if isinstance(result, dict):
        row["perplexity"] = result.get("perplexity")
        row["accuracy_top1"] = result.get("accuracy_top1")
        row["accuracy_topk"] = result.get("accuracy_topk")
        row["topk"] = result.get("topk")
        row["mrr"] = result.get("mrr")
        row["tokens"] = result.get("tokens")
        row["eval_elapsed_ms"] = result.get("elapsed_ms")
        row["model_size_bytes"] = result.get("model_size_bytes")
    else:
        row["error"] = stdout.splitlines()[-1] if stdout else "no_result"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="artifacts/benchmark.json")
    parser.add_argument("--qualitative", default=None)
    parser.add_argument("--out-json", default="artifacts/slide_summary.json")
    parser.add_argument("--out-csv", default="artifacts/slide_table.csv")
    args = parser.parse_args()

    benchmark = _load_json(args.benchmark)
    rows: List[Dict[str, Any]] = []

    for model_name in ["ngram", "lstm", "gpt2"]:
        entry = benchmark.get(model_name)
        if isinstance(entry, dict):
            rows.append(_model_row(model_name, entry))

    qualitative_summary = None
    if args.qualitative and os.path.exists(args.qualitative):
        q = _load_json(args.qualitative)
        qualitative_summary = q.get("summary")

    payload = {"rows": rows, "qualitative_summary": qualitative_summary}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "ok",
            "perplexity",
            "accuracy_top1",
            "accuracy_topk",
            "topk",
            "mrr",
            "tokens",
            "script_elapsed_ms",
            "eval_elapsed_ms",
            "model_size_bytes",
            "error",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(args.out_json)
    print(args.out_csv)


if __name__ == "__main__":
    main()
