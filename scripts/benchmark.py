import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nlp.common.artifacts import ArtifactPaths, save_json


def run_py(module_path: str, args: List[str]) -> Dict[str, Any]:
    cmd = [sys.executable, module_path, *args]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    out = proc.stdout.strip()
    payload: Dict[str, Any] = {"cmd": cmd, "returncode": proc.returncode, "stdout": out, "elapsed_ms": elapsed_ms}
    try:
        last_line = out.splitlines()[-1] if out else ""
        payload["result"] = json.loads(last_line)
    except Exception:
        payload["result"] = None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/benchmark.json")
    parser.add_argument("--limit-lines", type=int, default=200)
    parser.add_argument("--run-gpt2", action="store_true")
    parser.add_argument("--ngram-model", default=None)
    parser.add_argument("--lstm-model", default="artifacts/lstm/best_model.pth")
    parser.add_argument("--lstm-vocab", default="artifacts/lstm/vocab.pkl")
    parser.add_argument("--gpt2-model-name", default="gpt2")
    parser.add_argument("--run-qualitative", action="store_true")
    parser.add_argument("--make-slide-report", action="store_true")
    args = parser.parse_args()

    artifacts = ArtifactPaths()
    ngram_model = args.ngram_model
    if ngram_model is None:
        candidates = [
            os.path.join(artifacts.ngram_dir(), "kenlm_word.arpa.json"),
            os.path.join(artifacts.ngram_dir(), "kenlm_word_test.arpa.json"),
        ]
        ngram_model = next((p for p in candidates if os.path.exists(p)), None)
    if ngram_model is None:
        autogen = os.path.join(artifacts.ngram_dir(), "kenlm_word_autogen.arpa.json")
        _ = run_py(
            "scripts/train_ngram.py",
            ["--limit-lines", str(args.limit_lines), "--out", autogen],
        )
        if os.path.exists(autogen):
            ngram_model = autogen

    results: Dict[str, Any] = {
        "ngram": run_py(
            "scripts/evaluate_ngram.py",
            [
                "--limit-lines",
                str(args.limit_lines),
                *(["--model", ngram_model] if ngram_model else []),
            ],
        ),
        "lstm": run_py(
            "scripts/evaluate_lstm.py",
            [
                "--limit-lines",
                str(args.limit_lines),
                "--model-path",
                args.lstm_model,
                "--vocab-path",
                args.lstm_vocab,
            ],
        ),
    }
    if args.run_gpt2:
        results["gpt2"] = run_py(
            "scripts/evaluate_gpt2.py",
            ["--limit-lines", str(args.limit_lines), "--model-name", args.gpt2_model_name],
        )
    if args.run_qualitative:
        qualitative_args = ["--limit-lines", str(args.limit_lines)]
        if ngram_model:
            qualitative_args.extend(["--ngram-model", ngram_model])
        qualitative_args.extend(["--gpt2-model-name", args.gpt2_model_name])
        results["qualitative_100"] = run_py("scripts/qualitative_100.py", qualitative_args)

    save_json(args.out, results)
    if args.make_slide_report:
        q_path = "artifacts/qualitative_100.json" if args.run_qualitative else ""
        _ = run_py(
            "scripts/report.py",
            [
                "--benchmark",
                args.out,
                *(["--qualitative", q_path] if q_path else []),
            ],
        )
    print(args.out)


if __name__ == "__main__":
    main()
