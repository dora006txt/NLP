#!/usr/bin/env python3
"""Run evaluation and qualitative analysis on Colab, then sync artifacts to Drive."""

import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from colab_helper import mount_drive, setup_path, sync_paths_to_drive, write_json


def run_script(args: list[str]) -> dict:
    proc = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=PROJECT_ROOT, check=False)
    return {
        "command": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm-model", default="artifacts/lstm/best_model.pth")
    parser.add_argument("--lstm-vocab", default="artifacts/lstm/vocab.pkl")
    parser.add_argument("--ngram-model", default="artifacts/ngram/kenlm_word.arpa.json")
    parser.add_argument("--gpt2-model-name", default="artifacts/gpt2_finetuned/final_model")
    parser.add_argument("--limit-lines", type=int, default=500)
    parser.add_argument("--generate-samples", action="store_true")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--drive-dir", default="DA_NPL_Artifacts")
    parser.add_argument("--skip-drive-sync", action="store_true")
    args = parser.parse_args()

    setup_path()
    os.makedirs(os.path.join(PROJECT_ROOT, "artifacts", "colab"), exist_ok=True)

    runs = {
        "kenlm_eval": run_script(
            [
                "scripts/evaluate_ngram.py",
                "--model",
                args.ngram_model,
                "--limit-lines",
                str(args.limit_lines),
            ]
        ),
        "lstm_eval": run_script(
            [
                "scripts/evaluate_lstm.py",
                "--model-path",
                args.lstm_model,
                "--vocab-path",
                args.lstm_vocab,
                "--limit-lines",
                str(args.limit_lines),
            ]
        ),
        "gpt2_eval": run_script(
            [
                "scripts/evaluate_gpt2.py",
                "--model-name",
                args.gpt2_model_name,
                "--limit-lines",
                str(args.limit_lines),
            ]
        ),
        "qualitative": run_script(
            [
                "scripts/qualitative_100.py",
                "--ngram-model",
                args.ngram_model,
                "--lstm-model",
                args.lstm_model,
                "--lstm-vocab",
                args.lstm_vocab,
                "--gpt2-model-name",
                args.gpt2_model_name,
                "--limit-lines",
                str(max(args.limit_lines, args.n_samples)),
                "--n",
                str(args.n_samples),
            ]
        ),
        "benchmark": run_script(
            [
                "scripts/benchmark.py",
                "--ngram-model",
                args.ngram_model,
                "--lstm-model",
                args.lstm_model,
                "--lstm-vocab",
                args.lstm_vocab,
                "--gpt2-model-name",
                args.gpt2_model_name,
                "--limit-lines",
                str(args.limit_lines),
                "--run-gpt2",
                "--run-qualitative",
            ]
        ),
    }

    manifest_path = os.path.join(PROJECT_ROOT, "artifacts", "colab", "evaluation_manifest.json")
    write_json(manifest_path, runs)
    print(json.dumps(runs, ensure_ascii=False, indent=2))

    backup_root = None if args.skip_drive_sync else mount_drive(args.drive_dir)
    if backup_root:
        sync_paths_to_drive(
            [
                "artifacts/benchmark.json",
                "artifacts/qualitative_100.json",
                "artifacts/lstm",
                "artifacts/gpt2_finetuned",
                "artifacts/ngram",
                "artifacts/colab/evaluation_manifest.json",
            ],
            backup_root,
            PROJECT_ROOT,
        )


if __name__ == "__main__":
    main()
