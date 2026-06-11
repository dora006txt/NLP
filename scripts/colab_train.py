#!/usr/bin/env python3
"""Colab wrapper for LSTM training with Drive backup."""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from colab_helper import mount_drive, setup_path, sync_paths_to_drive, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--tie-weights", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seq-length", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--early-stopping", type=int, default=3)
    parser.add_argument("--train-limit-lines", type=int, default=150000)
    parser.add_argument("--valid-limit-lines", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--drive-dir", default="DA_NPL_Artifacts")
    parser.add_argument("--skip-drive-sync", action="store_true")
    args = parser.parse_args()

    setup_path()
    if args.tie_weights and args.embed_size != args.hidden_size:
        raise ValueError("When --tie-weights is enabled, --embed-size must equal --hidden-size.")

    from scripts.train_lstm import set_seed, train_model

    set_seed(args.seed)
    start = time.time()
    train_model(
        train_path="data/wikitext-103/wiki.train.tokens",
        valid_path="data/wikitext-103/wiki.valid.tokens",
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_length,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        early_stopping=args.early_stopping,
        use_glove=False,
        tie_weights=args.tie_weights,
        limit_lines=args.train_limit_lines,
        valid_limit_lines=args.valid_limit_lines,
        device_str=args.device,
    )

    elapsed_hours = (time.time() - start) / 3600.0
    artifact_dir = os.path.join(PROJECT_ROOT, "artifacts", "colab")
    os.makedirs(artifact_dir, exist_ok=True)
    run_manifest = {
        "script": "scripts/colab_train.py",
        "elapsed_hours": elapsed_hours,
        "config": {
            "embed_size": args.embed_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "tie_weights": args.tie_weights,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seq_length": args.seq_length,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "early_stopping": args.early_stopping,
            "train_limit_lines": args.train_limit_lines,
            "valid_limit_lines": args.valid_limit_lines,
            "seed": args.seed,
            "device": args.device,
        },
    }
    write_json(os.path.join(artifact_dir, "lstm_train_manifest.json"), run_manifest)

    backup_root = None if args.skip_drive_sync else mount_drive(args.drive_dir)
    if backup_root:
        sync_paths_to_drive(
            [
                "artifacts/lstm",
                "artifacts/colab/lstm_train_manifest.json",
            ],
            backup_root,
            PROJECT_ROOT,
        )

    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
