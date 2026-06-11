#!/usr/bin/env python3
"""Prepare data and shared vocabulary for Colab training."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from colab_helper import mount_drive, setup_path, sync_paths_to_drive, write_json


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "wikitext-103")
TRAIN_PATH = os.path.join(DATA_DIR, "wiki.train.tokens")
VALID_PATH = os.path.join(DATA_DIR, "wiki.valid.tokens")
TEST_PATH = os.path.join(DATA_DIR, "wiki.test.tokens")


def _has_dataset() -> bool:
    paths = [TRAIN_PATH, VALID_PATH, TEST_PATH]
    return all(os.path.exists(path) and os.path.getsize(path) > 1024 for path in paths)


def _download_wikitext() -> None:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    os.makedirs(DATA_DIR, exist_ok=True)

    def write_split(split_name: str, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for text in dataset[split_name]["text"]:
                text = text.rstrip()
                if text:
                    f.write(text + "\n")

    write_split("train", TRAIN_PATH)
    write_split("validation", VALID_PATH)
    write_split("test", TEST_PATH)


def _count_nonempty_lines(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def prepare_dataset() -> dict:
    if not _has_dataset():
        _download_wikitext()

    return {
        "dataset": "wikitext-103",
        "train_path": TRAIN_PATH,
        "valid_path": VALID_PATH,
        "test_path": TEST_PATH,
        "train_nonempty_lines": _count_nonempty_lines(TRAIN_PATH),
        "valid_nonempty_lines": _count_nonempty_lines(VALID_PATH),
        "test_nonempty_lines": _count_nonempty_lines(TEST_PATH),
    }


def build_vocabulary(max_vocab_size: int, min_freq: int, limit_lines: int | None) -> dict:
    from nlp.common.tokenizer import SharedVocabulary, VocabConfig, WordTokenizer

    vocab = SharedVocabulary(VocabConfig(max_vocab_size=max_vocab_size, min_freq=min_freq))
    vocab.build_from_corpus(TRAIN_PATH, WordTokenizer(), limit_lines=limit_lines)
    vocab_path = os.path.join(PROJECT_ROOT, "artifacts", "lstm", "vocab.pkl")
    Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)
    return {
        "vocab_path": vocab_path,
        "vocab_size": len(vocab),
        "vocab_limit_lines": limit_lines,
        "min_freq": min_freq,
        "max_vocab_size": max_vocab_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--vocab-limit-lines", type=int, default=150000)
    parser.add_argument("--drive-dir", default="DA_NPL_Artifacts")
    parser.add_argument("--skip-drive-sync", action="store_true")
    args = parser.parse_args()

    setup_path()
    dataset_info = prepare_dataset()
    vocab_info = build_vocabulary(args.vocab_size, args.min_freq, args.vocab_limit_lines)

    artifact_dir = os.path.join(PROJECT_ROOT, "artifacts", "colab")
    os.makedirs(artifact_dir, exist_ok=True)
    manifest_path = os.path.join(artifact_dir, "dataset_manifest.json")
    payload = {**dataset_info, **vocab_info}
    write_json(manifest_path, payload)

    backup_root = None if args.skip_drive_sync else mount_drive(args.drive_dir)
    if backup_root:
        sync_paths_to_drive(
            [
                "artifacts/lstm/vocab.pkl",
                "artifacts/colab/dataset_manifest.json",
                "data/wikitext-103/wiki.valid.tokens",
                "data/wikitext-103/wiki.test.tokens",
            ],
            backup_root,
            PROJECT_ROOT,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
