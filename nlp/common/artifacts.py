import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ArtifactPaths:
    root: str = "artifacts"

    def ngram_dir(self) -> str:
        return os.path.join(self.root, "ngram")

    def lstm_dir(self) -> str:
        return os.path.join(self.root, "lstm")

    def gpt2_dir(self) -> str:
        return os.path.join(self.root, "gpt2")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

