import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from nlp.common.corpus import normalize_for_word_models


def _require_kenlm():
    try:
        import kenlm  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError(
            "KenLM Python bindings are not installed. "
            "Install them with `pip install kenlm` or run the Colab setup script."
        ) from exc
    return kenlm


def resolve_binary(binary_name: str, kenlm_root: str | None = None) -> str:
    candidates = []
    if kenlm_root:
        candidates.extend(
            [
                os.path.join(kenlm_root, "build", "bin", binary_name),
                os.path.join(kenlm_root, "bin", binary_name),
            ]
        )
    env_root = os.environ.get("KENLM_ROOT")
    if env_root:
        candidates.extend(
            [
                os.path.join(env_root, "build", "bin", binary_name),
                os.path.join(env_root, "bin", binary_name),
            ]
        )
    candidates.extend(
        [
            os.path.join(os.getcwd(), "kenlm", "build", "bin", binary_name),
            os.path.join(os.getcwd(), "kenlm", "bin", binary_name),
            f"/usr/local/bin/{binary_name}",
            f"/usr/bin/{binary_name}",
        ]
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find KenLM binary `{binary_name}`. "
        "Set `KENLM_ROOT` or run the Colab setup script first."
    )


@dataclass
class KenLMManifest:
    model_type: str
    order: int
    binary_path: str
    arpa_path: str
    vocab_path: str
    normalized_train_path: str
    train_path: str
    limit_lines: int | None
    candidate_vocab_size: int
    memory: str
    prune: list[int]

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "order": self.order,
            "binary_path": self.binary_path,
            "arpa_path": self.arpa_path,
            "vocab_path": self.vocab_path,
            "normalized_train_path": self.normalized_train_path,
            "train_path": self.train_path,
            "limit_lines": self.limit_lines,
            "candidate_vocab_size": self.candidate_vocab_size,
            "memory": self.memory,
            "prune": self.prune,
        }


class KenLMNextWordPredictor:
    def __init__(self, manifest_path: str):
        self.manifest_path = os.path.abspath(manifest_path)
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.manifest = KenLMManifest(**raw)
        self._kenlm = _require_kenlm()
        self.model = self._kenlm.Model(self._resolve(self.manifest.binary_path))
        self.candidate_vocab = self._load_vocab(self._resolve(self.manifest.vocab_path))

    def _resolve(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(os.path.dirname(self.manifest_path), path))

    @staticmethod
    def _load_vocab(path: str) -> list[str]:
        words: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if token:
                    words.append(token)
        return words

    def _state_from_context(self, context: Sequence[str]):
        state = self._kenlm.State()
        self.model.BeginSentenceWrite(state)
        for token in context:
            out_state = self._kenlm.State()
            self.model.BaseScore(state, token, out_state)
            state = out_state
        return state

    def score_word_log10(self, context: Sequence[str], word: str) -> float:
        state = self._state_from_context(context)
        out_state = self._kenlm.State()
        return float(self.model.BaseScore(state, word, out_state))

    def prob(self, context: Sequence[str], word: str) -> float:
        return 10 ** self.score_word_log10(context, word)

    def predict_next(
        self,
        context: Sequence[str],
        *,
        top_k: int = 10,
        candidate_words: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        candidates = candidate_words if candidate_words is not None else self.candidate_vocab
        state = self._state_from_context(context)
        scored: list[tuple[str, float]] = []
        for word in candidates:
            out_state = self._kenlm.State()
            score_log10 = float(self.model.BaseScore(state, word, out_state))
            scored.append((word, 10 ** score_log10))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def perplexity(self, lines: Iterable[str]) -> tuple[float, float, int]:
        total_nll = 0.0
        total_tokens = 0
        for line in lines:
            normalized = normalize_for_word_models(line)
            if not normalized:
                continue
            total_nll += -float(self.model.score(normalized, bos=True, eos=True)) * math.log(10)
            total_tokens += len(normalized.split()) + 1
        if total_tokens <= 0:
            return float("inf"), float("inf"), 0
        avg_nll = total_nll / total_tokens
        return avg_nll, math.exp(avg_nll), total_tokens


def build_kenlm_model(
    *,
    train_path: str,
    output_manifest_path: str,
    order: int = 3,
    limit_lines: int | None = None,
    memory: str = "50%",
    candidate_vocab_size: int = 30000,
    prune: Sequence[int] | None = None,
    kenlm_root: str | None = None,
) -> KenLMManifest:
    if order < 1:
        raise ValueError("order must be >= 1")

    output_manifest_path = os.path.abspath(output_manifest_path)
    out_dir = os.path.dirname(output_manifest_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    prune_values = list(prune) if prune is not None else [0] * max(order - 1, 0)
    prefix = os.path.splitext(output_manifest_path)[0]
    normalized_train_path = f"{prefix}.normalized.txt"
    arpa_path = f"{prefix}.arpa"
    binary_path = f"{prefix}.bin"
    vocab_path = f"{prefix}.vocab.txt"
    temp_prefix = os.path.join(out_dir, "kenlm_tmp")
    Path(temp_prefix).mkdir(parents=True, exist_ok=True)

    token_counts: dict[str, int] = {}
    seen_lines = 0
    with open(train_path, "r", encoding="utf-8") as src, open(
        normalized_train_path, "w", encoding="utf-8"
    ) as dst:
        for raw_line in src:
            if limit_lines is not None and seen_lines >= limit_lines:
                break
            normalized = normalize_for_word_models(raw_line)
            if not normalized:
                continue
            dst.write(normalized + "\n")
            for token in normalized.split():
                token_counts[token] = token_counts.get(token, 0) + 1
            seen_lines += 1

    if not token_counts:
        raise RuntimeError("No training data found after normalization.")

    top_vocab = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[:candidate_vocab_size]
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in top_vocab:
            f.write(token + "\n")

    lmplz = resolve_binary("lmplz", kenlm_root=kenlm_root)
    build_binary = resolve_binary("build_binary", kenlm_root=kenlm_root)

    prune_args = [str(v) for v in prune_values] if prune_values else ["0"]
    with open(normalized_train_path, "r", encoding="utf-8") as src, open(
        arpa_path, "w", encoding="utf-8"
    ) as dst:
        subprocess.run(
            [
                lmplz,
                "-o",
                str(order),
                "--discount_fallback",
                "--memory",
                memory,
                "--temp_prefix",
                temp_prefix,
                "--prune",
                *prune_args,
            ],
            stdin=src,
            stdout=dst,
            check=True,
        )

    subprocess.run([build_binary, arpa_path, binary_path], check=True)

    manifest = KenLMManifest(
        model_type="kenlm",
        order=order,
        binary_path=os.path.relpath(binary_path, out_dir),
        arpa_path=os.path.relpath(arpa_path, out_dir),
        vocab_path=os.path.relpath(vocab_path, out_dir),
        normalized_train_path=os.path.relpath(normalized_train_path, out_dir),
        train_path=train_path,
        limit_lines=limit_lines,
        candidate_vocab_size=candidate_vocab_size,
        memory=memory,
        prune=prune_values,
    )
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
    return manifest
