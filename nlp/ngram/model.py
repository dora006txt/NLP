import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, Iterator, List, Sequence, Tuple

from nlp.common.metrics import safe_log


Context = Tuple[str, ...]


@dataclass
class NGramConfig:
    max_n: int = 3
    discount: float = 0.75


class NGramLanguageModel:
    def __init__(self, config: NGramConfig = NGramConfig()):
        if config.max_n < 1 or config.max_n > 3:
            raise ValueError("max_n must be in [1, 3]")
        self.config = config

        self.unigram: Counter[str] = Counter()
        self.bigram: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        self.trigram: DefaultDict[Tuple[str, str], Counter[str]] = defaultdict(Counter)

        self.bigram_context_totals: Counter[str] = Counter()
        self.trigram_context_totals: Counter[Tuple[str, str]] = Counter()

        self.continuation_counts: Counter[str] = Counter()
        self.total_unique_bigrams: int = 0

        self.vocab: set[str] = set()

    def update_from_sequences(self, sequences: Iterable[Sequence[str]]) -> None:
        for seq in sequences:
            self._update_from_sequence(seq)
        self.vocab = set(self.unigram.keys())

    def _update_from_sequence(self, seq: Sequence[str]) -> None:
        tokens = list(seq)
        if not tokens:
            return

        for w in tokens:
            self.unigram[w] += 1

        if self.config.max_n >= 2:
            prev_seen: set[tuple[str, str]] = set()
            for w_prev, w in zip(tokens[:-1], tokens[1:]):
                self.bigram[w_prev][w] += 1
                self.bigram_context_totals[w_prev] += 1

                pair = (w_prev, w)
                if pair not in prev_seen:
                    prev_seen.add(pair)
                    self.continuation_counts[w] += 1
                    self.total_unique_bigrams += 1

        if self.config.max_n >= 3:
            for w1, w2, w3 in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
                ctx = (w1, w2)
                self.trigram[ctx][w3] += 1
                self.trigram_context_totals[ctx] += 1

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "NGramLanguageModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def _p_continuation(self, word: str) -> float:
        if self.total_unique_bigrams <= 0:
            return 1.0 / max(len(self.vocab), 1)
        return self.continuation_counts[word] / self.total_unique_bigrams

    def prob(self, word: str, context: Sequence[str]) -> float:
        d = self.config.discount
        ctx = tuple(context)

        if self.config.max_n == 1:
            total = sum(self.unigram.values())
            return self.unigram[word] / total if total else 0.0

        if len(ctx) >= 2 and self.config.max_n >= 3:
            ctx2 = (ctx[-2], ctx[-1])
            return self._prob_kn_trigram(word, ctx2, d)

        if len(ctx) >= 1:
            return self._prob_kn_bigram(word, ctx[-1], d)

        return self._p_continuation(word)

    def _prob_kn_bigram(self, word: str, prev: str, d: float) -> float:
        ctx_total = self.bigram_context_totals[prev]
        if ctx_total <= 0:
            return self._p_continuation(word)

        count = self.bigram[prev][word]
        unique_followers = len(self.bigram[prev])
        lambda_h = (d * unique_followers) / ctx_total
        return max(count - d, 0.0) / ctx_total + lambda_h * self._p_continuation(word)

    def _prob_kn_trigram(self, word: str, ctx: Tuple[str, str], d: float) -> float:
        ctx_total = self.trigram_context_totals[ctx]
        if ctx_total <= 0:
            return self._prob_kn_bigram(word, ctx[1], d)

        count = self.trigram[ctx][word]
        unique_followers = len(self.trigram[ctx])
        lambda_h = (d * unique_followers) / ctx_total
        backoff = self._prob_kn_bigram(word, ctx[1], d)
        return max(count - d, 0.0) / ctx_total + lambda_h * backoff

    def predict_next(self, context: Sequence[str], *, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.vocab:
            return []

        candidates: Iterable[str]
        ctx = list(context)
        if self.config.max_n >= 3 and len(ctx) >= 2:
            ctx2 = (ctx[-2], ctx[-1])
            candidates = self.trigram.get(ctx2, {}).keys() or self.vocab
        elif self.config.max_n >= 2 and len(ctx) >= 1:
            candidates = self.bigram.get(ctx[-1], {}).keys() or self.vocab
        else:
            candidates = self.vocab

        scored = [(w, self.prob(w, ctx)) for w in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def neg_log_likelihood(self, sequences: Iterable[Sequence[str]]) -> tuple[float, int]:
        total_nll = 0.0
        total_tokens = 0
        for seq in sequences:
            ctx: list[str] = []
            for w in seq:
                p = self.prob(w, ctx)
                total_nll += -safe_log(p)
                total_tokens += 1
                ctx.append(w)
        return total_nll, total_tokens

