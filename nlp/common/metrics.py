import math
from typing import Iterable


def perplexity_from_nll(total_neg_log_likelihood: float, total_tokens: int) -> float:
    if total_tokens <= 0:
        return float("inf")
    return math.exp(total_neg_log_likelihood / total_tokens)


def safe_log(x: float, *, min_value: float = 1e-12) -> float:
    return math.log(max(x, min_value))

