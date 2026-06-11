import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nlp.common.artifacts import ArtifactPaths
from nlp.common.corpus import iter_lines, iter_word_sequences
from nlp.ngram.kenlm_model import KenLMNextWordPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default="data/wikitext-103/wiki.test.tokens")
    parser.add_argument("--limit-lines", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--accuracy-max-tokens", type=int, default=1000)
    args = parser.parse_args()

    artifacts = ArtifactPaths()
    model_path = args.model or os.path.join(artifacts.ngram_dir(), "kenlm_word.arpa.json")
    model = KenLMNextWordPredictor(model_path)

    sequences = iter_word_sequences(args.test_path, limit_lines=args.limit_lines, add_sentence_markers=False)
    t0 = time.perf_counter()
    nll = 0.0
    n_tokens = 0
    avg_nll, ppl, n_tokens = model.perplexity(iter_lines(args.test_path, limit_lines=args.limit_lines))
    nll = avg_nll * n_tokens

    sequences = iter_word_sequences(args.test_path, limit_lines=args.limit_lines, add_sentence_markers=False)
    top1_correct = 0
    topk_correct = 0
    mrr_sum = 0.0
    total = 0
    for seq in sequences:
        ctx: list[str] = []
        for w in seq:
            preds = model.predict_next(ctx, top_k=args.topk)
            if preds:
                pred_words = [p[0] for p in preds]
                if w == pred_words[0]:
                    top1_correct += 1
                if w in pred_words:
                    topk_correct += 1
                    rank = pred_words.index(w) + 1
                    mrr_sum += 1.0 / rank
                total += 1
                if total >= args.accuracy_max_tokens:
                    break
            ctx.append(w)
        if total >= args.accuracy_max_tokens:
            break
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    top1_acc = (top1_correct / total) if total else 0.0
    topk_acc = (topk_correct / total) if total else 0.0
    mrr = (mrr_sum / total) if total else 0.0

    model_size_bytes = None
    try:
        model_size_bytes = os.path.getsize(model_path)
    except Exception:
        model_size_bytes = None
    print(
        json.dumps(
            {
                "model": "ngram",
                "tokenizer": "word",
                "split": "test",
                "perplexity": ppl,
                "accuracy_top1": top1_acc,
                "accuracy_topk": topk_acc,
                "mrr": mrr,
                "topk": args.topk,
                "tokens": n_tokens,
                "accuracy_eval_tokens": total,
                "model_path": model_path,
                "model_size_bytes": model_size_bytes,
                "limit_lines": args.limit_lines,
                "avg_loss": avg_nll,
                "elapsed_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
