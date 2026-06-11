import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nlp.common.corpus import iter_lines, normalize_for_word_models


def _take_prompts(path: str, *, limit_lines: int, max_items: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in iter_lines(path, limit_lines=limit_lines):
        normalized = normalize_for_word_models(line)
        if not normalized:
            continue
        words = normalized.split()
        if len(words) < 3:
            continue
        prefix = " ".join(words[:-1])
        target = words[-1]
        items.append({"prefix": prefix, "target": target, "raw": line})
        if len(items) >= max_items:
            break
    return items


def _norm_token_for_word_match(s: str) -> str:
    s = normalize_for_word_models(s)
    parts = s.split()
    return parts[0] if parts else ""


def _compute_word_metrics(items: List[Dict[str, Any]], preds_rows: List[Dict[str, Any]], *, topk: int) -> Dict[str, Any]:
    top1 = 0
    topk_hits = 0
    mrr_sum = 0.0
    total = 0
    for ex, row in zip(items, preds_rows):
        target = _norm_token_for_word_match(ex["target"])
        preds = row.get("preds") or []
        pred_words = [_norm_token_for_word_match(p[0]) for p in preds]
        if not target:
            continue
        if pred_words:
            if target == pred_words[0]:
                top1 += 1
            if target in pred_words[:topk]:
                topk_hits += 1
                rank = pred_words.index(target) + 1
                mrr_sum += 1.0 / rank
            total += 1
    return {
        "eval_items": total,
        "accuracy_top1": (top1 / total) if total else 0.0,
        "accuracy_topk": (topk_hits / total) if total else 0.0,
        "mrr": (mrr_sum / total) if total else 0.0,
        "topk": topk,
    }


def _predict_ngram(items: List[Dict[str, Any]], *, model_path: str, topk: int) -> Dict[str, Any]:
    from nlp.ngram.kenlm_model import KenLMNextWordPredictor

    model = KenLMNextWordPredictor(model_path)
    rows = []
    for ex in items:
        ctx_words = ex["prefix"].split()
        preds = model.predict_next(ctx_words, top_k=topk)
        rows.append({"preds": preds})
    return {"model": "ngram", "tokenizer": "word", "topk": topk, "model_path": model_path, "rows": rows}


def _predict_lstm(items: List[Dict[str, Any]], *, model_path: str, vocab_path: str, topk: int, device: Optional[str]) -> Dict[str, Any]:
    try:
        import torch
    except Exception as e:
        return {"model": "lstm", "error": f"missing_torch: {e}"}

    from nlp.common.tokenizer import SharedVocabulary
    from nlp.lstm.model import LSTMLanguageModel

    if device is not None:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load vocab and model
    vocab = SharedVocabulary.load(vocab_path)
    checkpoint = torch.load(model_path, map_location=dev, weights_only=False)
    config = checkpoint['config']

    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        tie_weights=config['tie_weights']
    ).to(dev)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    seq_len = config.get('seq_length', 35)
    rows = []
    with torch.no_grad():
        for ex in items:
            ctx_words = ex["prefix"].split()
            ctx_words = ctx_words[-seq_len:]
            # Use vocab.encode which handles <UNK> automatically
            encoded = vocab.encode(ctx_words)
            if len(encoded) < seq_len:
                encoded = [vocab.pad_idx] * (seq_len - len(encoded)) + encoded
            x = torch.tensor([encoded], dtype=torch.long).to(dev)
            output, _ = model(x)
            probs = torch.softmax(output[0], dim=-1)
            top_probs, top_ids = torch.topk(probs, k=topk)
            pred_words = vocab.decode(top_ids.tolist())
            preds = [(w, float(p.item())) for w, p in zip(pred_words, top_probs)]
            rows.append({"preds": preds})

    return {
        "model": "lstm",
        "tokenizer": "word",
        "topk": topk,
        "model_path": model_path,
        "vocab_path": vocab_path,
        "device": str(dev),
        "rows": rows,
    }


def _predict_gpt2(items: List[Dict[str, Any]], *, model_name: str, topk: int, device: Optional[str]) -> Dict[str, Any]:
    try:
        from nlp.gpt2.model import GPT2Config, GPT2NextWordPredictor
    except Exception as e:
        return {"model": "gpt2", "error": f"missing_transformers: {e}"}

    cfg = GPT2Config(model_name=model_name, device=device)
    try:
        predictor = GPT2NextWordPredictor(cfg)
    except Exception as e:
        return {"model": "gpt2", "error": str(e), "model_name": model_name}

    rows = []
    for ex in items:
        preds = predictor.predict_next(ex["prefix"], top_k=topk)
        rows.append({"preds": preds})
    return {"model": "gpt2", "tokenizer": "gpt2_bpe", "topk": topk, "model_name": model_name, "device": str(predictor.device), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default="data/wikitext-103/wiki.test.tokens")
    parser.add_argument("--limit-lines", type=int, default=500)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="artifacts/qualitative_100.json")
    parser.add_argument("--ngram-model", default="artifacts/ngram/kenlm_word.arpa.json")
    parser.add_argument("--lstm-model", default="artifacts/lstm/best_model.pth")
    parser.add_argument("--lstm-vocab", default="artifacts/lstm/vocab.pkl")
    parser.add_argument("--gpt2-model-name", default="gpt2")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    items = _take_prompts(args.test_path, limit_lines=args.limit_lines, max_items=args.n)

    payload: Dict[str, Any] = {
        "source": {"test_path": args.test_path, "limit_lines": args.limit_lines, "n": len(items)},
        "examples": items,
        "predictions": {},
        "summary": {},
    }

    if os.path.exists(args.ngram_model):
        payload["predictions"]["ngram"] = _predict_ngram(items, model_path=args.ngram_model, topk=args.topk)
    else:
        payload["predictions"]["ngram"] = {"model": "ngram", "error": f"missing_model_file: {args.ngram_model}"}

    if os.path.exists(args.lstm_model) and os.path.exists(args.lstm_vocab):
        payload["predictions"]["lstm"] = _predict_lstm(
            items,
            model_path=args.lstm_model,
            vocab_path=args.lstm_vocab,
            topk=min(args.topk, 10),
            device=args.device,
        )
    else:
        payload["predictions"]["lstm"] = {"model": "lstm", "error": "missing_lstm_artifacts"}

    payload["predictions"]["gpt2"] = _predict_gpt2(items, model_name=args.gpt2_model_name, topk=args.topk, device=args.device)

    for k in ["ngram", "lstm", "gpt2"]:
        entry = payload["predictions"].get(k)
        if not isinstance(entry, dict) or "rows" not in entry:
            payload["summary"][k] = {"model": k, "error": entry.get("error") if isinstance(entry, dict) else "missing"}
            continue
        metrics = _compute_word_metrics(items, entry["rows"], topk=entry.get("topk", args.topk))
        payload["summary"][k] = {"model": k, **metrics}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(args.out)


if __name__ == "__main__":
    main()
