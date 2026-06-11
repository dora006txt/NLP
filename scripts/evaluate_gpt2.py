import argparse
import json
import itertools
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nlp.common.corpus import iter_lines
from nlp.gpt2.model import GPT2Config, GPT2NextWordPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default="data/wikitext-103/wiki.test.tokens")
    parser.add_argument("--limit-lines", type=int, default=200)
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    predictor = GPT2NextWordPredictor(GPT2Config(model_name=args.model_name))
    lines = list(itertools.islice(iter_lines(args.test_path, limit_lines=args.limit_lines), args.limit_lines))
    text = "\n".join(lines)
    torch = predictor.torch
    tokenizer = predictor.tokenizer
    model = predictor.model
    device = predictor.device

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    if input_ids.numel() < 2:
        print(
            json.dumps(
                {
                    "model": "gpt2",
                    "tokenizer": "gpt2_bpe",
                    "split": "test",
                    "perplexity": None,
                    "accuracy_top1": None,
                    "accuracy_topk": None,
                    "topk": 5,
                    "tokens": int(input_ids.numel()),
                    "lines": len(lines),
                    "model_name": args.model_name,
                    "limit_lines": args.limit_lines,
                }
            )
        )
        return

    total_nll = 0.0
    total_tokens = 0
    top1_correct = 0
    topk_correct = 0

    block = int(args.block_size)
    stride = int(args.stride)
    t0 = time.perf_counter()
    for start in range(0, input_ids.numel() - 1, stride):
        end = min(start + block, input_ids.numel() - 1)
        x = input_ids[start : end + 1].unsqueeze(0).to(device)
        labels = x[:, 1:].contiguous()

        with torch.no_grad():
            logits = model(input_ids=x).logits[:, :-1, :].contiguous()
            log_probs = torch.log_softmax(logits, dim=-1)
            gold = labels.unsqueeze(-1)
            nll = -log_probs.gather(dim=-1, index=gold).squeeze(-1)

            total_nll += float(nll.sum().item())
            total_tokens += int(labels.numel())

            top1 = torch.argmax(logits, dim=-1)
            top1_correct += int((top1 == labels).sum().item())

            top5 = torch.topk(logits, k=5, dim=-1).indices
            topk_correct += int(top5.eq(labels.unsqueeze(-1)).any(dim=-1).sum().item())

        if end >= input_ids.numel() - 2:
            break

    ppl = float(torch.exp(torch.tensor(total_nll / max(total_tokens, 1))).item())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        json.dumps(
            {
                "model": "gpt2",
                "tokenizer": "gpt2_bpe",
                "split": "test",
                "perplexity": ppl,
                "accuracy_top1": (top1_correct / total_tokens) if total_tokens else 0.0,
                "accuracy_topk": (topk_correct / total_tokens) if total_tokens else 0.0,
                "topk": 5,
                "tokens": total_tokens,
                "lines": len(lines),
                "model_name": args.model_name,
                "limit_lines": args.limit_lines,
                "block_size": block,
                "stride": stride,
                "elapsed_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
