import argparse
import json
import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from nlp.lstm.dataset import get_dataloader, load_vocab
from nlp.lstm.model import LSTMLanguageModel

def evaluate_model(test_path="data/wikitext-103/wiki.test.tokens", model_path="artifacts/lstm/best_model.pth", vocab_path="artifacts/lstm/vocab.pkl", batch_size=128, seq_len=20, limit_lines=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Evaluate trên thiết bị: {device}")
    
    # Load vocab
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab.word2idx) if hasattr(vocab, "word2idx") else len(vocab['word2idx'])
    
    # Load test data
    dataloader, _ = get_dataloader(
        test_path, 
        batch_size=batch_size, 
        seq_len=seq_len, 
        vocab=vocab,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        limit_lines=limit_lines
    )
    
    # Load mô hình
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = LSTMLanguageModel(
        vocab_size=vocab_size, 
        embed_size=config.get('embed_size', 300), 
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 2),
        tie_weights=config.get('tie_weights', False)
    )
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    total_loss = 0
    top1_correct = 0
    topk_correct = 0
    mrr_sum = 0.0
    total = 0
    
    t0 = time.perf_counter()
    with torch.no_grad():
        # SỬA LẠI: Chỉ nhận x và y
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # SỬA LẠI: Chỉ truyền x vào model
            outputs, _ = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            
            total += y.size(0)
            _, predicted = torch.max(outputs.data, 1)
            top1_correct += (predicted == y).sum().item()
            top3 = torch.topk(outputs.data, k=3, dim=1).indices
            topk_correct += top3.eq(y.unsqueeze(1)).any(dim=1).sum().item()
            match = top3.eq(y.unsqueeze(1))
            if match.any():
                ranks = match.float().argmax(dim=1) + 1
                mrr_sum += (1.0 / ranks.float()).sum().item()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
    avg_loss = total_loss / total
    perplexity = math.exp(avg_loss)
    accuracy_top1 = 100 * top1_correct / max(total, 1)
    accuracy_topk = 100 * topk_correct / max(total, 1)
    mrr = (mrr_sum / total) if total else 0.0

    model_size_bytes = None
    try:
        model_size_bytes = os.path.getsize(model_path)
    except Exception:
        model_size_bytes = None
    
    print(f"--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Accuracy@1: {accuracy_top1:.2f}%")
    print(f"Accuracy@3: {accuracy_topk:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")
    return {
        "model": "lstm",
        "tokenizer": "word",
        "split": "test",
        "perplexity": perplexity,
        "accuracy_top1": accuracy_top1 / 100,
        "accuracy_topk": accuracy_topk / 100,
        "topk": 3,
        "mrr": mrr,
        "avg_loss": avg_loss,
        "tokens": total,
        "limit_lines": limit_lines,
        "model_path": model_path,
        "vocab_path": vocab_path,
        "model_size_bytes": model_size_bytes,
        "elapsed_ms": elapsed_ms,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default="data/wikitext-103/wiki.test.tokens")
    parser.add_argument("--model-path", default="artifacts/lstm/best_model.pth")
    parser.add_argument("--vocab-path", default="artifacts/lstm/vocab.pkl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--limit-lines", type=int, default=None)
    args = parser.parse_args()
    metrics = evaluate_model(args.test_path, args.model_path, args.vocab_path, batch_size=args.batch_size, seq_len=args.seq_len, limit_lines=args.limit_lines)
    print(json.dumps(metrics))
