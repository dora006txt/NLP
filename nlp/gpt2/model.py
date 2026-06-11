from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GPT2Config:
    model_name: str = "gpt2"
    device: Optional[str] = None


class GPT2NextWordPredictor:
    def __init__(self, config: GPT2Config = GPT2Config()):
        self.config = config
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError("Missing dependency: transformers") from e

        import torch

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)

        if config.device is not None:
            device = torch.device(config.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict_next(self, text: str, *, top_k: int = 10) -> List[Tuple[str, float]]:
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            next_logits = logits[0, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=top_k)

        results: List[Tuple[str, float]] = []
        for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
            token = self.tokenizer.decode([token_id])
            results.append((token, float(prob)))
        return results

    def perplexity_for_text(self, text: str) -> float:
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
        return float(torch.exp(loss).item())

