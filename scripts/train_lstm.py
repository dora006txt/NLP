"""
Professional LSTM Language Model Training Script.

Features:
- Reproducibility with fixed random seeds
- Gradient clipping, weight decay (L2 regularization)
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping based on validation perplexity
- Pretrained GloVe embeddings support
- Mixed precision training (AMP)
- Checkpoint saving
"""

import argparse
import os
import sys
import time
import random
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nlp.lstm.dataset import get_dataloader, save_vocab
from nlp.lstm.model import LSTMLanguageModel
from nlp.common.tokenizer import SharedVocabulary

# Reproducibility
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_glove_embeddings(glove_path: str, vocab: SharedVocabulary, embed_size: int = 300) -> torch.Tensor:
    """Load pretrained GloVe embeddings."""
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    # Initialize with random embeddings
    embeddings = torch.randn(len(vocab), embed_size) * 0.01
    embeddings[vocab.pad_idx] = 0  # Zero padding token
    
    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab.word2idx:
                idx = vocab.word2idx[word]
                vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                if len(vector) == embed_size:
                    embeddings[idx] = vector
                    found += 1
    
    print(f"  Found {found}/{len(vocab)} words in GloVe ({100*found/len(vocab):.1f}%)")
    return embeddings


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, max_grad_norm=5.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs, _ = model(x)
            loss = criterion(outputs, y)
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient clipping (unscale first for proper clipping)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Track loss (adjust for accumulation)
        batch_tokens = (y != 0).sum().item()  # Non-padding tokens
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            outputs, _ = model(x)
            loss = criterion(outputs, y)
            
            batch_tokens = (y != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def train_model(
    train_path: str = "data/wikitext-103/wiki.train.tokens",
    valid_path: str = "data/wikitext-103/wiki.valid.tokens",
    epochs: int = 20,
    batch_size: int = 256,
    seq_len: int = 35,
    embed_size: int = 300,
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.5,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 5.0,
    early_stopping: int = 3,
    use_glove: bool = False,
    glove_path: str = "data/glove.6B.300d.txt",
    tie_weights: bool = False,
    limit_lines: int | None = None,
    valid_limit_lines: int | None = None,
    device_str: str | None = None,
):
    """Main training loop with professional features."""
    
    # Setup device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Create output directory
    os.makedirs('artifacts/lstm', exist_ok=True)
    
    # Build/load vocabulary from training data
    print("\n=== Building Vocabulary ===")
    train_loader, vocab = get_dataloader(
        train_path,
        batch_size=batch_size,
        seq_len=seq_len,
        limit_lines=limit_lines,
    )
    
    vocab_size = len(vocab.word2idx if isinstance(vocab, SharedVocabulary) else vocab['word2idx'])
    print(f"Vocabulary size: {vocab_size}")
    
    # Save vocabulary
    save_vocab(vocab, "artifacts/lstm/vocab.pkl")
    
    # Load validation data (using same vocab)
    print("\n=== Loading Validation Data ===")
    valid_loader, _ = get_dataloader(
        valid_path,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab=vocab,
        limit_lines=valid_limit_lines if valid_limit_lines is not None else limit_lines,
    )
    
    # Load pretrained embeddings if requested
    pretrained_embeddings = None
    if use_glove and os.path.exists(glove_path):
        pretrained_embeddings = load_glove_embeddings(glove_path, vocab, embed_size)
    
    # Initialize model
    print(f"\n=== Initializing Model ===")
    print(f"Architecture: LSTM-{num_layers} layers, {hidden_size} hidden units")
    print(f"Tied weights: {tie_weights}")
    
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        tie_weights=tie_weights,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Loss function (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    # Training state
    best_val_ppl = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'val_ppl': [], 'lr': []}
    
    print(f"\n=== Training for {epochs} epochs ===")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, max_grad_norm)
        
        # Validate
        val_loss, val_ppl = evaluate(model, valid_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
        history['lr'].append(current_lr)
        
        elapsed = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:.2f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.1f}s")
        
        # Save best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': val_ppl,
                'config': {
                    'vocab_size': vocab_size,
                    'embed_size': embed_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'tie_weights': tie_weights,
                }
            }
            torch.save(checkpoint, "artifacts/lstm/best_model.pth")
            print(f"  -> Saved best model (Val PPL: {val_ppl:.2f})")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement ({epochs_without_improvement}/{early_stopping})")
        
        # Early stopping
        if epochs_without_improvement >= early_stopping:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Clear cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save training history
    with open("artifacts/lstm/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    print(f"Model saved to: artifacts/lstm/best_model.pth")
    print(f"History saved to: artifacts/lstm/training_history.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Language Model")
    
    # Data
    parser.add_argument("--train-path", default="data/wikitext-103/wiki.train.tokens")
    parser.add_argument("--valid-path", default="data/wikitext-103/wiki.valid.tokens")
    parser.add_argument("--limit-lines", type=int, default=None)
    parser.add_argument("--valid-limit-lines", type=int, default=None)
    
    # Model architecture
    parser.add_argument("--embed-size", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=35)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--tie-weights", action="store_true", help="Tie embedding and output weights")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--early-stopping", type=int, default=3, help="Epochs without improvement before stopping")
    
    # Embeddings
    parser.add_argument("--use-glove", action="store_true", help="Use pretrained GloVe embeddings")
    parser.add_argument("--glove-path", default="data/glove.6B.300d.txt")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Device to use (cuda/mps/cpu)")
    
    args = parser.parse_args()
    
    # Set seed
    RANDOM_SEED = args.seed
    set_seed(args.seed)
    
    train_model(
        train_path=args.train_path,
        valid_path=args.valid_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        early_stopping=args.early_stopping,
        use_glove=args.use_glove,
        glove_path=args.glove_path,
        tie_weights=args.tie_weights,
        limit_lines=args.limit_lines,
        valid_limit_lines=args.valid_limit_lines,
        device_str=args.device,
    )
