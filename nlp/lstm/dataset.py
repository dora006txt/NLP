"""Dataset utilities for LSTM language model with shared vocabulary."""

import torch
from torch.utils.data import Dataset, DataLoader

from nlp.common.tokenizer import (
    SharedVocabulary,
    VocabConfig,
    WordTokenizer,
    TokenizerConfig,
)

class TextDataset(Dataset):
    """
    Text dataset for language modeling using shared vocabulary.
    
    Compatible with both legacy vocab format (dict) and new SharedVocabulary.
    """
    
    def __init__(self, file_path, seq_len=20, max_vocab_size=50000, vocab=None, limit_lines=None):
        self.seq_len = seq_len
        self.file_path = file_path
        
        # Load or build vocabulary
        if vocab is None:
            print(f"Building vocabulary from {file_path}...")
            config = VocabConfig(max_vocab_size=max_vocab_size, min_freq=2)
            shared_vocab = SharedVocabulary(config)
            tokenizer = WordTokenizer(TokenizerConfig(lowercase=True, remove_punctuation=True))
            shared_vocab.build_from_corpus(file_path, tokenizer, limit_lines)
            self.vocab = shared_vocab
            self.word2idx = shared_vocab.word2idx
            self.idx2word = shared_vocab.idx2word
        elif isinstance(vocab, SharedVocabulary):
            # New SharedVocabulary format
            self.vocab = vocab
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word
        elif isinstance(vocab, dict):
            # Legacy format for backward compatibility
            self.vocab = vocab
            self.word2idx = vocab['word2idx']
            self.idx2word = vocab['idx2word']
        else:
            raise ValueError(f"Unsupported vocab type: {type(vocab)}")
        
        # Tokenize and encode data
        print(f"Loading data from {file_path}...")
        tokenizer = WordTokenizer(TokenizerConfig(lowercase=True, remove_punctuation=True))
        encoded_words = []
        
        for tokens in tokenizer.tokenize_file(file_path, limit_lines):
            if isinstance(self.vocab, SharedVocabulary):
                encoded_words.extend(self.vocab.encode(tokens))
            else:
                # Legacy format
                encoded_words.extend([self.word2idx.get(t, self.word2idx.get('<UNK>', 1)) for t in tokens])
        
        self.encoded_data = torch.tensor(encoded_words, dtype=torch.long)
        print(f"  Total tokens: {len(self.encoded_data)}")
        print(f"  Total samples: {len(self)}")
        print(f"  Vocab size: {len(self.word2idx)}")

    # Legacy methods removed - now using SharedVocabulary from tokenizer module

    def __len__(self):
        # Trừ đi seq_len để đảm bảo lúc cắt mảng không bị lố index ở cuối
        return len(self.encoded_data) - self.seq_len

    def __getitem__(self, idx):
        # Slice trực tiếp từ mảng 1D (Cực nhanh cho GPU)
        x = self.encoded_data[idx : idx + self.seq_len]
        y = self.encoded_data[idx + self.seq_len]
        return x, y

def get_dataloader(file_path, batch_size=128, seq_len=20, max_vocab_size=50000, 
                   vocab=None, num_workers=2, pin_memory=True, limit_lines=None):
    """
    Create DataLoader for language modeling.
    
    Args:
        vocab: Can be None (build new), SharedVocabulary, path string, or legacy dict
    """
    # Load vocab if path provided
    if isinstance(vocab, str):
        vocab = SharedVocabulary.load(vocab)
    
    dataset = TextDataset(
        file_path, 
        seq_len=seq_len, 
        max_vocab_size=max_vocab_size, 
        vocab=vocab, 
        limit_lines=limit_lines
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return dataloader, dataset.vocab


def save_vocab(vocab, file_path):
    """Save vocabulary - supports both SharedVocabulary and legacy dict."""
    if isinstance(vocab, SharedVocabulary):
        vocab.save(file_path)
    else:
        # Legacy format
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(vocab, f)


def load_vocab(file_path):
    """Load vocabulary - auto-detects SharedVocabulary format."""
    try:
        # Try new format first
        return SharedVocabulary.load(file_path)
    except Exception:
        # Fall back to legacy format
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
