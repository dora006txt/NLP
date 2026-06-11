"""
Shared Tokenizer and Vocabulary Management for Fair Model Comparison.

Ensures N-gram and LSTM use identical tokenization and vocabulary,
enabling fair comparison of model architectures.
"""

import pickle
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VocabConfig:
    """Configuration for vocabulary building."""
    max_vocab_size: int = 50000
    min_freq: int = 2
    add_special_tokens: bool = True
    PAD_token: str = "<PAD>"
    UNK_token: str = "<UNK>"
    SOS_token: str = "<s>"
    EOS_token: str = "</s>"


@dataclass
class TokenizerConfig:
    """Configuration for tokenization."""
    lowercase: bool = True
    remove_punctuation: bool = True
    add_sentence_markers: bool = True


class WordTokenizer:
    """
    Standardized word-level tokenizer for both N-gram and LSTM models.
    
    Handles:
    - Text normalization (lowercase, punctuation removal)
    - Sentence boundary marking
    - Consistent tokenization across models
    """
    
    def __init__(self, config: TokenizerConfig | None = None):
        self.config: TokenizerConfig = config if config is not None else TokenizerConfig()
    
    def normalize(self, text: str) -> str:
        """Normalize text: lowercase and remove punctuation."""
        if self.config.lowercase:
            text = text.lower()
        if self.config.remove_punctuation:
            # Keep alphanumeric and whitespace only
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            # Collapse multiple spaces
            text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        normalized = self.normalize(text)
        if not normalized:
            return []
        return normalized.split()
    
    def tokenize_with_markers(self, text: str) -> list[str]:
        """Tokenize with sentence boundary markers."""
        tokens = self.tokenize(text)
        if not tokens:
            return []
        
        if self.config.add_sentence_markers:
            return ["<s>"] + tokens + ["</s>"]
        return tokens
    
    def tokenize_file(self, path: str, limit_lines: int | None = None) -> Iterator[list[str]]:
        """Tokenize an entire file, yielding sentence token lists."""
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if limit_lines is not None and count >= limit_lines:
                    break
                tokens = self.tokenize_with_markers(line)
                if tokens:
                    yield tokens
                count += 1
    
    def tokenize_file_flat(self, path: str, limit_lines: int | None = None) -> Iterator[str]:
        """Tokenize file and yield flat token stream."""
        for sent_tokens in self.tokenize_file(path, limit_lines):
            yield from sent_tokens


class SharedVocabulary:
    """
    Shared vocabulary for N-gram and LSTM models.
    
    Features:
    - Bidirectional word <-> index mapping
    - Consistent across both model types
    - Pickle-serializable
    """
    
    def __init__(self, config: VocabConfig | None = None):
        self.config: VocabConfig = config if config is not None else VocabConfig()
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_counts: Counter[str] = Counter()
        self._initialized: bool = False
    
    def build_from_counter(self, counter: Counter[str]) -> None:
        """Build vocabulary from word frequency counter."""
        # Filter by minimum frequency
        filtered = {w: c for w, c in counter.items() if c >= self.config.min_freq}
        
        # Sort by frequency (descending)
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        
        # Reserve space for special tokens
        max_words = self.config.max_vocab_size - 4  # PAD, UNK, SOS, EOS
        sorted_words = sorted_words[:max_words]
        
        # Initialize with special tokens
        self.word2idx = {}
        self.idx2word = {}
        
        if self.config.add_special_tokens:
            self.word2idx[self.config.PAD_token] = 0
            self.word2idx[self.config.UNK_token] = 1
            self.word2idx[self.config.SOS_token] = 2
            self.word2idx[self.config.EOS_token] = 3
            
            self.idx2word[0] = self.config.PAD_token
            self.idx2word[1] = self.config.UNK_token
            self.idx2word[2] = self.config.SOS_token
            self.idx2word[3] = self.config.EOS_token
            
            start_idx = 4
        else:
            start_idx = 0
        
        # Add words
        for i, (word, _) in enumerate(sorted_words):
            if word not in self.word2idx:  # Avoid duplicates
                idx = start_idx + i
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.word_counts = counter
        self._initialized = True
    
    def build_from_corpus(self, corpus_path: str, tokenizer: WordTokenizer, 
                          limit_lines: int | None = None) -> None:
        """Build vocabulary from a corpus file."""
        counter = Counter()
        for tokens in tokenizer.tokenize_file(corpus_path, limit_lines):
            # Don't count special tokens
            filtered = [t for t in tokens if t not in 
                       [self.config.SOS_token, self.config.EOS_token]]
            counter.update(filtered)
        
        self.build_from_counter(counter)
    
    def encode(self, words: Sequence[str]) -> list[int]:
        """Convert words to indices."""
        return [self.word2idx.get(w, self.word2idx[self.config.UNK_token]) 
                for w in words]
    
    def decode(self, indices: Sequence[int]) -> list[str]:
        """Convert indices to words."""
        return [self.idx2word.get(i, self.config.UNK_token) for i in indices]
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    @property
    def vocab_size(self) -> int:
        return len(self)
    
    @property
    def pad_idx(self) -> int:
        return self.word2idx.get(self.config.PAD_token, 0)
    
    @property
    def unk_idx(self) -> int:
        return self.word2idx.get(self.config.UNK_token, 1)
    
    @property
    def sos_idx(self) -> int:
        return self.word2idx.get(self.config.SOS_token, 2)
    
    @property
    def eos_idx(self) -> int:
        return self.word2idx.get(self.config.EOS_token, 3)
    
    def save(self, path: str) -> None:
        """Save vocabulary to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "config": self.config,
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
                "word_counts": self.word_counts,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "SharedVocabulary":
        """Load vocabulary from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        vocab = cls(data["config"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        vocab.word_counts = data["word_counts"]
        vocab._initialized = True
        return vocab


def create_shared_vocab(train_path: str,
                        vocab_path: str,
                        max_vocab_size: int = 50000,
                        limit_lines: int | None = None) -> SharedVocabulary:
    """
    Create and save shared vocabulary from training corpus.
    
    This is the main entry point for ensuring both N-gram and LSTM
    use the same vocabulary.
    
    Args:
        train_path: Path to training corpus
        vocab_path: Where to save the vocabulary
        max_vocab_size: Maximum vocabulary size
        limit_lines: Limit lines for debugging/quick testing
    
    Returns:
        SharedVocabulary instance
    """
    tokenizer = WordTokenizer(TokenizerConfig(
        lowercase=True,
        remove_punctuation=True,
        add_sentence_markers=True
    ))
    
    config = VocabConfig(
        max_vocab_size=max_vocab_size,
        min_freq=2,
        add_special_tokens=True
    )
    
    vocab = SharedVocabulary(config)
    vocab.build_from_corpus(train_path, tokenizer, limit_lines)
    vocab.save(vocab_path)
    
    print(f"Created shared vocabulary:")
    print(f"  - Vocab size: {len(vocab)}")
    print(f"  - Train file: {train_path}")
    print(f"  - Saved to: {vocab_path}")
    
    return vocab


# Backward compatibility functions
def normalize_for_word_models(text: str) -> str:
    """Backward compatible normalization."""
    tokenizer = WordTokenizer()
    return tokenizer.normalize(text)


def iter_word_tokens(path: str, *, limit_lines: int | None = None,
                     add_sentence_markers: bool = True) -> Iterator[str]:
    """Backward compatible word token iterator."""
    tokenizer = WordTokenizer(TokenizerConfig(
        add_sentence_markers=add_sentence_markers
    ))
    yield from tokenizer.tokenize_file_flat(path, limit_lines)


def iter_word_sequences(path: str, *, limit_lines: int | None = None,
                       add_sentence_markers: bool = True) -> Iterator[list[str]]:
    """Backward compatible word sequence iterator."""
    tokenizer = WordTokenizer(TokenizerConfig(
        add_sentence_markers=add_sentence_markers
    ))
    yield from tokenizer.tokenize_file(path, limit_lines)
