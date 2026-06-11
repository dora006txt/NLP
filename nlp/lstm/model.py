"""
Professional Baseline LSTM Language Model.

Implements a highly optimized, pure LSTM architecture for Language Modeling.
Attention mechanisms have been structurally removed to fit the Many-to-One 
autoregressive prediction task efficiently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LSTMLanguageModel(nn.Module):
    """
    Baseline LSTM Language Model.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 300,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        tie_weights: bool = False,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_weights = tie_weights
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocab_size, embed_size)
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        else:
            nn.init.xavier_uniform_(self.embedding.weight)
        
        # 2. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # 3. Output Projection Layer
        if tie_weights:
            if embed_size != hidden_size:
                raise ValueError("For tied weights, embed_size must equal hidden_size")
            self.fc = nn.Linear(hidden_size, vocab_size)
            # Tie weights: Dùng chung trọng số giữa Embedding và Fully Connected
            self.fc.weight = self.embedding.weight
        else:
            self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_lstm_weights()
    
    def _init_lstm_weights(self):
        """Khởi tạo trọng số LSTM chuẩn mực để gradient flow tốt hơn."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                n = param.size(0)
                param.data.fill_(0)
                # Set forget gate bias to 1
                param.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None) -> tuple[torch.Tensor, tuple]:
        """
        Forward pass (Tối ưu hóa cho tốc độ).
        """
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, embed_size)
        
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(embedded, hidden)  
        
        # Chỉ lấy hidden state của layer cuối cùng ở bước thời gian cuối
        # h_n shape: (num_layers, batch_size, hidden_size)
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        # Trực tiếp đi qua Dropout và Linear layer (Loại bỏ hoàn toàn cổ chai Attention)
        combined = self.dropout(last_hidden)
        output = self.fc(combined)  # (batch_size, vocab_size)
        
        return output, (h_n, c_n)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 20, 
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        generated = []
        hidden = None
        
        with torch.no_grad():
            logits, hidden = self(input_ids, hidden)
            
            for _ in range(max_length):
                next_token_logits = logits / temperature
                
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_token)
                logits, hidden = self(next_token, hidden)
        
        return torch.cat(generated, dim=1)

# Backward compatibility
NextWordPredictor = LSTMLanguageModel