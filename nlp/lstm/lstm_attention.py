import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardAttention(nn.Module):
    def __init__(self, hidden_size):
        super(StandardAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Mở rộng hidden state để tính toán attention với mọi token
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, hidden_size)
        
        # Tính năng lượng attention: e = v * tanh(W * encoder_outputs)
        energy = torch.tanh(self.attn(encoder_outputs)) # (batch_size, seq_len, hidden_size)
        energy = energy.transpose(1, 2) # (batch_size, hidden_size, seq_len)
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1) # (batch_size, 1, hidden_size)
        
        attention_scores = torch.bmm(v, energy).squeeze(1) # (batch_size, seq_len)
        
        # Áp dụng softmax để lấy trọng số
        return F.softmax(attention_scores, dim=1)

class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=256, num_layers=2, dropout=0.5):
        super(NextWordPredictor, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. ĐẢM BẢO DÙNG EMBEDDING CƠ BẢN CỦA PYTORCH (Output ra 300)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # 2. ĐẢM BẢO INPUT_SIZE CỦA LSTM TRÙNG VỚI EMBED_SIZE (Tức là 300)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = StandardAttention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x)) # (batch_size, seq_len, 300)
        
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        last_hidden = hidden[-1]
        
        attn_weights = self.attention(last_hidden, lstm_out)
        attn_weights = attn_weights.unsqueeze(1)
        
        context = torch.bmm(attn_weights, lstm_out).squeeze(1)
        
        combined = torch.cat((context, last_hidden), dim=1)
        
        output = self.fc(self.dropout(combined))
        return output
