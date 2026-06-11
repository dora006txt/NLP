import torch
import torch.nn as nn
import torch.nn.functional as F

class CharLevelCNN(nn.Module):
    def __init__(self, char_vocab_size=256, char_embed_size=30, num_filters=50, filter_sizes=[2, 3, 4, 5]):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_size, padding_idx=0)
        
        # Mạng CNN trích xuất đặc trưng mức ký tự
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, char_embed_size)) 
            for fs in filter_sizes
        ])
        self.output_dim = len(filter_sizes) * num_filters

    def forward(self, char_seqs):
        # char_seqs: (batch_size, seq_len, word_len)
        batch_size, seq_len, word_len = char_seqs.size()
        
        # Gộp batch_size và seq_len để xử lý như batch of words
        x = char_seqs.view(-1, word_len) # (batch_size * seq_len, word_len)
        x = self.char_embed(x) # (batch_size * seq_len, word_len, char_embed_size)
        x = x.unsqueeze(1) # (batch_size * seq_len, 1, word_len, char_embed_size)
        
        # Áp dụng Conv + ReLU + MaxPool
        conv_outs = []
        for conv in self.convs:
            out = F.relu(conv(x)).squeeze(3) # (batch_size * seq_len, num_filters, word_len - fs + 1)
            out = F.max_pool1d(out, out.size(2)).squeeze(2) # (batch_size * seq_len, num_filters)
            conv_outs.append(out)
            
        char_features = torch.cat(conv_outs, 1) # (batch_size * seq_len, sum(num_filters))
        char_features = char_features.view(batch_size, seq_len, -1) # (batch_size, seq_len, sum(num_filters))
        
        return char_features

class WordCharEmbeddings(nn.Module):
    def __init__(self, vocab_size, char_vocab_size=256, word_embed_size=300, char_embed_size=30, num_filters=50, filter_sizes=[2, 3, 4, 5]):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, word_embed_size, padding_idx=0)
        self.char_cnn = CharLevelCNN(char_vocab_size=char_vocab_size, char_embed_size=char_embed_size, 
                                     num_filters=num_filters, filter_sizes=filter_sizes)
        self.output_dim = word_embed_size + self.char_cnn.output_dim

    def forward(self, word_seqs, char_seqs=None):
        # word_seqs: (batch_size, seq_len)
        # char_seqs: (batch_size, seq_len, word_len)
        
        word_features = self.word_embed(word_seqs)
        
        if char_seqs is not None:
            char_features = self.char_cnn(char_seqs)
            # Nối (concatenate) vector từ và vector ký tự
            combined = torch.cat([word_features, char_features], dim=-1)
            return combined
        
        return word_features
