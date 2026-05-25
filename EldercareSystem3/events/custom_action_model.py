import torch
import torch.nn as nn
import math

class TransformerLSTMActionModel(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=128, num_classes=7, num_heads=4, num_layers=2, dropout=0.3):
        super(TransformerLSTMActionModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. Feature Projection
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. Positional Encoding for Transformer
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*2, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. LSTM Layer (Bidirectional for better context)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, 51)
        
        # 1. Linear Projection: (batch_size, seq_len, 128)
        x = self.feature_proj(x)
        
        # 2. Positional Encoding
        x = self.pos_encoder(x)
        
        # 3. Transformer (Global attention across frames)
        x = self.transformer_encoder(x)
        
        # 4. LSTM (Strict temporal sequencing)
        # We take the final hidden state of the LSTM
        out, (hn, cn) = self.lstm(x)
        # hn shape for bidirectional: (2, batch_size, hidden_dim)
        # We concatenate the final forward and backward states
        final_state = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) 
        
        # 5. Classification
        logits = self.classifier(final_state)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
