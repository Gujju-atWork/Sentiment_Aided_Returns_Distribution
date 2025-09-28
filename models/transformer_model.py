# models/transformer_model.py - Update for 4 classes
import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # UPDATED: 4 classes instead of 13
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        
        # Use the last time step for classification
        x = x[:, -1, :]
        output = self.classifier(x)
        
        return output
