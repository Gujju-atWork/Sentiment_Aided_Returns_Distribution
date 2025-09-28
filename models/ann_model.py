# models/ann_model.py - Update for 4 classes
import torch.nn as nn
import torch

class AdvancedANN(nn.Module):
    def __init__(self, input_size, hidden_layers=[512, 256, 128], dropout_rate=0.3):
        super(AdvancedANN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Final classifier - UPDATED to 4 classes
        self.classifier = nn.Sequential(
            nn.Linear(prev_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # UPDATED: 4 classes instead of 13
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Normalize input
        x = self.input_bn(x)
        
        # Forward pass
        features = self.feature_layers(x)
        output = self.classifier(features)
        
        return output
