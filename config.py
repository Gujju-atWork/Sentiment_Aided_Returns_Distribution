# config.py - Update return categories
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
DATA_PATH = 'Data/Reliance_with_features_updated.csv'

# Model hyperparameters
HIDDEN_LAYERS = [512, 256, 128]
HIDDEN_SIZE = 256
LSTM_LAYERS = 3
D_MODEL = 256
NHEAD = 8
TRANSFORMER_LAYERS = 4
DROPOUT_RATE = 0.3

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 10
WEIGHT_DECAY = 1e-4

# NEW: 4 categories for returns
RETURN_BINS = [-float('inf'), -1.0, 0.0, 1.0, float('inf')]  # Thresholds for categories
RETURN_CATEGORIES = ['<-1%', '-1% to 0%', '0% to 1%', '>1%']  # Category labels
NUM_CLASSES = 4  # Number of output classes

# Feature groups
BASIC_FEATURES = [
    'prev_close', 'open', 'high', 'low', 'close', 'volume', 
    'turnover', 'trades', 'deliverable_qty', 'deliverable_pct',
    'india_vix', 'india_vix_pct_change'
]

FEATURE_MODULES = [
    'trend', 'momentum', 'oscillator', 'volatility', 'bands', 
    'cycle', 'strength', 'risk', 'liquidity', 'institutional',
    'delivery', 'market_structure'
]
