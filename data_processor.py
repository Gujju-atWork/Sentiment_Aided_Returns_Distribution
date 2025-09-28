# data_processor.py - Update for 4 categories
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import config

class FinancialDataProcessor:
    def __init__(self):
        self.return_bins = np.array(config.RETURN_BINS)
        self.category_labels = config.RETURN_CATEGORIES
        self.num_classes = config.NUM_CLASSES
        
    def create_target_vector(self, log_returns):
        """Convert log returns to 4-category probability distribution"""
        # Handle NaN/inf in log returns
        if np.isnan(log_returns) or np.isinf(log_returns):
            log_returns = 0.0
            
        # Categorize into 4 classes
        if log_returns < -1.0:      # <-1%
            target_idx = 0
        elif log_returns < 0.0:     # -1% to 0%
            target_idx = 1
        elif log_returns <= 1.0:    # 0% to 1%
            target_idx = 2
        else:                       # >1%
            target_idx = 3
            
        # Create smoothed probability distribution for 4 classes
        probabilities = np.zeros(self.num_classes)
        probabilities[target_idx] = 0.6  # Highest probability for target class
        
        # Smooth to neighboring classes
        if target_idx > 0:
            probabilities[target_idx-1] = 0.2
        if target_idx < self.num_classes - 1:
            probabilities[target_idx+1] = 0.2
            
        # If at edges, redistribute the smoothing probability
        if target_idx == 0:  # First category
            probabilities[target_idx+1] += 0.2
        elif target_idx == self.num_classes - 1:  # Last category
            probabilities[target_idx-1] += 0.2
            
        # Normalize to sum to 1
        probabilities /= probabilities.sum()
        
        return probabilities
    
    def get_category_from_return(self, log_returns):
        """Get category index from log returns"""
        if log_returns < -1.0:
            return 0  # <-1%
        elif log_returns < 0.0:
            return 1  # -1% to 0%
        elif log_returns <= 1.0:
            return 2  # 0% to 1%
        else:
            return 3  # >1%
    
    def clean_data(self, df):
        """Clean and preprocess the data to avoid NaN issues"""
        # Handle missing values in india_vix columns
        if 'india_vix' in df.columns:
            df['india_vix'] = df['india_vix'].fillna(method='ffill').fillna(method='bfill')
        if 'india_vix_pct_change' in df.columns:
            df['india_vix_pct_change'] = df['india_vix_pct_change'].fillna(0)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove any rows that still have NaN values
        df = df.dropna()
        
        return df
    
    def prepare_features(self, df):
        """Prepare all feature modules with proper validation"""
        all_features = [f for f in config.BASIC_FEATURES if f != 'date']
        
        # Add module features
        for module in config.FEATURE_MODULES:
            module_features = [f for f in df.columns if f.startswith(f'{module}_')]
            all_features.extend(module_features)
        
        # Add recent close features
        recent_features = [f for f in df.columns if f.startswith('recent_close_')]
        all_features.extend(recent_features)
        
        # Ensure all features exist in dataframe
        valid_features = [f for f in all_features if f in df.columns]
        
        print(f"Using {len(valid_features)} features out of {len(all_features)} requested")
        return valid_features
    
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the financial data with robust error handling"""
        df = pd.read_csv(data_path)
        
        # Clean the data first
        df = self.clean_data(df)
        
        # Convert Date column if needed
        if 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
            df = df.drop('Date', axis=1)
        
        # Calculate log returns for next day with error handling
        df['next_day_log_return'] = np.log(df['close'].shift(-1) / df['close']) * 100
        
        # Remove last row (no next day return) and any rows with invalid returns
        df = df[:-1].copy()
        df = df[np.isfinite(df['next_day_log_return'])]
        
        # Get all features
        features = self.prepare_features(df)
        
        # Prepare target vectors with validation
        target_vectors = []
        valid_indices = []
        
        for idx, return_val in enumerate(df['next_day_log_return']):
            if not np.isnan(return_val) and not np.isinf(return_val):
                target_vectors.append(self.create_target_vector(return_val))
                valid_indices.append(idx)
        
        # Filter X to only include valid indices
        X = df[features].iloc[valid_indices].values
        y = np.array(target_vectors)
        
        # Print category distribution
        category_counts = [0] * self.num_classes
        for return_val in df['next_day_log_return'].iloc[valid_indices]:
            category_counts[self.get_category_from_return(return_val)] += 1
            
        print(f"Category distribution:")
        for i, category in enumerate(self.category_labels):
            print(f"  {category}: {category_counts[i]} samples ({category_counts[i]/len(valid_indices)*100:.1f}%)")
        
        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
        
        return X, y, features

class FinancialDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FinancialSequenceDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X_seq = torch.FloatTensor(X_seq)
        self.y_seq = torch.FloatTensor(y_seq)
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        return self.X_seq[idx], self.y_seq[idx]
