# main.py - Fix the evaluator initialization
import numpy as np
import pandas as pd
import torch
from data_processor import FinancialDataProcessor, FinancialDataset, FinancialSequenceDataset
from models.ann_model import AdvancedANN
from models.lstm_model import AdvancedLSTM
from models.transformer_model import TransformerModel
from training.trainer import FinancialTrainer
from training.evaluator import ModelEvaluator
from utils.visualization import plot_training_history, plot_confusion_matrix
import config

def main():
    print("=== Financial AI Trading System ===")
    print(f"Using device: {config.DEVICE}")
    
    # Initialize components
    processor = FinancialDataProcessor()
    evaluator = ModelEvaluator(processor.category_labels)  # FIXED: Only pass category_labels
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, feature_names = processor.load_and_preprocess_data(config.DATA_PATH)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    from torch.utils.data import DataLoader
    
    # ANN datasets
    ann_train_dataset = FinancialDataset(X_train_scaled, y_train)
    ann_val_dataset = FinancialDataset(X_val_scaled, y_val)
    ann_test_dataset = FinancialDataset(X_test_scaled, y_test)
    
    ann_train_loader = DataLoader(ann_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    ann_val_loader = DataLoader(ann_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    ann_test_loader = DataLoader(ann_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Sequence datasets for LSTM/Transformer
    X_seq_train, y_seq_train = processor.create_sequences(X_train_scaled, y_train, config.SEQUENCE_LENGTH)
    X_seq_val, y_seq_val = processor.create_sequences(X_val_scaled, y_val, config.SEQUENCE_LENGTH)
    X_seq_test, y_seq_test = processor.create_sequences(X_test_scaled, y_test, config.SEQUENCE_LENGTH)
    
    seq_train_dataset = FinancialSequenceDataset(X_seq_train, y_seq_train)
    seq_val_dataset = FinancialSequenceDataset(X_seq_val, y_seq_val)
    seq_test_dataset = FinancialSequenceDataset(X_seq_test, y_seq_test)
    
    seq_train_loader = DataLoader(seq_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    seq_val_loader = DataLoader(seq_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    seq_test_loader = DataLoader(seq_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Models to train
    models_config = {
        'AdvancedANN': {
            'model': AdvancedANN(X_train_scaled.shape[1], config.HIDDEN_LAYERS, config.DROPOUT_RATE),
            'train_loader': ann_train_loader,
            'val_loader': ann_val_loader,
            'test_loader': ann_test_loader
        },
        'AdvancedLSTM': {
            'model': AdvancedLSTM(X_seq_train.shape[2], config.HIDDEN_SIZE, config.LSTM_LAYERS, config.DROPOUT_RATE),
            'train_loader': seq_train_loader,
            'val_loader': seq_val_loader,
            'test_loader': seq_test_loader
        },
        'Transformer': {
            'model': TransformerModel(X_seq_train.shape[2], config.D_MODEL, config.NHEAD, config.TRANSFORMER_LAYERS, config.DROPOUT_RATE),
            'train_loader': seq_train_loader,
            'val_loader': seq_val_loader,
            'test_loader': seq_test_loader
        }
    }
    
    results = {}
    
    for model_name, config_dict in models_config.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        model = config_dict['model']
        trainer = FinancialTrainer(model, config.DEVICE, model_name)
        
        # Train model
        trainer.train_model(
            config_dict['train_loader'],
            config_dict['val_loader'],
            config.NUM_EPOCHS,
            config.LEARNING_RATE
        )
        
        # Plot training history
        plot_training_history(trainer.train_losses, trainer.val_losses, model_name)
        
        # Evaluate model
        preds, targets, probabilities = evaluator.evaluate_model(
            trainer.model, 
            config_dict['test_loader'], 
            config.DEVICE
        )
        
        # Generate reports - FIXED: pass category_labels to plot_confusion_matrix
        evaluator.generate_classification_report(preds, targets, model_name)
        plot_confusion_matrix(preds, targets, model_name, processor.category_labels)
        
        results[model_name] = {
            'preds': preds,
            'targets': targets,
            'probabilities': probabilities,
            'trainer': trainer
        }
    
    # Compare all models
    print("\n=== MODEL COMPARISON ===")
    for model_name, result in results.items():
        accuracy = (result['preds'] == result['targets']).mean()
        print(f"{model_name}: Accuracy = {accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
