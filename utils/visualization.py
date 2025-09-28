# utils/visualization.py - Fix parameter name
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_training_history(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(preds, targets, model_name, category_labels):  # FIXED: changed return_bins to category_labels
    cm = confusion_matrix(targets, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=category_labels, yticklabels=category_labels)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_probability_distribution(probabilities, model_name, category_labels, sample_idx=0):
    """Plot probability distribution for a sample prediction"""
    plt.figure(figsize=(8, 6))
    
    plt.bar(category_labels, probabilities[sample_idx])
    plt.title(f'{model_name} - Probability Distribution (Sample {sample_idx})')
    plt.xlabel('Return Category')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
