# training/evaluator.py - Fix the constructor
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch

class ModelEvaluator:
    def __init__(self, category_labels):  # REMOVED return_bins parameter
        self.category_labels = category_labels
        
    def evaluate_model(self, model, test_loader, device):
        model.eval()
        all_preds = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                # Convert probabilities to class predictions
                pred_classes = torch.argmax(outputs, dim=1)
                target_classes = torch.argmax(batch_y, dim=1)
                
                all_preds.extend(pred_classes.cpu().numpy())
                all_targets.extend(target_classes.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_targets), np.array(all_probabilities)
    
    def generate_classification_report(self, preds, targets, model_name):
        print(f"\n{model_name} Classification Report:")
        print(classification_report(targets, preds, target_names=self.category_labels, digits=4))
        
        # Calculate accuracy for each return category
        accuracy_by_class = []
        for i in range(len(self.category_labels)):
            class_mask = targets == i
            if class_mask.sum() > 0:
                class_accuracy = (preds[class_mask] == i).mean()
                accuracy_by_class.append((self.category_labels[i], class_accuracy))
        
        print("Accuracy by Return Category:")
        for class_name, acc in accuracy_by_class:
            print(f"{class_name}: {acc:.4f}")
