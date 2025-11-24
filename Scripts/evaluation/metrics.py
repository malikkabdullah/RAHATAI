"""
Comprehensive evaluation metrics for RahatAI

Calculates all required metrics:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, Exact Match, Top-k Accuracy
- Confusion Matrix
- Training/validation plots (accuracy & loss curves)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_proba: Optional[np.ndarray] = None,
                                     labels: Optional[List] = None,
                                     average: str = 'weighted') -> Dict:
    """
    Calculate all classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (for AUC)
        labels: List of label names
        average: Averaging strategy for multi-class
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # AUC (requires probabilities)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.flatten())
            else:  # Multi-class
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            metrics['auc'] = None
    
    # Exact Match (for sequence classification)
    metrics['exact_match'] = accuracy_score(y_true, y_pred)  # Same as accuracy for single-label
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

def calculate_top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """
    Calculate Top-k Accuracy
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities (shape: [n_samples, n_classes])
        k: Top k predictions to consider
        
    Returns:
        float: Top-k accuracy
    """
    if y_proba.shape[1] < k:
        k = y_proba.shape[1]
    
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None,
                         labels: Optional[List] = None,
                         top_k: int = 3) -> Dict:
    """
    Calculate all required metrics for evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        labels: List of label names
        top_k: k for top-k accuracy
        
    Returns:
        dict: Complete metrics dictionary
    """
    metrics = calculate_classification_metrics(y_true, y_pred, y_proba, labels)
    
    # Add Top-k Accuracy
    if y_proba is not None:
        metrics['top_k_accuracy'] = calculate_top_k_accuracy(y_true, y_proba, k=top_k)
    else:
        metrics['top_k_accuracy'] = None
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Optional[List] = None,
                         save_path: Optional[str] = None,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (10, 8)):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_training_history(history: Dict, save_dir: str = "Outputs/plots/",
                         model_name: str = "model"):
    """
    Plot training and validation accuracy/loss curves
    
    Args:
        history: Training history dictionary with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        save_dir: Directory to save plots
        model_name: Name of the model (for filename)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot Accuracy
    if 'accuracy' in history or 'acc' in history:
        acc_key = 'accuracy' if 'accuracy' in history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
        
        plt.figure(figsize=(10, 6))
        plt.plot(history[acc_key], label='Training Accuracy', marker='o')
        if val_acc_key in history:
            plt.plot(history[val_acc_key], label='Validation Accuracy', marker='s')
        plt.title(f'{model_name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f"{model_name}_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Accuracy plot saved to {save_dir / f'{model_name}_accuracy.png'}")
    
    # Plot Loss
    if 'loss' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss', marker='o')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss', marker='s')
        plt.title(f'{model_name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f"{model_name}_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to {save_dir / f'{model_name}_loss.png'}")

def save_metrics(metrics: Dict, save_path: str):
    """
    Save metrics to JSON file
    
    Args:
        metrics: Metrics dictionary
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Metrics saved to {save_path}")

def print_metrics_summary(metrics: Dict):
    """
    Print a formatted summary of metrics
    
    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS SUMMARY")
    print("="*50)
    
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'auc': 'AUC-ROC',
        'exact_match': 'Exact Match',
        'top_k_accuracy': 'Top-k Accuracy'
    }
    
    for key, display_name in metric_names.items():
        if key in metrics and metrics[key] is not None:
            value = metrics[key]
            if isinstance(value, float):
                print(f"{display_name:20s}: {value:.4f}")
            else:
                print(f"{display_name:20s}: {value}")
    
    print("="*50 + "\n")

