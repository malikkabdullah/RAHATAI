"""
STEP 5 (Efficient): TRAIN MODEL 3 - LSTM with Advanced Balancing
This version uses data oversampling, stronger class weights, and optimized training
"""

# ----------------------------------------------------
# Import all required libraries
# ----------------------------------------------------
import os
from pathlib import Path
import sys

# Setting project root path so Python can find modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Basic imports
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Importing custom-built scripts
from Scripts.classification.dl_models import LSTMClassifier
from Scripts.evaluation.metrics import (
    calculate_all_metrics,
    plot_confusion_matrix,
    plot_training_history,
    save_metrics,
    print_metrics_summary,
)

# Disable GPU usage (force CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("=" * 70)
print("STEP 5 (Efficient): TRAINING MODEL 3 - LSTM with Advanced Balancing")
print("=" * 70)

# ----------------------------------------------------
# 1. Load preprocessed data
# ----------------------------------------------------
print("\n1. Loading preprocessed data...")

# Load CSV files
train_df = pd.read_csv("Data/Preprocessed/train_preprocessed.csv")
val_df = pd.read_csv("Data/Preprocessed/val_preprocessed.csv")
test_df = pd.read_csv("Data/Preprocessed/test_preprocessed.csv")

# Print dataset sizes
print("   Training samples   :", len(train_df))
print("   Validation samples :", len(val_df))
print("   Test samples       :", len(test_df))

# Extract text and labels from dataframes
X_train = train_df["text"].tolist()
y_train = train_df["label_encoded"].values

X_val = val_df["text"].tolist()
y_val = val_df["label_encoded"].values

X_test = test_df["text"].tolist()
y_test = test_df["label_encoded"].values

# Count number of classes
unique_classes = np.unique(y_train)
num_classes = len(unique_classes)
print("   Number of classes  :", num_classes)

# Show class distribution BEFORE balancing
print("\n   Class distribution (BEFORE balancing):")
class_counts_before = Counter(y_train)
for cls in sorted(unique_classes):
    count = class_counts_before[cls]
    pct = (count / len(y_train)) * 100
    print(f"      Class {cls}: {count} samples ({pct:.1f}%)")

# ----------------------------------------------------
# 2. Balance training data using random oversampling
# ----------------------------------------------------
print("\n2. Balancing training data using random oversampling...")

# Find the maximum class count
max_count = max(class_counts_before.values())
print(f"   Target samples per class: {max_count}")

# Oversample minority classes
X_train_balanced = []
y_train_balanced = []

for cls in unique_classes:
    # Get all samples for this class
    class_indices = np.where(y_train == cls)[0]
    class_samples = [X_train[i] for i in class_indices]
    class_labels = y_train[class_indices]
    
    # Add original samples
    X_train_balanced.extend(class_samples)
    y_train_balanced.extend(class_labels)
    
    # If this is a minority class, oversample it
    if len(class_samples) < max_count:
        # Calculate how many samples to add
        num_to_add = max_count - len(class_samples)
        
        # Randomly sample with replacement
        indices_to_add = np.random.choice(len(class_samples), size=num_to_add, replace=True)
        
        for idx in indices_to_add:
            X_train_balanced.append(class_samples[idx])
            y_train_balanced.append(class_labels[idx])
        
        print(f"   Class {cls}: {len(class_samples)} -> {len(class_samples) + num_to_add} samples")

# Convert to numpy arrays
X_train_balanced = np.array(X_train_balanced)
y_train_balanced = np.array(y_train_balanced)

# Shuffle the balanced dataset
shuffle_indices = np.random.permutation(len(y_train_balanced))
X_train_balanced = X_train_balanced[shuffle_indices]
y_train_balanced = y_train_balanced[shuffle_indices]

print(f"\n   Total training samples after balancing: {len(y_train_balanced)}")

# Show class distribution AFTER balancing
print("\n   Class distribution (AFTER balancing):")
class_counts_after = Counter(y_train_balanced)
for cls in sorted(unique_classes):
    count = class_counts_after[cls]
    pct = (count / len(y_train_balanced)) * 100
    print(f"      Class {cls}: {count} samples ({pct:.1f}%)")

# Convert back to list for LSTM (it expects list of strings)
X_train_balanced = X_train_balanced.tolist()

# ----------------------------------------------------
# 3. Calculate STRONG class weights (as backup)
# ----------------------------------------------------
print("\n3. Calculating STRONG class weights (as additional balancing)...")

# Compute balanced weights
weights_balanced = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train  # Use original distribution for weight calculation
)

# Apply additional multiplier for very imbalanced classes
class_counts_orig = np.array([class_counts_before[cls] for cls in unique_classes])
max_count_orig = class_counts_orig.max()
min_count_orig = class_counts_orig.min()

# Create stronger weights
weights_strong = []
for i, cls in enumerate(unique_classes):
    count = class_counts_before[cls]
    # Base balanced weight
    base_weight = weights_balanced[i]
    
    # If class is very small, boost weight more
    if count < max_count_orig * 0.2:
        weight = base_weight * 2.5  # Very strong boost
    elif count < max_count_orig * 0.4:
        weight = base_weight * 1.8  # Strong boost
    else:
        weight = base_weight * 1.2  # Moderate boost
    
    weights_strong.append(weight)

weights_array = np.array(weights_strong)

# Convert weights to dictionary
class_weights = {}
for cls, weight in zip(unique_classes, weights_array):
    class_weights[cls] = float(weight)
    count = class_counts_before[cls]
    print(f"   Class {cls}: weight={weight:.2f} (original samples: {count})")

# ----------------------------------------------------
# 4. Create optimized LSTM model
# ----------------------------------------------------
print("\n4. Creating optimized LSTM model...")
print("   Architecture: Bidirectional LSTM with attention-like features")

# Create LSTM model with optimized hyperparameters
lstm_model = LSTMClassifier(
    max_features=20000,      # Increased vocabulary
    max_length=150,          # Slightly longer sequences
    embedding_dim=128,       # Standard embedding
    lstm_units=128,          # Standard LSTM units
    num_classes=num_classes,
    bidirectional=True,      # Use bidirectional LSTM
    spatial_dropout=0.3      # Moderate dropout
)

# ----------------------------------------------------
# 5. Train model with optimized settings
# ----------------------------------------------------
print("\n5. Training LSTM model with optimized settings...")
print("   - Balanced training data (oversampled)")
print("   - Strong class weights applied")
print("   - Learning rate scheduling")
print("   - Early stopping with patience")
print("   - This may take 45-60 minutes...")

# Train the model with more epochs and better monitoring
history = lstm_model.train(
    X_train_balanced,  # Use balanced data
    y_train_balanced,  # Use balanced labels
    X_val,
    y_val,
    epochs=30,          # More epochs
    batch_size=64,      # Larger batch size for efficiency
    verbose=1,
    class_weight=class_weights  # Still apply class weights
)

# ----------------------------------------------------
# 6. Evaluate on test set
# ----------------------------------------------------
print("\n6. Evaluating on TEST set...")

# Predict class labels
preds = lstm_model.predict(X_test)

# Predict probabilities
probs = lstm_model.predict_proba(X_test)

# Compute all evaluation metrics
metrics = calculate_all_metrics(
    y_test,
    preds,
    probs,
    labels=list(range(num_classes))
)

# Print summary in simple format
print_metrics_summary(metrics)

# Check if model is predicting multiple classes
unique_preds = np.unique(preds)
print(f"\n   Model predicts {len(unique_preds)} different classes: {unique_preds}")
if len(unique_preds) == 1:
    print("   WARNING: Model still predicting only one class!")
else:
    print("   SUCCESS: Model is predicting multiple classes")

# Check per-class performance
print("\n   Per-class predictions on test set:")
for cls in unique_classes:
    count = (preds == cls).sum()
    actual = (y_test == cls).sum()
    pct = (count / len(preds)) * 100 if len(preds) > 0 else 0
    print(f"      Class {cls}: {count} predicted (actual: {actual}, {pct:.1f}%)")

# ----------------------------------------------------
# 7. Save model, metrics, and plots
# ----------------------------------------------------
print("\n7. Saving model and results...")

# Create folders if not exist
models_dir = Path("Models/lstm")
plots_dir = Path("Outputs/plots")
results_dir = Path("Outputs/results")

models_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Save model (will overwrite previous)
lstm_model.save(models_dir)

# Save test metrics
save_metrics(metrics, results_dir / "lstm_test_metrics.json")

# Save confusion matrix plot
plot_confusion_matrix(
    y_test,
    preds,
    labels=list(range(num_classes)),
    save_path=plots_dir / "lstm_confusion_matrix.png",
    title="LSTM (Efficient) - Confusion Matrix (Test Set)"
)

# Save training curves (loss + accuracy)
plot_training_history(
    history,
    str(plots_dir),
    "lstm_efficient"
)

# ----------------------------------------------------
# Summary
# ----------------------------------------------------
print("\n" + "=" * 70)
print("MODEL 3 TRAINING COMPLETE - LSTM (EFFICIENT)")
print("=" * 70)
print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
print(f"Final Test F1-Score: {metrics['f1']:.4f}")
print(f"Final Test Precision: {metrics['precision']:.4f}")
print(f"Final Test Recall: {metrics['recall']:.4f}")

if metrics['accuracy'] > 0.50:
    print("\nSUCCESS: Model performance is good!")
elif metrics['accuracy'] > 0.40:
    print("\nPROGRESS: Model performance is acceptable")
elif len(unique_preds) > 1:
    print("\nPROGRESS: Model is learning (predicting multiple classes)")
    print("         Accuracy can be improved further")
else:
    print("\nWARNING: Model still needs work - consider further adjustments")

print("=" * 70)

