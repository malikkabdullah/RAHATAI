"""
STEP 5 (Improved): TRAIN MODEL 3 - LSTM with Better Class Balancing
This version uses stronger class weights and improved training strategy
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
print("STEP 5 (Improved): TRAINING MODEL 3 - LSTM with Better Balancing")
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

# Show class distribution
print("\n   Class distribution:")
for cls in unique_classes:
    count = (y_train == cls).sum()
    pct = (count / len(y_train)) * 100
    print(f"      Class {cls}: {count} samples ({pct:.1f}%)")

# ----------------------------------------------------
# 2. Calculate STRONGER class weights
# ----------------------------------------------------
print("\n2. Calculating STRONGER class weights for imbalanced classes...")

# Method 1: Compute balanced weights
weights_balanced = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train
)

# Method 2: Calculate custom weights (inverse frequency with multiplier)
class_counts = np.array([(y_train == cls).sum() for cls in unique_classes])
total_samples = len(y_train)
max_count = class_counts.max()

# Stronger weighting: multiply balanced weights by 2 for minority classes
weights_custom = []
for cls in unique_classes:
    count = (y_train == cls).sum()
    # Inverse frequency weighting
    weight = total_samples / (num_classes * count)
    # Boost minority classes even more
    if count < max_count * 0.3:  # If class has less than 30% of max
        weight = weight * 2.0  # Double the weight
    weights_custom.append(weight)

# Use the stronger custom weights
weights_array = np.array(weights_custom)

# Convert weights to dictionary
class_weights = {}
for cls, weight in zip(unique_classes, weights_array):
    class_weights[cls] = float(weight)
    count = (y_train == cls).sum()
    print(f"   Class {cls}: weight={weight:.2f} (samples: {count})")

# ----------------------------------------------------
# 3. Create improved LSTM model
# ----------------------------------------------------
print("\n3. Creating improved LSTM model...")
print("   Using smaller model to prevent overfitting to majority class")

# Create LSTM model with adjusted hyperparameters
# Smaller model to force learning, not just memorization
lstm_model = LSTMClassifier(
    max_features=15000,  # Reduced from 20000
    max_length=128,      # Reduced from 160
    embedding_dim=128,   # Reduced from 256
    lstm_units=128,      # Reduced from 256
    num_classes=num_classes,
    bidirectional=True,
    spatial_dropout=0.4  # Increased dropout for regularization
)

# ----------------------------------------------------
# 4. Train model with improved settings
# ----------------------------------------------------
print("\n4. Training LSTM model with improved settings...")
print("   - Stronger class weights applied")
print("   - More epochs for better learning")
print("   - This may take 40-50 minutes...")

# Train the model with more epochs and better monitoring
history = lstm_model.train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=25,        # More epochs
    batch_size=32,   # Smaller batch size for better gradient updates
    verbose=1,
    class_weight=class_weights
)

# ----------------------------------------------------
# 5. Evaluate on test set
# ----------------------------------------------------
print("\n5. Evaluating on TEST set...")

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
    print("   Good: Model is predicting multiple classes")

# ----------------------------------------------------
# 6. Save model, metrics, and plots
# ----------------------------------------------------
print("\n6. Saving model and results...")

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
    title="LSTM (Improved) - Confusion Matrix (Test Set)"
)

# Save training curves (loss + accuracy)
plot_training_history(
    history,
    str(plots_dir),
    "lstm_improved"
)

# ----------------------------------------------------
# Summary
# ----------------------------------------------------
print("\n" + "=" * 70)
print("MODEL 3 TRAINING COMPLETE - LSTM (IMPROVED)")
print("=" * 70)
print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
print(f"Final Test F1-Score: {metrics['f1']:.4f}")

if metrics['accuracy'] > 0.40:
    print("\nSUCCESS: Model performance improved!")
elif len(unique_preds) > 1:
    print("\nPROGRESS: Model is learning (predicting multiple classes)")
    print("         But accuracy still needs improvement")
else:
    print("\nWARNING: Model still needs work - consider further adjustments")

print("=" * 70)


