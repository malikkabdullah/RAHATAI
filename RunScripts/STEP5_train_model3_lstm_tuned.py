"""
STEP 5 (Tuned): TRAIN MODEL 3 - LSTM
Very simple and easy-to-read version.
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
print("STEP 5 (Tuned): TRAINING MODEL 3 - LSTM")
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

# ----------------------------------------------------
# 2. Calculate class weights
# ----------------------------------------------------
print("\n2. Calculating class weights for imbalanced classes...")

# Compute weights
weights_array = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train
)

# Convert weights to dictionary
class_weights = {}
for cls, weight in zip(unique_classes, weights_array):
    class_weights[cls] = weight
    print(f"   Class {cls}: weight={weight:.2f}")

# ----------------------------------------------------
# 3. Create tuned LSTM model
# ----------------------------------------------------
print("\n3. Creating tuned LSTM model...")

# Create LSTM model with chosen hyperparameters
lstm_model = LSTMClassifier(
    max_features=20000,
    max_length=160,
    embedding_dim=256,
    lstm_units=256,
    num_classes=num_classes,
    bidirectional=True,
    spatial_dropout=0.3
)

# ----------------------------------------------------
# 4. Train model
# ----------------------------------------------------
print("\n4. Training LSTM model (this may take a while)...")

# Train the model step-by-step
history = lstm_model.train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=18,
    batch_size=64,
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

# Save model
lstm_model.save(models_dir)

# Save test metrics
save_metrics(metrics, results_dir / "lstm_test_metrics.json")

# Save confusion matrix plot
plot_confusion_matrix(
    y_test,
    preds,
    labels=list(range(num_classes)),
    save_path=plots_dir / "lstm_confusion_matrix.png",
    title="LSTM (Tuned) - Confusion Matrix (Test Set)"
)

# Save training curves (loss + accuracy)
plot_training_history(
    history,
    str(plots_dir),
    "lstm_tuned"
)

# ----------------------------------------------------
# Summary
# ----------------------------------------------------
print("\n" + "=" * 70)
print("MODEL 3 TRAINING COMPLETE - LSTM (TUNED)")
print("=" * 70)
