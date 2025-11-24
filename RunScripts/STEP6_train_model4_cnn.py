"""
STEP 6: TRAIN MODEL 4 - CNN
Simple CNN classifier for text classification
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

# Importing custom-built scripts
from Scripts.classification.dl_models import CNNClassifier
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
print("STEP 6: TRAINING MODEL 4 - CNN")
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
# 2. Create CNN model
# ----------------------------------------------------
print("\n2. Creating CNN model...")

# Create CNN model with standard hyperparameters
cnn_model = CNNClassifier(
    max_features=20000,
    max_length=160,
    embedding_dim=128,
    filter_sizes=[3, 4, 5],
    num_filters=128,
    num_classes=num_classes
)

# ----------------------------------------------------
# 3. Train model
# ----------------------------------------------------
print("\n3. Training CNN model (this may take a while)...")

# Train the model
history = cnn_model.train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    batch_size=64,
    verbose=1
)

# ----------------------------------------------------
# 4. Evaluate on test set
# ----------------------------------------------------
print("\n4. Evaluating on TEST set...")

# Predict class labels
preds = cnn_model.predict(X_test)

# Predict probabilities
probs = cnn_model.predict_proba(X_test)

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
# 5. Save model, metrics, and plots
# ----------------------------------------------------
print("\n5. Saving model and results...")

# Create folders if not exist
models_dir = Path("Models/cnn")
plots_dir = Path("Outputs/plots")
results_dir = Path("Outputs/results")

models_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Save model
cnn_model.save(models_dir)

# Save test metrics
save_metrics(metrics, results_dir / "cnn_test_metrics.json")

# Save confusion matrix plot
plot_confusion_matrix(
    y_test,
    preds,
    labels=list(range(num_classes)),
    save_path=plots_dir / "cnn_confusion_matrix.png",
    title="CNN - Confusion Matrix (Test Set)"
)

# Save training curves (loss + accuracy)
plot_training_history(
    history,
    str(plots_dir),
    "cnn"
)

# ----------------------------------------------------
# Summary
# ----------------------------------------------------
print("\n" + "=" * 70)
print("MODEL 4 TRAINING COMPLETE - CNN")
print("=" * 70)



