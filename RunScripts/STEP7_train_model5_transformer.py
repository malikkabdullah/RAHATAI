"""
STEP 7: TRAIN MODEL 5 - TRANSFORMER (XLM-RoBERTa)
Multilingual transformer model for text classification
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
from Scripts.classification.transformer_models import TransformerClassifier
from Scripts.evaluation.metrics import (
    calculate_all_metrics,
    plot_confusion_matrix,
    plot_training_history,
    save_metrics,
    print_metrics_summary,
)

# Force CPU (since GPU may not be available)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("=" * 70)
print("STEP 7: TRAINING MODEL 5 - TRANSFORMER (XLM-RoBERTa)")
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
# 2. Create Transformer model
# ----------------------------------------------------
print("\n2. Creating Transformer model (XLM-RoBERTa)...")

# Create transformer model
transformer_model = TransformerClassifier(
    model_name="xlm-roberta-base",
    num_labels=num_classes,
    max_length=128,
    device="cpu"  # Force CPU
)

# ----------------------------------------------------
# 3. Train model
# ----------------------------------------------------
print("\n3. Training Transformer model (this may take a while)...")
print("   Note: First run will download the model (~1GB)")

# Train the model
history = transformer_model.train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=5,  # Fewer epochs for transformer (they learn faster)
    batch_size=16,  # Smaller batch size for transformers
    learning_rate=2e-5,
    output_dir="Models/transformer"
)

# ----------------------------------------------------
# 4. Evaluate on test set
# ----------------------------------------------------
print("\n4. Evaluating on TEST set...")

# Predict class labels
preds = transformer_model.predict(X_test, batch_size=16)

# Predict probabilities
probs = transformer_model.predict_proba(X_test, batch_size=16)

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
models_dir = Path("Models/transformer")
plots_dir = Path("Outputs/plots")
results_dir = Path("Outputs/results")

models_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Save model (already saved during training, but save again to be sure)
transformer_model.save(str(models_dir))

# Save test metrics
save_metrics(metrics, results_dir / "transformer_test_metrics.json")

# Save confusion matrix plot
plot_confusion_matrix(
    y_test,
    preds,
    labels=list(range(num_classes)),
    save_path=plots_dir / "transformer_confusion_matrix.png",
    title="Transformer (XLM-RoBERTa) - Confusion Matrix (Test Set)"
)

# Save training curves if history available
if history:
    plot_training_history(
        history,
        str(plots_dir),
        "transformer"
    )

# ----------------------------------------------------
# Summary
# ----------------------------------------------------
print("\n" + "=" * 70)
print("MODEL 5 TRAINING COMPLETE - TRANSFORMER (XLM-RoBERTa)")
print("=" * 70)



