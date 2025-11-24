"""
STEP 3: TRAIN MODEL 1 - NAIVE BAYES
This is the simplest ML model. We train it using TF-IDF + Naive Bayes.
"""

import sys
import json
import pickle
import pandas as pd
from pathlib import Path

# Sklearn imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Custom imports
from Scripts.evaluation.metrics import (
    calculate_all_metrics,
    plot_confusion_matrix,
    save_metrics,
    print_metrics_summary
)

# =====================================================
# PRINT STEP HEADER
# =====================================================
print("=" * 70)
print("STEP 3: TRAINING MODEL 1 - NAIVE BAYES")
print("=" * 70)


# =====================================================
# 1. LOAD THE PREPROCESSED DATA
# =====================================================
print("\n1. Loading preprocessed data...")

# Load CSV files
train_df = pd.read_csv("Data/Preprocessed/train_preprocessed.csv")
val_df   = pd.read_csv("Data/Preprocessed/val_preprocessed.csv")
test_df  = pd.read_csv("Data/Preprocessed/test_preprocessed.csv")

# Load label mappings (dictionary)
with open("Data/Preprocessed/label_mappings.json", "r") as f:
    mappings = json.load(f)
    idx_to_label = {int(k): v for k, v in mappings["idx_to_label"].items()}

# Print shapes
print("   Training samples:", len(train_df))
print("   Validation samples:", len(val_df))
print("   Test samples:", len(test_df))
print("   Number of classes:", len(idx_to_label))


# =====================================================
# 2. SEPARATE TEXT & LABELS
# =====================================================
print("\n2. Preparing data for training...")

# Training data
X_train = train_df["text"].values
y_train = train_df["label_encoded"].values

# Validation data
X_val = val_df["text"].values
y_val = val_df["label_encoded"].values

# Test data
X_test = test_df["text"].values
y_test = test_df["label_encoded"].values

# Print counts
print("   Training texts:", len(X_train))
print("   Training labels:", len(y_train))


# =====================================================
# 3. CREATE THE MODEL (TF-IDF + NAIVE BAYES)
# =====================================================
print("\n3. Creating Naive Bayes model...")
print("   - First convert text to TF-IDF features")
print("   - Then train Multinomial Naive Bayes")

# Create a full pipeline model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )),
    ("nb", MultinomialNB())
])


# =====================================================
# 4. TRAIN THE MODEL
# =====================================================
print("\n4. Training model...")
print("   Training... (might take a few minutes)")

# Train the model
model.fit(X_train, y_train)

print("   Training complete!")


# =====================================================
# 5. VALIDATION EVALUATION
# =====================================================
print("\n5. Evaluating on validation set...")

# Predictions
val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)

# Metrics
val_metrics = calculate_all_metrics(
    y_val,
    val_preds,
    val_probs,
    labels=list(range(len(idx_to_label)))
)

# Print summary
print("\n   Validation Set Results:")
print_metrics_summary(val_metrics)


# =====================================================
# 6. TEST SET EVALUATION
# =====================================================
print("\n6. Evaluating on TEST set...")

# Predictions
test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)

# Metrics
test_metrics = calculate_all_metrics(
    y_test,
    test_preds,
    test_probs,
    labels=list(range(len(idx_to_label)))
)

# Print summary
print("\n   TEST Set Results:")
print_metrics_summary(test_metrics)


# =====================================================
# 7. SAVE MODEL, METRICS & PLOTS
# =====================================================
print("\n7. Saving model and results...")

# Make directories
Path("Outputs/results").mkdir(parents=True, exist_ok=True)
Path("Outputs/plots").mkdir(parents=True, exist_ok=True)
Path("Models").mkdir(parents=True, exist_ok=True)

# Save model file
model_file = Path("Models/naive_bayes.pkl")
with open(model_file, "wb") as f:
    pickle.dump(model, f)

print("   Model saved to:", model_file)

# Save metrics
save_metrics(test_metrics, "Outputs/results/naive_bayes_test_metrics.json")
save_metrics(val_metrics, "Outputs/results/naive_bayes_val_metrics.json")

# Save confusion matrix plot
plot_confusion_matrix(
    y_test,
    test_preds,
    labels=list(range(len(idx_to_label))),
    save_path="Outputs/plots/naive_bayes_confusion_matrix.png",
    title="Naive Bayes - Confusion Matrix (Test Set)"
)


# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "=" * 70)
print("MODEL 1 TRAINING COMPLETE - NAIVE BAYES")
print("=" * 70)

print(f"""
Results Summary:
- Validation Accuracy: {val_metrics['accuracy']:.4f}
- Test Accuracy: {test_metrics['accuracy']:.4f}
- Test F1-Score: {test_metrics['f1']:.4f}

Saved Files:
- Model: Models/naive_bayes.pkl
- Metrics JSON: Outputs/results/
- Confusion Matrix: Outputs/plots/naive_bayes_confusion_matrix.png

Next Step:
→ Train Model 2 (SVM Model)
""")

print("=" * 70)
