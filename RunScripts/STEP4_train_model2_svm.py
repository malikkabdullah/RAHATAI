"""
STEP 4: TRAIN MODEL 2 - SVM (Support Vector Machine)
This model is more powerful than Naive Bayes.
"""

import sys
import json
import pickle
import pandas as pd
from pathlib import Path

# Sklearn imports
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Custom metric functions
from Scripts.evaluation.metrics import (
    calculate_all_metrics,
    plot_confusion_matrix,
    save_metrics,
    print_metrics_summary
)


# ============================================================
# HEADER PRINT
# ============================================================
print("=" * 70)
print("STEP 4: TRAINING MODEL 2 - SVM (Support Vector Machine)")
print("=" * 70)


# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================
print("\n1. Loading preprocessed data...")

train_df = pd.read_csv("Data/Preprocessed/train_preprocessed.csv")
val_df   = pd.read_csv("Data/Preprocessed/val_preprocessed.csv")
test_df  = pd.read_csv("Data/Preprocessed/test_preprocessed.csv")

# Load label mapping file (dictionary)
with open("Data/Preprocessed/label_mappings.json", "r") as f:
    mapping_file = json.load(f)
    idx_to_label = {int(k): v for k, v in mapping_file["idx_to_label"].items()}

print("   Training samples:", len(train_df))
print("   Validation samples:", len(val_df))
print("   Test samples:", len(test_df))
print("   Number of classes:", len(idx_to_label))


# ============================================================
# 2. SEPARATE FEATURES (TEXT) AND LABELS
# ============================================================
print("\n2. Preparing data for training...")

X_train = train_df["text"].values
y_train = train_df["label_encoded"].values

X_val = val_df["text"].values
y_val = val_df["label_encoded"].values

X_test = test_df["text"].values
y_test = test_df["label_encoded"].values

print("   Training texts:", len(X_train))
print("   Training labels:", len(y_train))


# ============================================================
# 3. CREATE SVM MODEL (TF-IDF + SVM)
# ============================================================
print("\n3. Creating SVM model...")
print("   - TF-IDF converts text to numerical features")
print("   - SVM with linear kernel (best for text classification)")
print("   - probability=True allows probability scores")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )),
    ("svm", SVC(
        C=1.0,
        kernel="linear",
        probability=True,
        random_state=42
    ))
])


# ============================================================
# 4. TRAIN THE MODEL
# ============================================================
print("\n4. Training model...")
print("   (SVM is slower than Naive Bayes — may take several minutes)")

model.fit(X_train, y_train)

print("   Training complete!")


# ============================================================
# 5. VALIDATION SET EVALUATION
# ============================================================
print("\n5. Evaluating on validation set...")

val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)

val_metrics = calculate_all_metrics(
    y_val,
    val_preds,
    val_probs,
    labels=list(range(len(idx_to_label)))
)

print("\n   Validation Results:")
print_metrics_summary(val_metrics)


# ============================================================
# 6. TEST SET EVALUATION
# ============================================================
print("\n6. Evaluating on TEST set...")

test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)

test_metrics = calculate_all_metrics(
    y_test,
    test_preds,
    test_probs,
    labels=list(range(len(idx_to_label)))
)

print("\n   Test Set Results:")
print_metrics_summary(test_metrics)


# ============================================================
# 7. SAVE MODEL, METRICS, AND CONFUSION MATRIX
# ============================================================
print("\n7. Saving model and results...")

Path("Models").mkdir(parents=True, exist_ok=True)
Path("Outputs/results").mkdir(parents=True, exist_ok=True)
Path("Outputs/plots").mkdir(parents=True, exist_ok=True)

model_file = Path("Models/svm.pkl")

with open(model_file, "wb") as f:
    pickle.dump(model, f)

print("   Model saved to:", model_file)

save_metrics(test_metrics, "Outputs/results/svm_test_metrics.json")
save_metrics(val_metrics, "Outputs/results/svm_val_metrics.json")

plot_confusion_matrix(
    y_test,
    test_preds,
    labels=list(range(len(idx_to_label))),
    save_path="Outputs/plots/svm_confusion_matrix.png",
    title="SVM - Confusion Matrix (Test Set)"
)


# ============================================================
# 8. COMPARE WITH NAIVE BAYES
# ============================================================
print("\n8. Comparing with Naive Bayes...")

try:
    with open("Outputs/results/naive_bayes_test_metrics.json", "r") as f:
        nb_metrics = json.load(f)

    print("\n   Model Comparison (Test Set):")
    print(f"   {'Metric':<20} {'Naive Bayes':<15} {'SVM':<15}")
    print("   " + "-" * 50)

    print(f"   {'Accuracy':<20} {nb_metrics['accuracy']:<15.4f} {test_metrics['accuracy']:<15.4f}")
    print(f"   {'F1-Score':<20} {nb_metrics['f1']:<15.4f} {test_metrics['f1']:<15.4f}")
    print(f"   {'Precision':<20} {nb_metrics['precision']:<15.4f} {test_metrics['precision']:<15.4f}")
    print(f"   {'Recall':<20} {nb_metrics['recall']:<15.4f} {test_metrics['recall']:<15.4f}")
    print(f"   {'AUC-ROC':<20} {nb_metrics['auc']:<15.4f} {test_metrics['auc']:<15.4f}")

    if test_metrics["accuracy"] > nb_metrics["accuracy"]:
        print("\n   SVM performs better than Naive Bayes!")
    elif test_metrics["accuracy"] < nb_metrics["accuracy"]:
        print("\n   Warning: Naive Bayes performs better than SVM")
    else:
        print("\n   Both models perform similarly")

except Exception:
    print("   (Could not load Naive Bayes results for comparison)")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("MODEL 2 TRAINING COMPLETE - SVM")
print("=" * 70)

print(f"""
Summary:
- Validation Accuracy: {val_metrics['accuracy']:.4f}
- Test Accuracy: {test_metrics['accuracy']:.4f}
- Test F1-Score: {test_metrics['f1']:.4f}

Saved Files:
- Model: Models/svm.pkl
- Metrics JSON: Outputs/results/
- Confusion Matrix: Outputs/plots/svm_confusion_matrix.png

Next Step:
→ Train Model 3 (LSTM Deep Learning)
""")

print("=" * 70)
