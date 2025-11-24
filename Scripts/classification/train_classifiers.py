"""
Simplified Main Training Script for Rahat AI Classification Models
This version is written in a very simple and readable way.
"""

import sys
from pathlib import Path

# =========================================================
# Add the project root so imports work
# =========================================================
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# =========================================================
# Import basic libraries
# =========================================================
import numpy as np
import pandas as pd
import json

# =========================================================
# Import project utilities
# =========================================================
from Scripts.utils import load_config, load_crisisnlp_datasets
from Scripts.data_preprocessing.preprocessor import (
    MultilingualPreprocessor,
    prepare_classification_data
)

# Machine Learning models
from Scripts.classification.ml_models import NaiveBayesClassifier, SVMClassifier

# Deep Learning models
from Scripts.classification.dl_models import LSTMClassifier, CNNClassifier

# Transformer model
from Scripts.classification.transformer_models import TransformerClassifier

# Evaluation utilities
from Scripts.evaluation.metrics import (
    calculate_all_metrics,
    plot_confusion_matrix,
    plot_training_history,
    save_metrics,
    print_metrics_summary
)


# ====================================================================
#                     MAIN TRAINING FUNCTION
# ====================================================================

def train_all_models(config_path="config/config.yaml"):
    """
    Train all required ML, DL, and Transformer models.
    This version is written step-by-step in simple readable code.
    """

    print("=" * 70)
    print("RAHAT AI - TRAINING ALL MODELS")
    print("=" * 70)

    # ------------------------------------------------------------
    # LOAD CONFIG FILE
    # ------------------------------------------------------------
    config = load_config(config_path)
    cls_config = config["classification"]

    # ------------------------------------------------------------
    # LOAD DATASETS
    # ------------------------------------------------------------
    print("\nLoading CrisisNLP datasets...")
    train_df, dev_df, test_df = load_crisisnlp_datasets()

    # ------------------------------------------------------------
    # PREPROCESS TEXT DATA
    # ------------------------------------------------------------
    print("\nPreprocessing text...")
    preprocessor = MultilingualPreprocessor()

    # Prepare train/dev/test separately
    train_data = prepare_classification_data(train_df, "item", "label", preprocessor)
    dev_data = prepare_classification_data(dev_df, "item", "label", preprocessor)
    test_data = prepare_classification_data(test_df, "item", "label", preprocessor)

    # ------------------------------------------------------------
    # Keep label mapping consistent
    # ------------------------------------------------------------
    label_to_idx = train_data["label_to_idx"]
    idx_to_label = train_data["idx_to_label"]

    dev_data["encoded_labels"] = np.array([label_to_idx.get(lbl, 0) for lbl in dev_data["labels"]])
    test_data["encoded_labels"] = np.array([label_to_idx.get(lbl, 0) for lbl in test_data["labels"]])

    # ------------------------------------------------------------
    # PRINT BASIC DATA INFO
    # ------------------------------------------------------------
    print("\nDataset Information:")
    print(f"  Train samples: {len(train_data['texts'])}")
    print(f"  Dev samples:   {len(dev_data['texts'])}")
    print(f"  Test samples:  {len(test_data['texts'])}")
    print(f"  Total classes: {train_data['num_classes']}")
    print(f"  Class names:   {list(label_to_idx.keys())}")

    # ------------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # ------------------------------------------------------------
    results_dir = Path(config["outputs"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(config["outputs"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(config["models"]["save_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Store results for all models
    all_results = {}

    # =====================================================================
    #                   TRAIN MACHINE LEARNING MODELS
    # =====================================================================
    print("\n" + "=" * 70)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 70)

    # ************************************************************
    # 1. NAIVE BAYES
    # ************************************************************
    print("\n[1/5] Training Naive Bayes...")
    nb_model = NaiveBayesClassifier()
    nb_model.train(train_data["texts"], train_data["encoded_labels"])

    nb_preds = nb_model.predict(test_data["texts"])
    nb_probs = nb_model.predict_proba(test_data["texts"])

    nb_metrics = calculate_all_metrics(
        test_data["encoded_labels"],
        nb_preds,
        nb_probs,
        labels=list(range(train_data["num_classes"]))
    )

    all_results["naive_bayes"] = nb_metrics
    print_metrics_summary(nb_metrics)

    nb_model.save(models_dir / "naive_bayes.pkl")
    save_metrics(nb_metrics, results_dir / "naive_bayes_metrics.json")

    plot_confusion_matrix(
        test_data["encoded_labels"],
        nb_preds,
        labels=list(range(train_data["num_classes"])),
        save_path=plots_dir / "naive_bayes_confusion_matrix.png",
        title="Naive Bayes – Confusion Matrix"
    )

    # ************************************************************
    # 2. SVM MODEL
    # ************************************************************
    print("\n[2/5] Training SVM...")
    svm_model = SVMClassifier()
    svm_model.train(train_data["texts"], train_data["encoded_labels"])

    svm_preds = svm_model.predict(test_data["texts"])
    svm_probs = svm_model.predict_proba(test_data["texts"])

    svm_metrics = calculate_all_metrics(
        test_data["encoded_labels"],
        svm_preds,
        svm_probs,
        labels=list(range(train_data["num_classes"]))
    )

    all_results["svm"] = svm_metrics
    print_metrics_summary(svm_metrics)

    svm_model.save(models_dir / "svm.pkl")
    save_metrics(svm_metrics, results_dir / "svm_metrics.json")

    plot_confusion_matrix(
        test_data["encoded_labels"],
        svm_preds,
        labels=list(range(train_data["num_classes"])),
        save_path=plots_dir / "svm_confusion_matrix.png",
        title="SVM – Confusion Matrix"
    )

    # =====================================================================
    #                   TRAIN DEEP LEARNING MODELS
    # =====================================================================
    print("\n" + "=" * 70)
    print("TRAINING DEEP LEARNING MODELS")
    print("=" * 70)

    # ************************************************************
    # 3. LSTM CLASSIFIER
    # ************************************************************
    print("\n[3/5] Training LSTM...")
    lstm_model = LSTMClassifier(
        max_length=cls_config.get("max_length", 128),
        num_classes=train_data["num_classes"]
    )

    lstm_history = lstm_model.train(
        train_data["texts"], train_data["encoded_labels"],
        dev_data["texts"], dev_data["encoded_labels"],
        epochs=cls_config.get("epochs", 50),
        batch_size=cls_config.get("batch_size", 32)
    )

    plot_training_history(lstm_history, str(plots_dir), "lstm")

    lstm_preds = lstm_model.predict(test_data["texts"])
    lstm_probs = lstm_model.predict_proba(test_data["texts"])

    lstm_metrics = calculate_all_metrics(
        test_data["encoded_labels"],
        lstm_preds,
        lstm_probs,
        labels=list(range(train_data["num_classes"]))
    )

    all_results["lstm"] = lstm_metrics
    print_metrics_summary(lstm_metrics)

    lstm_model.save(models_dir / "lstm")
    save_metrics(lstm_metrics, results_dir / "lstm_metrics.json")

    plot_confusion_matrix(
        test_data["encoded_labels"],
        lstm_preds,
        labels=list(range(train_data["num_classes"])),
        save_path=plots_dir / "lstm_confusion_matrix.png",
        title="LSTM – Confusion Matrix"
    )

    # ************************************************************
    # 4. CNN CLASSIFIER
    # ************************************************************
    print("\n[4/5] Training CNN...")
    cnn_model = CNNClassifier(
        max_length=cls_config.get("max_length", 128),
        num_classes=train_data["num_classes"]
    )

    cnn_history = cnn_model.train(
        train_data["texts"], train_data["encoded_labels"],
        dev_data["texts"], dev_data["encoded_labels"],
        epochs=cls_config.get("epochs", 50),
        batch_size=cls_config.get("batch_size", 32)
    )

    plot_training_history(cnn_history, str(plots_dir), "cnn")

    cnn_preds = cnn_model.predict(test_data["texts"])
    cnn_probs = cnn_model.predict_proba(test_data["texts"])

    cnn_metrics = calculate_all_metrics(
        test_data["encoded_labels"],
        cnn_preds,
        cnn_probs,
        labels=list(range(train_data["num_classes"]))
    )

    all_results["cnn"] = cnn_metrics
    print_metrics_summary(cnn_metrics)

    cnn_model.save(models_dir / "cnn")
    save_metrics(cnn_metrics, results_dir / "cnn_metrics.json")

    plot_confusion_matrix(
        test_data["encoded_labels"],
        cnn_preds,
        labels=list(range(train_data["num_classes"])),
        save_path=plots_dir / "cnn_confusion_matrix.png",
        title="CNN – Confusion Matrix"
    )

    # =====================================================================
    #                   TRAIN TRANSFORMER MODEL
    # =====================================================================
    print("\n" + "=" * 70)
    print("TRAINING TRANSFORMER MODEL")
    print("=" * 70)

    # ************************************************************
    # 5. TRANSFORMER (XLM-RoBERTa)
    # ************************************************************
    print("\n[5/5] Training Transformer Model...")
    transformer_model = TransformerClassifier(
        model_name=cls_config["transformer_models"][0],
        num_labels=train_data["num_classes"],
        max_length=cls_config.get("max_length", 128),
        device=config["general"].get("device", "cuda")
    )

    transformer_history = transformer_model.train(
        train_data["texts"], train_data["encoded_labels"],
        dev_data["texts"], dev_data["encoded_labels"],
        epochs=cls_config.get("epochs", 10),
        batch_size=cls_config.get("batch_size", 16),
        learning_rate=cls_config.get("learning_rate", 2e-5),
        output_dir=str(models_dir / "transformer")
    )

    if transformer_history:
        plot_training_history(transformer_history, str(plots_dir), "transformer")

    transformer_preds = transformer_model.predict(test_data["texts"])
    transformer_probs = transformer_model.predict_proba(test_data["texts"])

    transformer_metrics = calculate_all_metrics(
        test_data["encoded_labels"],
        transformer_preds,
        transformer_probs,
        labels=list(range(train_data["num_classes"]))
    )

    all_results["transformer"] = transformer_metrics
    print_metrics_summary(transformer_metrics)

    transformer_model.save(models_dir / "transformer")
    save_metrics(transformer_metrics, results_dir / "transformer_metrics.json")

    plot_confusion_matrix(
        test_data["encoded_labels"],
        transformer_preds,
        labels=list(range(train_data["num_classes"])),
        save_path=plots_dir / "transformer_confusion_matrix.png",
        title="Transformer – Confusion Matrix"
    )

    # =====================================================================
    #                        FINAL SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE – SUMMARY")
    print("=" * 70)

    summary_table = pd.DataFrame({
        "Model": ["Naive Bayes", "SVM", "LSTM", "CNN", "Transformer"],
        "Accuracy": [
            all_results["naive_bayes"]["accuracy"],
            all_results["svm"]["accuracy"],
            all_results["lstm"]["accuracy"],
            all_results["cnn"]["accuracy"],
            all_results["transformer"]["accuracy"]
        ],
        "F1 Score": [
            all_results["naive_bayes"]["f1"],
            all_results["svm"]["f1"],
            all_results["lstm"]["f1"],
            all_results["cnn"]["f1"],
            all_results["transformer"]["f1"]
        ]
    })

    print("\nModel Comparison Table:")
    print(summary_table.to_string(index=False))

    summary_table.to_csv(results_dir / "model_comparison.csv", index=False)

    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll results saved successfully.")
    print("=" * 70)


# =========================================================
# Run the training function
# =========================================================
if __name__ == "__main__":
    train_all_models()
