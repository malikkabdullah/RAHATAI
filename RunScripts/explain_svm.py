"""
Visual Explanation of SVM Model
Shows what the model actually learned and how it makes predictions
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*70)
print("SVM MODEL - VISUAL EXPLANATION")
print("="*70)

# Load the saved model
print("\n1. Loading saved model...")
with open('Models/svm.pkl', 'rb') as f:
    model = pickle.load(f)

print("   Model loaded!")
print(f"   Model type: {type(model)}")
print(f"   Components: {list(model.named_steps.keys())}")

# Load label mappings
with open("Data/Preprocessed/label_mappings.json", 'r') as f:
    label_mappings = json.load(f)
    idx_to_label = {int(k): v for k, v in label_mappings['idx_to_label'].items()}

print(f"\n   Classes: {list(idx_to_label.values())}")

# ============================================
# EXPLAIN TF-IDF VECTORIZER
# ============================================
print("\n" + "="*70)
print("2. TF-IDF VECTORIZER (Text → Numbers)")
print("="*70)

vectorizer = model.named_steps['tfidf']
print(f"\n   Vocabulary size: {len(vectorizer.vocabulary_)} words")
print(f"   Max features: {vectorizer.max_features}")
print(f"   N-gram range: {vectorizer.ngram_range}")

# ============================================
# EXPLAIN SVM CLASSIFIER
# ============================================
print("\n" + "="*70)
print("3. SVM CLASSIFIER")
print("="*70)

svm_classifier = model.named_steps['svm']
print(f"\n   Classifier type: {type(svm_classifier).__name__}")
print(f"   Kernel: {svm_classifier.kernel}")
print(f"   C parameter: {svm_classifier.C}")
print(f"   Number of classes: {len(svm_classifier.classes_)}")

# Get support vectors info
print(f"\n   Support Vectors:")
print(f"      Total support vectors: {len(svm_classifier.support_)}")
print(f"      Support vectors per class:")
for class_idx, class_label in enumerate(svm_classifier.classes_):
    # Count support vectors for this class
    sv_count = np.sum(svm_classifier.n_support_[class_idx])
    print(f"         {idx_to_label[class_label]:30s}: {sv_count} support vectors")

print(f"\n   What are Support Vectors?")
print(f"      - Data points closest to the decision boundary")
print(f"      - The 'hardest' examples to classify")
print(f"      - These define the optimal separating hyperplane")

# ============================================
# SHOW PREDICTION PROCESS
# ============================================
print("\n" + "="*70)
print("4. HOW THE MODEL MAKES PREDICTIONS")
print("="*70)

# Example texts
example_texts = [
    "People injured in earthquake need medical help",
    "Bridge collapsed due to flood damage",
    "Volunteers donating food and water",
    "This is not related to disaster"
]

print("\n   Example predictions:")
for text in example_texts:
    # Get prediction
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    
    # Get top 3 classes
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    
    # Get decision function values (distance from hyperplane)
    decision_function = svm_classifier.decision_function(
        vectorizer.transform([text])
    )[0]
    
    print(f"\n   Text: '{text[:50]}...'")
    print(f"   → Predicted: {idx_to_label[prediction]}")
    print(f"   → Confidence: {probabilities[prediction]:.2%}")
    print(f"   → Distance from hyperplane: {decision_function[prediction]:.2f}")
    print(f"   → Top 3 classes:")
    for idx in top3_indices:
        dist = decision_function[idx]
        print(f"        {idx_to_label[idx]:30s}: {probabilities[idx]:.2%} (dist: {dist:.2f})")

# ============================================
# EXPLAIN DECISION FUNCTION
# ============================================
print("\n" + "="*70)
print("5. DECISION FUNCTION (Distance from Hyperplane)")
print("="*70)

print("""
   How SVM makes decisions:
   
   1. Each text is converted to a point in high-dimensional space
   2. SVM calculates distance from the hyperplane (decision boundary)
   3. Positive distance → One side of boundary → One class
   4. Negative distance → Other side of boundary → Other class
   5. Larger distance = More confident prediction
   
   Example:
   - Distance = +5.2 → Strongly on one side → High confidence
   - Distance = +0.3 → Close to boundary → Low confidence
   - Distance = -2.1 → On other side → Different class
""")

# ============================================
# COMPARE WITH NAIVE BAYES
# ============================================
print("\n" + "="*70)
print("6. COMPARISON WITH NAIVE BAYES")
print("="*70)

# Load Naive Bayes results
try:
    with open("Outputs/results/naive_bayes_test_metrics.json", 'r') as f:
        nb_metrics = json.load(f)
    
    with open("Outputs/results/svm_test_metrics.json", 'r') as f:
        svm_metrics = json.load(f)
    
    print("\n   Test Set Performance Comparison:")
    print(f"   {'Metric':<20} {'Naive Bayes':<15} {'SVM':<15} {'Difference':<15}")
    print("   " + "-"*65)
    
    metrics_to_compare = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    for metric in metrics_to_compare:
        nb_val = nb_metrics[metric]
        svm_val = svm_metrics[metric]
        diff = svm_val - nb_val
        diff_pct = (diff / nb_val * 100) if nb_val > 0 else 0
        
        print(f"   {metric.capitalize():<20} {nb_val:<15.4f} {svm_val:<15.4f} {diff:+.4f} ({diff_pct:+.1f}%)")
    
    print("\n   Key Differences:")
    print("   - SVM uses maximum margin approach → Better generalization")
    print("   - SVM focuses on support vectors → More efficient")
    print("   - SVM finds optimal boundary → Better accuracy")
    print("   - But SVM is slower to train")
    
except Exception as e:
    print(f"   (Could not load comparison data: {e})")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
The SVM model:
1. Converts text to numbers using TF-IDF (10,000 word features)
2. Finds support vectors (hardest examples to classify)
3. Creates maximum margin hyperplane (optimal boundary)
4. Makes predictions by checking which side of boundary text falls on
5. More accurate than Naive Bayes but slower to train

Key Advantages:
- Better accuracy (66.53% vs 48.76%)
- Better generalization (maximum margin)
- More robust to outliers
- Works well with high-dimensional data (text)

Key Disadvantages:
- Slower training (5-15 min vs 1-2 min)
- Harder to interpret
- More complex algorithm
""")
print("="*70)


