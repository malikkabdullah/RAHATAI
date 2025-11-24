"""
Visual Explanation of Naive Bayes Model
Shows what the model actually learned and how it makes predictions
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*70)
print("NAIVE BAYES MODEL - VISUAL EXPLANATION")
print("="*70)

# Load the saved model
print("\n1. Loading saved model...")
with open('Models/naive_bayes.pkl', 'rb') as f:
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

# Show some example words
print(f"\n   Example words in vocabulary:")
vocab_items = list(vectorizer.vocabulary_.items())[:10]
for word, idx in vocab_items:
    print(f"      '{word}' → index {idx}")

# ============================================
# EXPLAIN NAIVE BAYES CLASSIFIER
# ============================================
print("\n" + "="*70)
print("3. NAIVE BAYES CLASSIFIER")
print("="*70)

nb_classifier = model.named_steps['nb']
print(f"\n   Classifier type: {type(nb_classifier).__name__}")
print(f"   Number of classes: {nb_classifier.class_count_.shape[0]}")
print(f"   Total training samples: {nb_classifier.class_count_.sum():.0f}")

print(f"\n   Class priors (how common each class is):")
for idx, count in enumerate(nb_classifier.class_count_):
    prior = count / nb_classifier.class_count_.sum()
    print(f"      {idx_to_label[idx]:30s}: {prior:.2%} ({count:.0f} samples)")

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
    
    print(f"\n   Text: '{text[:50]}...'")
    print(f"   → Predicted: {idx_to_label[prediction]}")
    print(f"   → Confidence: {probabilities[prediction]:.2%}")
    print(f"   → Top 3 classes:")
    for idx in top3_indices:
        print(f"        {idx_to_label[idx]:30s}: {probabilities[idx]:.2%}")

# ============================================
# SHOW WHAT THE MODEL LEARNED
# ============================================
print("\n" + "="*70)
print("5. WHAT THE MODEL LEARNED (Word Importance)")
print("="*70)

# Get feature log probabilities (what words are important for each class)
feature_log_probs = nb_classifier.feature_log_prob_

# For each class, find most important words
print("\n   Most important words for each class:")
for class_idx in range(len(idx_to_label)):
    class_name = idx_to_label[class_idx]
    
    # Get word importances for this class
    word_importances = feature_log_probs[class_idx]
    
    # Get top 10 words
    top_word_indices = np.argsort(word_importances)[-10:][::-1]
    
    # Get word names from vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\n   {class_name}:")
    for word_idx in top_word_indices[:5]:  # Show top 5
        word = feature_names[word_idx]
        importance = word_importances[word_idx]
        print(f"      '{word}': {importance:.2f}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
The Naive Bayes model:
1. Converts text to numbers using TF-IDF (10,000 word features)
2. Learns word probabilities for each of 6 classes
3. Makes predictions by calculating class probabilities
4. Fast and simple, but effective for text classification

The model learned patterns like:
- "injured", "people", "help" → Affected individuals
- "bridge", "road", "damage" → Infrastructure
- "donation", "volunteer" → Donations and volunteering
""")
print("="*70)


