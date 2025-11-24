"""
Machine Learning Models for Classification

Implements 2 ML models (required):
- Naive Bayes: Uses TF-IDF + MultinomialNB
- SVM: Uses TF-IDF + Support Vector Machine

Both models use sklearn pipeline for easy training and prediction.
"""
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

class NaiveBayesClassifier:
    """
    Naive Bayes Classifier with TF-IDF vectorization
    """
    
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ('nb', MultinomialNB())
        ])
        self.is_trained = False
    
    def train(self, X_train: List[str], y_train: np.ndarray):
        """
        Train the Naive Bayes model
        
        Args:
            X_train: Training texts
            y_train: Training labels
        """
        print("Training Naive Bayes Classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Naive Bayes training completed")
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict labels"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {path}")

class SVMClassifier:
    """
    Support Vector Machine Classifier with TF-IDF vectorization
    """
    
    def __init__(self, max_features: int = 10000, 
                 ngram_range: Tuple[int, int] = (1, 2),
                 C: float = 1.0,
                 kernel: str = 'linear'):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.kernel = kernel
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            ('svm', SVC(C=C, kernel=kernel, probability=True, random_state=42))
        ])
        self.is_trained = False
    
    def train(self, X_train: List[str], y_train: np.ndarray):
        """
        Train the SVM model
        
        Args:
            X_train: Training texts
            y_train: Training labels
        """
        print("Training SVM Classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("SVM training completed")
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Predict labels"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {path}")

