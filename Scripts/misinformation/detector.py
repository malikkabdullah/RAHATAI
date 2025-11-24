"""
Misinformation Detection Module
Binary classification with linguistic + semantic cues
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict
import pandas as pd

from Scripts.utils import load_config
from Scripts.classification.transformer_models import TransformerClassifier

class MisinformationDetector:
    """
    Detector for misinformation in crisis/disaster posts
    """
    
    def __init__(self, model_name: str = "xlm-roberta-base", 
                 threshold: float = 0.5,
                 device: str = None):
        """
        Initialize misinformation detector
        
        Args:
            model_name: HuggingFace model name
            threshold: Classification threshold
            device: Device to use
        """
        self.model_name = model_name
        self.threshold = threshold
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize as binary classifier
        # In practice, you'd fine-tune this on misinformation dataset
        self.classifier = TransformerClassifier(
            model_name=model_name,
            num_labels=2,  # Binary: misinformation vs verified
            device=str(self.device)
        )
        
        # Linguistic features
        self.uncertainty_markers = [
            'maybe', 'perhaps', 'might', 'could', 'possibly', 'rumor', 'heard',
            'unconfirmed', 'allegedly', 'supposedly'
        ]
        
        self.credibility_markers = [
            'verified', 'confirmed', 'official', 'source', 'authority',
            'government', 'news', 'report'
        ]
    
    def extract_linguistic_features(self, text: str) -> Dict:
        """
        Extract linguistic features that indicate misinformation
        
        Args:
            text: Input text
            
        Returns:
            dict: Linguistic features
        """
        text_lower = text.lower()
        
        features = {
            'uncertainty_count': sum(1 for marker in self.uncertainty_markers if marker in text_lower),
            'credibility_count': sum(1 for marker in self.credibility_markers if marker in text_lower),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'url_count': text.count('http') + text.count('www'),
            'hashtag_count': text.count('#'),
            'mention_count': text.count('@')
        }
        
        return features
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict misinformation (1) or verified (0)
        
        Args:
            texts: List of texts
            
        Returns:
            np.ndarray: Binary predictions (1 = misinformation, 0 = verified)
        """
        if not self.classifier.is_trained:
            # Use linguistic features as fallback
            predictions = []
            for text in texts:
                features = self.extract_linguistic_features(text)
                # Simple heuristic: high uncertainty, low credibility = misinformation
                score = (features['uncertainty_count'] * 0.3) - (features['credibility_count'] * 0.2)
                prediction = 1 if score > self.threshold else 0
                predictions.append(prediction)
            return np.array(predictions)
        
        # Use trained model
        probs = self.classifier.predict_proba(texts)
        predictions = (probs[:, 1] > self.threshold).astype(int)
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict misinformation probabilities
        
        Args:
            texts: List of texts
            
        Returns:
            np.ndarray: Probabilities [verified_prob, misinformation_prob]
        """
        if not self.classifier.is_trained:
            # Use linguistic features
            probs = []
            for text in texts:
                features = self.extract_linguistic_features(text)
                score = (features['uncertainty_count'] * 0.3) - (features['credibility_count'] * 0.2)
                # Normalize to [0, 1]
                misinformation_prob = min(max(score, 0), 1)
                verified_prob = 1 - misinformation_prob
                probs.append([verified_prob, misinformation_prob])
            return np.array(probs)
        
        return self.classifier.predict_proba(texts)
    
    def train(self, X_train: List[str], y_train: np.ndarray,
              X_val: List[str] = None, y_val: np.ndarray = None,
              epochs: int = 10):
        """
        Train the misinformation detector
        
        Args:
            X_train: Training texts
            y_train: Training labels (0 = verified, 1 = misinformation)
            X_val: Validation texts
            y_val: Validation labels
            epochs: Number of epochs
        """
        print("Training Misinformation Detector...")
        self.classifier.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs
        )
        print("Training completed")

if __name__ == "__main__":
    # Example usage
    detector = MisinformationDetector()
    
    sample_texts = [
        "Official report: Cyclone Pam hits Vanuatu. Verified by authorities.",
        "Maybe there's a cyclone? I heard from someone it might be bad.",
        "Rumor: Cyclone approaching. Unconfirmed reports."
    ]
    
    predictions = detector.predict(sample_texts)
    probs = detector.predict_proba(sample_texts)
    
    for text, pred, prob in zip(sample_texts, predictions, probs):
        label = "Misinformation" if pred == 1 else "Verified"
        print(f"{label} ({prob[1]:.2f}): {text[:50]}...")


