"""
Transformer-based Models for Classification

Implements 1 Transformer model (required):
- XLM-RoBERTa: Multilingual transformer for text classification
- Supports English, Urdu, Roman-Urdu text
- Uses HuggingFace Transformers library
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from pathlib import Path

class TextClassificationDataset(Dataset):
    """Dataset class for text classification"""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerClassifier:
    """
    Transformer-based Text Classifier using XLM-RoBERTa
    """
    
    def __init__(self, model_name: str = "xlm-roberta-base",
                 num_labels: int = None,
                 max_length: int = 128,
                 device: str = None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Model will be initialized during training
        self.model = None
        self.is_trained = False
    
    def _initialize_model(self, num_labels: int):
        """Initialize the transformer model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        self.num_labels = num_labels
    
    def train(self, X_train: List[str], y_train: np.ndarray,
              X_val: Optional[List[str]] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 10,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              output_dir: str = "Models/checkpoints/transformer",
              save_steps: int = 500) -> Dict:
        """
        Train the transformer model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Directory to save checkpoints
            save_steps: Steps between checkpoints
            
        Returns:
            dict: Training history
        """
        print("Training Transformer Classifier...")
        
        num_labels = len(np.unique(y_train))
        if self.model is None:
            self._initialize_model(num_labels)
        
        # Create datasets
        train_dataset = TextClassificationDataset(X_train, y_train, self.tokenizer, self.max_length)
        
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_dataset = TextClassificationDataset(X_val, y_val, self.tokenizer, self.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_dataset else None,
        )
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        self.is_trained = True
        print("Transformer training completed")
        
        # Extract training history
        history = {
            'loss': [train_result.training_loss] if hasattr(train_result, 'training_loss') else [],
            'train_loss': trainer.state.log_history[-1].get('train_loss', []) if trainer.state.log_history else []
        }
        
        return history
    
    def predict(self, X: List[str], batch_size: int = 32) -> np.ndarray:
        """Predict labels"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_texts = X[i:i+batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def predict_proba(self, X: List[str], batch_size: int = 32) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_texts = X[i:i+batch_size]
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.extend(probs)
        
        return np.array(all_probs)
    
    def save(self, path: str):
        """Save model and tokenizer"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"Model saved to {path}")
        else:
            raise ValueError("Model not trained yet")
    
    def load(self, path: str):
        """Load model and tokenizer"""
        path = Path(path)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        self.num_labels = self.model.config.num_labels
        
        self.is_trained = True
        print(f"Model loaded from {path}")

