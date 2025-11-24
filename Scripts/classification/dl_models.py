"""
Deep Learning Models for Classification (Very Easy Version)
LSTM and CNN text classifiers using TensorFlow / Keras.
This file is rewritten in simple, beginner-friendly style.
"""

# ----------------------------------------------------
# BASIC IMPORTS
# ----------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# For converting text to numbers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# For saving/loading tokenizer
import pickle
from pathlib import Path

from typing import List, Dict, Optional


# =====================================================================
# ========================= LSTM CLASSIFIER ===========================
# =====================================================================

class LSTMClassifier:
    """
    Simple LSTM Text Classifier (Beginner-Friendly Version)
    """

    def __init__(
        self,
        max_features: int = 10000,
        max_length: int = 128,
        embedding_dim: int = 128,
        lstm_units: int = 128,
        num_classes: int = None,
        bidirectional: bool = True,
        spatial_dropout: float = 0.2
    ):
        # Save settings
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.spatial_dropout = spatial_dropout

        # Create tokenizer object
        self.tokenizer = Tokenizer(
            num_words=max_features,
            oov_token="<OOV>"
        )

        # Model will be created later
        self.model = None
        self.is_trained = False

    # ----------------------------------------------------
    # BUILD THE LSTM MODEL
    # ----------------------------------------------------
    def _build_model(self, vocab_size: int, num_classes: int):
        """
        Build the model layer-by-layer (explained simply)
        """

        # Input layer: takes sequences of integers
        input_layer = layers.Input(shape=(self.max_length,))

        # Step 1: Word embedding layer
        embed_layer = layers.Embedding(
            vocab_size,
            self.embedding_dim
        )(input_layer)

        # Step 2: Dropout for regularization
        drop_layer = layers.SpatialDropout1D(
            self.spatial_dropout
        )(embed_layer)

        # Step 3: LSTM layers
        if self.bidirectional is True:
            # First BiLSTM layer
            lstm_1 = layers.Bidirectional(
                layers.LSTM(
                    self.lstm_units,
                    dropout=0.2,
                    recurrent_dropout=0.2,
                    return_sequences=True
                )
            )(drop_layer)

            # Second BiLSTM layer
            lstm_2 = layers.Bidirectional(
                layers.LSTM(
                    self.lstm_units // 2,
                    dropout=0.2,
                    recurrent_dropout=0.2
                )
            )(lstm_1)

            lstm_out = lstm_2

        else:
            # Single-direction LSTM
            lstm_1 = layers.LSTM(
                self.lstm_units,
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=True
            )(drop_layer)

            lstm_2 = layers.LSTM(
                self.lstm_units // 2,
                dropout=0.2,
                recurrent_dropout=0.2
            )(lstm_1)

            lstm_out = lstm_2

        # Step 4: Batch Normalization
        norm_layer = layers.BatchNormalization()(lstm_out)

        # Step 5: Dense layer
        dense_layer = layers.Dense(
            128,
            activation='relu'
        )(norm_layer)

        # Step 6: Dropout again
        final_dropout = layers.Dropout(0.4)(dense_layer)

        # Step 7: Output layer
        output_layer = layers.Dense(
            num_classes,
            activation='softmax'
        )(final_dropout)

        # Build entire model
        model = keras.Model(
            inputs=input_layer,
            outputs=output_layer
        )

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # ----------------------------------------------------
    # TRAINING FUNCTION
    # ----------------------------------------------------
    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
        class_weight: Optional[Dict[int, float]] = None
    ):

        print("Training LSTM Classifier...")

        # Convert text → numbers
        self.tokenizer.fit_on_texts(X_train)

        # Build vocabulary size
        vocab_size = min(
            self.max_features,
            len(self.tokenizer.word_index) + 1
        )

        # Convert training text into sequences
        seq_train = self.tokenizer.texts_to_sequences(X_train)

        # Pad sequences to equal length
        padded_train = pad_sequences(
            seq_train,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )

        # Count number of classes
        num_classes = len(np.unique(y_train))
        self.num_classes = num_classes

        # Build model
        self.model = self._build_model(
            vocab_size,
            num_classes
        )

        # Prepare validation data (optional)
        validation_data = None

        if X_val is not None and y_val is not None:
            seq_val = self.tokenizer.texts_to_sequences(X_val)
            padded_val = pad_sequences(
                seq_val,
                maxlen=self.max_length,
                padding='post',
                truncating='post'
            )

            validation_data = (padded_val, y_val)

        # Train model
        history = self.model.fit(
            padded_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3
                )
            ]
        )

        self.is_trained = True
        print("LSTM training finished!")

        return history.history

    # ----------------------------------------------------
    # PREDICT LABEL CLASSES
    # ----------------------------------------------------
    def predict(self, X: List[str]) -> np.ndarray:
        if self.is_trained is False:
            raise ValueError("Model not trained yet!")

        seq = self.tokenizer.texts_to_sequences(X)
        padded = pad_sequences(seq, maxlen=self.max_length)
        probs = self.model.predict(padded, verbose=0)

        return np.argmax(probs, axis=1)

    # ----------------------------------------------------
    # PREDICT PROBABILITIES
    # ----------------------------------------------------
    def predict_proba(self, X: List[str]) -> np.ndarray:
        if self.is_trained is False:
            raise ValueError("Model not trained yet!")

        seq = self.tokenizer.texts_to_sequences(X)
        padded = pad_sequences(seq, maxlen=self.max_length)

        return self.model.predict(padded, verbose=0)

    # ----------------------------------------------------
    # SAVE MODEL + TOKENIZER
    # ----------------------------------------------------
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(path / "model.h5")

        # Save tokenizer
        with open(path / "tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

        print(f"Model saved at: {path}")

    # ----------------------------------------------------
    # LOAD MODEL + TOKENIZER
    # ----------------------------------------------------
    def load(self, path: str):
        path = Path(path)

        # Load model
        self.model = keras.models.load_model(path / "model.h5")

        # Load tokenizer
        with open(path / "tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        self.is_trained = True
        print(f"Model loaded from: {path}")


# =====================================================================
# ========================= CNN CLASSIFIER =============================
# =====================================================================

class CNNClassifier:
    """
    Simple CNN Text Classifier (Easy Version)
    """

    def __init__(
        self,
        max_features: int = 10000,
        max_length: int = 128,
        embedding_dim: int = 128,
        num_filters: int = 128,
        filter_sizes: List[int] = [3, 4, 5],
        num_classes: int = None
    ):
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes

        # Tokenizer for text → numbers
        self.tokenizer = Tokenizer(
            num_words=max_features,
            oov_token="<OOV>"
        )

        self.model = None
        self.is_trained = False

    # ----------------------------------------------------
    # BUILD THE CNN MODEL
    # ----------------------------------------------------
    def _build_model(self, vocab_size: int, num_classes: int):
        # Input layer
        input_layer = layers.Input(shape=(self.max_length,))

        # Embedding layer
        embed_layer = layers.Embedding(
            vocab_size,
            self.embedding_dim,
            input_length=self.max_length
        )(input_layer)

        # Apply multiple convolution filters
        conv_outputs = []

        for size in self.filter_sizes:
            conv_layer = layers.Conv1D(
                self.num_filters,
                size,
                activation='relu'
            )(embed_layer)

            pooled = layers.GlobalMaxPooling1D()(conv_layer)
            conv_outputs.append(pooled)

        # Combine all filters
        merged = layers.Concatenate()(conv_outputs)

        # Dropout
        drop_1 = layers.Dropout(0.5)(merged)

        # Dense layer
        dense_1 = layers.Dense(64, activation='relu')(drop_1)

        # Another Dropout
        drop_2 = layers.Dropout(0.3)(dense_1)

        # Output layer
        output_layer = layers.Dense(
            num_classes,
            activation='softmax'
        )(drop_2)

        # Model creation
        model = keras.Model(
            inputs=input_layer,
            outputs=output_layer
        )

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # ----------------------------------------------------
    # TRAIN CNN
    # ----------------------------------------------------
    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ):

        print("Training CNN Classifier...")

        # Tokenize
        self.tokenizer.fit_on_texts(X_train)

        vocab_size = len(self.tokenizer.word_index) + 1

        # Convert sentences to numbers
        seq_train = self.tokenizer.texts_to_sequences(X_train)

        # Pad sequences
        padded_train = pad_sequences(
            seq_train,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )

        # Number of unique labels
        num_classes = len(np.unique(y_train))
        self.num_classes = num_classes

        # Build CNN model
        self.model = self._build_model(
            vocab_size,
            num_classes
        )

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            seq_val = self.tokenizer.texts_to_sequences(X_val)
            padded_val = pad_sequences(seq_val, maxlen=self.max_length)
            validation_data = (padded_val, y_val)

        # Train
        history = self.model.fit(
            padded_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3
                )
            ]
        )

        self.is_trained = True
        print("CNN training finished!")
        return history.history

    # ----------------------------------------------------
    # PREDICT LABELS
    # ----------------------------------------------------
    def predict(self, X: List[str]):
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        seq = self.tokenizer.texts_to_sequences(X)
        padded = pad_sequences(seq, maxlen=self.max_length)
        predictions = self.model.predict(padded, verbose=0)

        return np.argmax(predictions, axis=1)

    # ----------------------------------------------------
    # PREDICT PROBABILITIES
    # ----------------------------------------------------
    def predict_proba(self, X: List[str]):
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        seq = self.tokenizer.texts_to_sequences(X)
        padded = pad_sequences(seq, maxlen=self.max_length)

        return self.model.predict(padded, verbose=0)

    # ----------------------------------------------------
    # SAVE MODEL
    # ----------------------------------------------------
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path / "model.h5")

        with open(path / "tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

        print(f"CNN model saved at: {path}")

    # ----------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------
    def load(self, path: str):
        path = Path(path)

        self.model = keras.models.load_model(path / "model.h5")

        with open(path / "tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        self.is_trained = True
        print(f"CNN model loaded from: {path}")
