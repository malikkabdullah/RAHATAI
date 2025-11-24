"""
Data preprocessing utilities for multilingual text

Cleans text for English, Urdu, and Roman-Urdu:
- Removes URLs, mentions, extra spaces
- Converts to lowercase
- Encodes labels for classification
"""
import re
import pandas as pd
import numpy as np
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class MultilingualPreprocessor:
    """
    Preprocessor for multilingual text (English, Urdu, Roman-Urdu)
    """
    
    def __init__(self, remove_stopwords: bool = True, 
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = False,
                 lowercase: bool = True):
        self.remove_stopwords = remove_stopwords
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase
        
        # English stopwords
        try:
            self.english_stopwords = set(stopwords.words('english'))
        except:
            self.english_stopwords = set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Input text string
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
            text = re.sub(r'httpAddress', '', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+|usrId', '', text)
        
        # Remove hashtags (optional - sometimes useful for classification)
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            # Keep hashtags but remove the # symbol
            text = re.sub(r'#', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List[str]: List of preprocessed texts
        """
        return [self.clean_text(text) for text in texts]
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Preprocess text column in a dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of the text column
            
        Returns:
            pd.DataFrame: Dataframe with preprocessed text
        """
        df = df.copy()
        df[f'{text_column}_cleaned'] = df[text_column].apply(self.clean_text)
        return df

def encode_labels(labels: List[str]) -> tuple:
    """
    Encode string labels to integers
    
    Args:
        labels: List of label strings
        
    Returns:
        tuple: (encoded_labels, label_to_idx, idx_to_label)
    """
    unique_labels = sorted(list(set(labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    encoded = [label_to_idx[label] for label in labels]
    
    return np.array(encoded), label_to_idx, idx_to_label

def prepare_classification_data(df: pd.DataFrame, 
                                text_column: str,
                                label_column: str,
                                preprocessor: Optional[MultilingualPreprocessor] = None) -> dict:
    """
    Prepare data for classification
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        label_column: Name of label column
        preprocessor: Optional preprocessor instance
        
    Returns:
        dict: Dictionary with 'texts', 'labels', 'encoded_labels', 'label_mappings'
    """
    if preprocessor is None:
        preprocessor = MultilingualPreprocessor()
    
    # Preprocess texts
    texts = preprocessor.preprocess(df[text_column].tolist())
    
    # Encode labels
    labels = df[label_column].tolist()
    encoded_labels, label_to_idx, idx_to_label = encode_labels(labels)
    
    return {
        'texts': texts,
        'labels': labels,
        'encoded_labels': encoded_labels,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'num_classes': len(label_to_idx)
    }

