"""
Text Summarization Module
Cluster-level abstractive summaries of disaster reports
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict
import pandas as pd

from Scripts.utils import load_config

class CrisisSummarizer:
    """
    Summarizer for crisis/disaster reports
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        """
        Initialize summarizer
        
        Args:
            model_name: HuggingFace model name
            device: Device (-1 for CPU, 0+ for GPU)
        """
        self.model_name = model_name
        self.device = device
        
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=device
            )
        except:
            print("Warning: Using default summarization model")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Summarize a single text
        
        Args:
            text: Input text
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            str: Summary
        """
        if not text or pd.isna(text) or len(text.strip()) < 10:
            return ""
        
        try:
            # Truncate if too long (models have token limits)
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            result = self.summarizer(text, max_length=max_length, min_length=min_length)
            return result[0]['summary_text']
        except Exception as e:
            print(f"Error in summarization: {e}")
            return text[:max_length]  # Fallback: truncate
    
    def summarize_cluster(self, texts: List[str], max_length: int = 200) -> str:
        """
        Summarize a cluster of related texts
        
        Args:
            texts: List of texts in the cluster
            max_length: Maximum summary length
            
        Returns:
            str: Cluster summary
        """
        # Combine texts
        combined_text = " ".join(texts)
        
        # Summarize
        return self.summarize(combined_text, max_length=max_length, min_length=50)
    
    def summarize_by_region(self, df: pd.DataFrame, text_column: str, 
                            region_column: str = None) -> Dict[str, str]:
        """
        Summarize reports grouped by region
        
        Args:
            df: Dataframe with texts
            text_column: Name of text column
            region_column: Name of region column (if None, creates single summary)
            
        Returns:
            dict: Dictionary mapping region -> summary
        """
        summaries = {}
        
        if region_column and region_column in df.columns:
            # Group by region
            for region, group in df.groupby(region_column):
                texts = group[text_column].tolist()
                summary = self.summarize_cluster(texts)
                summaries[region] = summary
        else:
            # Single summary for all
            texts = df[text_column].tolist()
            summary = self.summarize_cluster(texts)
            summaries['all'] = summary
        
        return summaries

if __name__ == "__main__":
    # Example usage
    summarizer = CrisisSummarizer()
    
    sample_text = """
    Cyclone Pam has devastated Vanuatu with category 5 winds. Emergency response teams from Australia 
    are on their way. Thousands await aid as death toll increases. Infrastructure and utilities are 
    severely damaged. Regional disaster insurance scheme can help fund government rebuilding efforts.
    """
    
    summary = summarizer.summarize(sample_text)
    print("Original:", sample_text[:100] + "...")
    print("Summary:", summary)


