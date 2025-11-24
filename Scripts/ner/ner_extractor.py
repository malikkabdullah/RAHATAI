"""
Multilingual Named Entity Recognition (NER)
Extracts: locations, phone numbers, resources, persons, organizations
Supports: English, Urdu, Roman-Urdu
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict
import pandas as pd

from Scripts.utils import load_config

class MultilingualNER:
    """
    Multilingual NER for crisis text
    """
    
    def __init__(self, model_name: str = "xlm-roberta-base", device: str = None):
        """
        Initialize NER model
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == 'cuda' else -1
        
        # Initialize NER pipeline
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
                device=self.device
            )
        except:
            # Fallback to a different model
            print("Warning: Using default NER model")
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=self.device
            )
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            list: List of entities with 'word', 'score', 'entity_group'
        """
        if not text or pd.isna(text):
            return []
        
        try:
            entities = self.ner_pipeline(text)
            return entities
        except Exception as e:
            print(f"Error in NER: {e}")
            return []
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """
        Extract phone numbers using regex
        
        Args:
            text: Input text
            
        Returns:
            list: List of phone numbers
        """
        # Pakistani phone number patterns
        patterns = [
            r'\+92\s?\d{2}\s?\d{7}',  # +92 XX XXXXXXX
            r'0\d{2}[\s-]?\d{7}',      # 0XX-XXXXXXX
            r'\d{4}[\s-]?\d{7}',       # XXXX-XXXXXXX
            r'\d{11}',                 # 11 digits
        ]
        
        phone_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            phone_numbers.extend(matches)
        
        # Clean and deduplicate
        phone_numbers = list(set([re.sub(r'[\s-]', '', p) for p in phone_numbers]))
        return phone_numbers
    
    def extract_locations(self, text: str) -> List[str]:
        """
        Extract location entities
        
        Args:
            text: Input text
            
        Returns:
            list: List of locations
        """
        entities = self.extract_entities(text)
        locations = [
            e['word'] for e in entities 
            if e.get('entity_group', '').upper() in ['LOC', 'LOCATION', 'GPE']
        ]
        return locations
    
    def extract_all(self, text: str) -> Dict:
        """
        Extract all entity types
        
        Args:
            text: Input text
            
        Returns:
            dict: Dictionary with all extracted entities
        """
        entities = self.extract_entities(text)
        
        result = {
            'locations': [],
            'persons': [],
            'organizations': [],
            'phone_numbers': [],
            'resources': [],
            'all_entities': entities
        }
        
        # Categorize entities
        for entity in entities:
            entity_group = entity.get('entity_group', '').upper()
            word = entity.get('word', '')
            
            if entity_group in ['LOC', 'LOCATION', 'GPE']:
                result['locations'].append(word)
            elif entity_group in ['PER', 'PERSON']:
                result['persons'].append(word)
            elif entity_group in ['ORG', 'ORGANIZATION']:
                result['organizations'].append(word)
        
        # Extract phone numbers
        result['phone_numbers'] = self.extract_phone_numbers(text)
        
        # Extract resources (keywords)
        resource_keywords = ['food', 'water', 'shelter', 'medical', 'medicine', 'aid', 'help']
        text_lower = text.lower()
        result['resources'] = [kw for kw in resource_keywords if kw in text_lower]
        
        return result

def process_dataset_ner(df: pd.DataFrame, text_column: str, 
                       ner_model: MultilingualNER) -> pd.DataFrame:
    """
    Process entire dataset with NER
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        ner_model: NER model instance
        
    Returns:
        pd.DataFrame: Dataframe with extracted entities
    """
    results = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        entities = ner_model.extract_all(text)
        
        results.append({
            'text': text,
            'locations': ', '.join(entities['locations']),
            'phone_numbers': ', '.join(entities['phone_numbers']),
            'persons': ', '.join(entities['persons']),
            'organizations': ', '.join(entities['organizations']),
            'resources': ', '.join(entities['resources']),
            'all_entities': str(entities['all_entities'])
        })
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} texts...")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    ner = MultilingualNER()
    
    sample_text = "Emergency in Lahore. Contact 03001234567. Need food and water. Dr. Ahmed is helping."
    entities = ner.extract_all(sample_text)
    
    print("Extracted Entities:")
    for key, value in entities.items():
        if value:
            print(f"  {key}: {value}")


