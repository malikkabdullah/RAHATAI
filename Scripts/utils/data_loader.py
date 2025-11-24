"""
Data loading utilities for RahatAI
"""
import pandas as pd
import os
from pathlib import Path

def load_dataset(file_path, sep=None):
    """
    Automatically loads a CSV or TSV dataset into a Pandas DataFrame.
    
    Args:
        file_path: Path to the dataset file
        sep: Separator (None for auto-detection)
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect separator
    if sep is None:
        if file_path.suffix == '.csv':
            sep = ','
        elif file_path.suffix == '.tsv':
            sep = '\t'
        else:
            # Try to detect
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    sep = '\t'
                else:
                    sep = ','
    
    try:
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        # Try with different encoding
        df = pd.read_csv(file_path, sep=sep, encoding='latin-1', on_bad_lines='skip')
    
    print(f"Loaded {file_path.name} | Shape: {df.shape}")
    return df

def load_crisisnlp_datasets(data_dir="Data/CrisisNLP"):
    """
    Load all CrisisNLP datasets
    
    Returns:
        tuple: (train_df, dev_df, test_df)
    """
    base_path = Path(data_dir)
    
    train_df = load_dataset(base_path / "sample_prccd_train.csv")
    dev_df = load_dataset(base_path / "sample_prccd_dev.csv")
    test_df = load_dataset(base_path / "sample_prccd_test.csv")
    
    return train_df, dev_df, test_df

def load_humaid_datasets(data_dir="Data/HumAid"):
    """
    Load all HumAID datasets
    
    Returns:
        tuple: (train_df, dev_df, test_df)
    """
    base_path = Path(data_dir)
    
    train_df = load_dataset(base_path / "all_train.tsv")
    dev_df = load_dataset(base_path / "all_dev.tsv")
    test_df = load_dataset(base_path / "all_test.tsv")
    
    return train_df, dev_df, test_df

def combine_datasets(crisisnlp_dfs, humaid_dfs):
    """
    Combine CrisisNLP and HumAID datasets
    
    Args:
        crisisnlp_dfs: Tuple of (train, dev, test) from CrisisNLP
        humaid_dfs: Tuple of (train, dev, test) from HumAID
        
    Returns:
        tuple: Combined (train, dev, test) datasets
    """
    crisisnlp_train, crisisnlp_dev, crisisnlp_test = crisisnlp_dfs
    humaid_train, humaid_dev, humaid_test = humaid_dfs
    
    # Standardize column names
    # CrisisNLP: item_id, item, label
    # HumAID: tweet_id, class_label (need to get tweet text separately)
    
    # For now, return separate datasets
    # TODO: Implement proper merging logic based on available columns
    return {
        'crisisnlp': {
            'train': crisisnlp_train,
            'dev': crisisnlp_dev,
            'test': crisisnlp_test
        },
        'humaid': {
            'train': humaid_train,
            'dev': humaid_dev,
            'test': humaid_test
        }
    }


