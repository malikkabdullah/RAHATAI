"""
Configuration loader utility
"""
import yaml
import os
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_data_paths(config):
    """Extract data paths from config"""
    return config.get('data', {})

def get_model_paths(config):
    """Extract model paths from config"""
    return config.get('models', {})

def get_output_paths(config):
    """Extract output paths from config"""
    return config.get('outputs', {})


