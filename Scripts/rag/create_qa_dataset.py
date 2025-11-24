"""
Create 100 QA pairs dataset for RAG evaluation
Template and helper script
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict

def create_qa_dataset_template():
    """
    Create a template CSV file for QA pairs
    Each row should have: question, answer, source_document, page_number
    """
    template_data = {
        'question': [],
        'answer': [],
        'source_document': [],
        'page_number': [],
        'notes': []
    }
    
    # Create template with example structure
    template_df = pd.DataFrame(template_data)
    
    # Save template
    output_path = Path("Data/rag_qa_pairs_template.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(output_path, index=False)
    
    print(f"Template created at {output_path}")
    print("\nPlease fill in 100 QA pairs with the following columns:")
    print("  - question: The question to ask")
    print("  - answer: The expected answer")
    print("  - source_document: Name of the PDF/document")
    print("  - page_number: Page number(s) where answer is found (e.g., '5' or '10-12')")
    print("  - notes: Optional notes")
    
    return output_path

def validate_qa_dataset(qa_path: str = "Data/rag_qa_pairs.csv") -> bool:
    """
    Validate QA dataset has 100 pairs and required columns
    
    Args:
        qa_path: Path to QA dataset
        
    Returns:
        bool: True if valid
    """
    qa_path = Path(qa_path)
    
    if not qa_path.exists():
        print(f"QA dataset not found: {qa_path}")
        return False
    
    df = pd.read_csv(qa_path)
    
    # Check required columns
    required_cols = ['question', 'answer', 'source_document', 'page_number']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return False
    
    # Check number of pairs
    if len(df) < 100:
        print(f"Warning: Only {len(df)} QA pairs found. Need 100.")
        return False
    
    if len(df) > 100:
        print(f"Warning: {len(df)} QA pairs found. Using first 100.")
        df = df.head(100)
        df.to_csv(qa_path, index=False)
    
    # Check for empty values
    empty_questions = df['question'].isna().sum()
    empty_answers = df['answer'].isna().sum()
    
    if empty_questions > 0 or empty_answers > 0:
        print(f"Warning: Found {empty_questions} empty questions and {empty_answers} empty answers")
    
    print(f"QA dataset validated: {len(df)} pairs")
    return True

def format_qa_pairs_for_evaluation(qa_path: str = "Data/rag_qa_pairs.csv") -> List[Dict]:
    """
    Format QA pairs for evaluation
    
    Args:
        qa_path: Path to QA dataset
        
    Returns:
        list: Formatted QA pairs
    """
    df = pd.read_csv(qa_path)
    
    qa_pairs = []
    for _, row in df.iterrows():
        qa_pairs.append({
            'question': str(row['question']),
            'answer': str(row['answer']),
            'source': f"{row.get('source_document', 'unknown')} (Page {row.get('page_number', 'N/A')})"
        })
    
    return qa_pairs

if __name__ == "__main__":
    print("="*70)
    print("QA DATASET CREATION HELPER")
    print("="*70)
    
    # Create template
    template_path = create_qa_dataset_template()
    
    print("\n" + "="*70)
    print("INSTRUCTIONS:")
    print("="*70)
    print("""
1. Fill in the template CSV with 100 Question-Answer pairs
2. Each QA pair should:
   - Have a clear, specific question
   - Have an accurate answer based on your disaster response documents
   - Include the source document name
   - Include the page number(s) where the answer is found
   - Some questions may require information from multiple documents/pages

3. Example format:
   question,answer,source_document,page_number,notes
   "What is the emergency helpline number?","The emergency helpline is 112","disaster_response_guide.pdf","5",
   "Where can I find shelter during floods?","Shelters are located at...","emergency_contacts.txt","2-3","Multiple pages"

4. Once complete, save as: Data/rag_qa_pairs.csv

5. Run validation:
   python Scripts/rag/create_qa_dataset.py
    """)
    
    # Check if dataset already exists
    qa_path = Path("Data/rag_qa_pairs.csv")
    if qa_path.exists():
        print(f"\nFound existing dataset: {qa_path}")
        if validate_qa_dataset(str(qa_path)):
            print("Dataset is valid!")
        else:
            print("Warning: Dataset needs corrections")


