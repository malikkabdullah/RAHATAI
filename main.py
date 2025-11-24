"""
RahatAI - Main Entry Point
Multilingual Crisis Response NLP System

Usage:
    python main.py train              # Train all 5 classification models
    python main.py rag_setup          # Set up RAG vector store
    python main.py rag_eval           # Evaluate RAG vs baseline
    python main.py ner                # Run Named Entity Recognition
    python main.py summarize          # Generate summaries
    python main.py misinformation     # Detect misinformation
"""
import sys
from pathlib import Path
import argparse

# Add project root to path so we can import Scripts modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description="RahatAI - Crisis Response NLP System")
    parser.add_argument(
        'task',
        choices=['train', 'rag_setup', 'rag_eval', 'ner', 'summarize', 'misinformation'],
        help='Task to perform'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    if args.task == 'train':
        print("Starting Classification Model Training...")
        from Scripts.classification.train_classifiers import train_all_models
        train_all_models(args.config)
    
    elif args.task == 'rag_setup':
        print("Setting up RAG Vector Store...")
        from Scripts.rag.setup_rag import setup_rag_vectorstore
        print("Please provide document paths in the script or config file")
        setup_rag_vectorstore(config_path=args.config)
    
    elif args.task == 'rag_eval':
        print("Evaluating RAG System...")
        from Scripts.rag.evaluate_rag import compare_rag_vs_baseline
        import pandas as pd
        qa_path = Path("Data/rag_qa_pairs.csv")
        if qa_path.exists():
            qa_df = pd.read_csv(qa_path)
            qa_pairs = qa_df.to_dict('records')
            compare_rag_vs_baseline(qa_pairs, args.config)
        else:
            print(f"QA pairs file not found: {qa_path}")
            print("   Run: python Scripts/rag/create_qa_dataset.py")
    
    elif args.task == 'ner':
        print("Running NER Extraction...")
        from Scripts.ner.ner_extractor import MultilingualNER, process_dataset_ner
        from Scripts.utils import load_crisisnlp_datasets
        import pandas as pd
        
        ner = MultilingualNER()
        train_df, _, test_df = load_crisisnlp_datasets()
        
        print("Processing test set...")
        results = process_dataset_ner(test_df, 'item', ner)
        results.to_csv("Outputs/ner_results.csv", index=False)
        print("NER results saved to Outputs/ner_results.csv")
    
    elif args.task == 'summarize':
        print("Running Summarization...")
        from Scripts.summarization.summarizer import CrisisSummarizer
        from Scripts.utils import load_crisisnlp_datasets
        
        summarizer = CrisisSummarizer()
        train_df, _, _ = load_crisisnlp_datasets()
        
        # Summarize by label
        summaries = summarizer.summarize_by_region(train_df, 'item', 'label')
        
        for label, summary in summaries.items():
            print(f"\n{label}:")
            print(f"  {summary[:200]}...")
    
    elif args.task == 'misinformation':
        print("Running Misinformation Detection...")
        from Scripts.misinformation.detector import MisinformationDetector
        from Scripts.utils import load_crisisnlp_datasets
        import pandas as pd
        
        detector = MisinformationDetector()
        train_df, _, test_df = load_crisisnlp_datasets()
        
        # Example: detect misinformation in test set
        sample_texts = test_df['item'].head(10).tolist()
        predictions = detector.predict(sample_texts)
        
        results_df = pd.DataFrame({
            'text': sample_texts,
            'is_misinformation': predictions
        })
        results_df.to_csv("Outputs/misinformation_results.csv", index=False)
        print("Misinformation detection results saved to Outputs/misinformation_results.csv")

if __name__ == "__main__":
    main()

