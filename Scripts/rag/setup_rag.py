"""
Setup RAG pipeline - Document ingestion and indexing

Loads PDFs/text documents, splits into chunks, creates embeddings,
and builds FAISS vector store for retrieval-augmented generation.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os

from Scripts.utils import load_config

def load_documents(doc_paths: list):
    """
    Load documents from various sources
    
    Args:
        doc_paths: List of document paths
        
    Returns:
        list: List of document objects
    """
    documents = []
    
    for doc_path in doc_paths:
        doc_path = Path(doc_path)
        if not doc_path.exists():
            print(f"Warning: Document not found: {doc_path}")
            continue
        
        try:
            if doc_path.suffix == '.pdf':
                loader = PyPDFLoader(str(doc_path))
                docs = loader.load()
                # Add metadata
                for doc in docs:
                    doc.metadata['source'] = doc_path.name
                    doc.metadata['page'] = doc.metadata.get('page', 0)
                documents.extend(docs)
                print(f"Loaded PDF: {doc_path.name} ({len(docs)} pages)")
            elif doc_path.suffix in ['.txt', '.md']:
                loader = TextLoader(str(doc_path))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = doc_path.name
                documents.extend(docs)
                print(f"Loaded text file: {doc_path.name}")
        except Exception as e:
            print(f"Error loading {doc_path}: {e}")
    
    return documents

def setup_rag_vectorstore(config_path: str = "config/config.yaml",
                          doc_paths: list = None,
                          output_dir: str = "Models/rag_vectorstore"):
    """
    Set up RAG vector store from documents
    
    Args:
        config_path: Path to config file
        doc_paths: List of document paths to ingest
        output_dir: Directory to save vector store
    """
    print("="*70)
    print("SETTING UP RAG VECTOR STORE")
    print("="*70)
    
    # Load config
    config = load_config(config_path)
    rag_config = config['rag']
    
    # Default document paths (user should provide their own)
    if doc_paths is None:
        print("\nWarning: No documents provided. Please add PDF/text files to Data/documents/")
        print("   Example document paths:")
        print("   - Data/documents/disaster_response_guide.pdf")
        print("   - Data/documents/emergency_contacts.txt")
        return
    
    # Load documents
    print("\n📂 Loading documents...")
    documents = load_documents(doc_paths)
    
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_config.get('chunk_size', 512),
        chunk_overlap=rag_config.get('chunk_overlap', 50)
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Create embeddings
    print("\nCreating embeddings...")
    embedding_model = rag_config.get('embedding_model', 
                                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'}  # Use 'cuda' if GPU available
    )
    
    # Create vector store
    print("\nCreating vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save vector store
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_dir))
    
    # Save metadata
    metadata = {
        'num_documents': len(documents),
        'num_chunks': len(chunks),
        'embedding_model': embedding_model,
        'chunk_size': rag_config.get('chunk_size', 512),
        'chunk_overlap': rag_config.get('chunk_overlap', 50),
        'documents': [doc.metadata.get('source', 'unknown') for doc in documents]
    }
    
    with open(output_dir / "metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\nVector store saved to {output_dir}")
    print(f"   - {len(chunks)} chunks indexed")
    print(f"   - Embedding model: {embedding_model}")
    print("="*70)

if __name__ == "__main__":
    # Example usage - user should modify these paths
    doc_paths = [
        # Add your document paths here
        # "Data/documents/disaster_response_guide.pdf",
        # "Data/documents/emergency_contacts.txt",
    ]
    
    setup_rag_vectorstore(doc_paths=doc_paths)

