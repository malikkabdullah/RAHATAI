"""
Setup RAG with Documents
Edit this file to add your document paths, then run it
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Scripts.rag.setup_rag import setup_rag_vectorstore

print("=" * 70)
print("RAG SETUP - ADD YOUR DOCUMENT PATHS BELOW")
print("=" * 70)

# ============================================================
# STEP 1: Add your document paths here
# ============================================================
# Place your PDF/text files in Data/documents/ folder
# Then add their paths below:

doc_paths = [
    # Example paths (uncomment and modify):
    # "Data/documents/disaster_response_guide.pdf",
    # "Data/documents/emergency_contacts.txt",
    # "Data/documents/evacuation_procedures.pdf",
    
    # Add your document paths here:
    # "Data/documents/your_file1.pdf",
    # "Data/documents/your_file2.txt",
]

# ============================================================
# STEP 2: Check if documents folder exists
# ============================================================
documents_dir = Path("Data/documents")
if not documents_dir.exists():
    documents_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated documents folder: {documents_dir}")
    print("Please add your PDF/text files to this folder first!")

# ============================================================
# STEP 3: Check if any documents are specified
# ============================================================
if not doc_paths:
    print("\n" + "=" * 70)
    print("NO DOCUMENTS SPECIFIED")
    print("=" * 70)
    print("\nTo set up RAG:")
    print("1. Add PDF/text files to: Data/documents/")
    print("2. Edit this file and add document paths to 'doc_paths' list")
    print("3. Run this script again")
    print("\nExample document paths:")
    print("  - Data/documents/disaster_response_guide.pdf")
    print("  - Data/documents/emergency_contacts.txt")
    print("\nSee: Docs/RAG_DOCUMENTS_GUIDE.md for where to find documents")
    print("=" * 70)
    sys.exit(0)

# ============================================================
# STEP 4: Setup RAG vector store
# ============================================================
print(f"\nSetting up RAG with {len(doc_paths)} document(s)...")
setup_rag_vectorstore(doc_paths=doc_paths)

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Create 100 QA pairs: python Scripts/rag/create_qa_dataset.py")
print("2. Evaluate RAG: python main.py rag_eval")
print("=" * 70)

