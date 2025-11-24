"""
RAG Query System - Question Answering with Retrieval-Augmented Generation
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional
import pickle

from Scripts.utils import load_config

class RAGSystem:
    """
    RAG-based Question Answering System
    """
    
    def __init__(self, vectorstore_path: str = "Models/rag_vectorstore",
                 embedding_model: str = None,
                 llm_model: str = None,
                 use_rag: bool = True):
        """
        Initialize RAG system
        
        Args:
            vectorstore_path: Path to FAISS vector store
            embedding_model: Embedding model name
            llm_model: LLM model name (or None for baseline)
            use_rag: Whether to use RAG (False for baseline)
        """
        self.vectorstore_path = Path(vectorstore_path)
        self.use_rag = use_rag
        self.vectorstore = None
        self.qa_chain = None
        
        # Load config
        config = load_config()
        rag_config = config['rag']
        
        self.embedding_model = embedding_model or rag_config.get('embedding_model')
        self.llm_model = llm_model or rag_config.get('llm_model')
        self.top_k = rag_config.get('top_k', 5)
        
        if use_rag:
            self._load_vectorstore()
            self._setup_qa_chain()
        else:
            # Baseline: LLM without retrieval
            self._setup_baseline_chain()
    
    def _load_vectorstore(self):
        """Load FAISS vector store"""
        if not self.vectorstore_path.exists():
            raise FileNotFoundError(f"Vector store not found: {self.vectorstore_path}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectorstore = FAISS.load_local(str(self.vectorstore_path), embeddings)
        print(f"Loaded vector store from {self.vectorstore_path}")
    
    def _setup_qa_chain(self):
        """Set up RAG QA chain"""
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite the source document and page number when available.

Context:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Initialize LLM (using OpenAI API - user needs to set OPENAI_API_KEY)
        # For open source alternatives, use HuggingFacePipeline
        try:
            llm = OpenAI(temperature=0, model_name=self.llm_model)
        except:
            print("Warning: OpenAI API key not set. Using HuggingFace model...")
            from langchain import HuggingFacePipeline
            from transformers import pipeline
            
            # Use a smaller model for local inference
            pipe = pipeline(
                "text-generation",
                model="gpt2",  # Replace with better model if available
                max_new_tokens=200,
                temperature=0.7
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.top_k}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def _setup_baseline_chain(self):
        """Set up baseline LLM chain (without RAG)"""
        try:
            llm = OpenAI(temperature=0, model_name=self.llm_model)
        except:
            from langchain import HuggingFacePipeline
            from transformers import pipeline
            
            pipe = pipeline(
                "text-generation",
                model="gpt2",
                max_new_tokens=200,
                temperature=0.7
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        
        self.qa_chain = llm
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: Question string
            
        Returns:
            dict: Answer with sources
        """
        if self.use_rag:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", "N/A")
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        else:
            # Baseline: direct LLM response
            answer = self.qa_chain(question)
            return {
                "answer": answer,
                "sources": []  # No sources for baseline
            }

def evaluate_rag_qa(qa_pairs: List[Dict], rag_system: RAGSystem) -> Dict:
    """
    Evaluate RAG system on QA pairs
    
    Args:
        qa_pairs: List of dicts with 'question', 'answer', 'source' keys
        rag_system: RAGSystem instance
        
    Returns:
        dict: Evaluation results
    """
    results = []
    
    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair['question']
        expected_answer = qa_pair['answer']
        expected_source = qa_pair.get('source', '')
        
        # Get prediction
        prediction = rag_system.query(question)
        predicted_answer = prediction['answer']
        predicted_sources = prediction.get('sources', [])
        
        results.append({
            'question': question,
            'expected_answer': expected_answer,
            'predicted_answer': predicted_answer,
            'expected_source': expected_source,
            'predicted_sources': predicted_sources
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(qa_pairs)} questions...")
    
    return results

if __name__ == "__main__":
    # Example usage
    rag_system = RAGSystem(use_rag=True)
    
    question = "What are the emergency contact numbers for disaster response?"
    result = rag_system.query(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")


