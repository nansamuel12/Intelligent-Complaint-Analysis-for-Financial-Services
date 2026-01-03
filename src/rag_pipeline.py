import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ComplaintRAG:
    def __init__(self, vector_store_dir: str = 'vector_store', model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the RAG pipeline.
        """
        self.vector_store_dir = vector_store_dir
        self.embedding_model_name = model_name
        
        self.index = None
        self.metadata = None
        self.config = None
        self.embedding_model = None
        self.llm = None
        self.chain = None
        
        # Initialize LLM (Ensure OPENAI_API_KEY is set in .env)
        # Using a temperature of 0 for consistent, factual answers
        try:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        except Exception as e:
            print(f"Warning: LLM could not be initialized. Check API keys. {e}")
            self.llm = None

    def load_resources(self):
        """Load vector store and embedding model."""
        print("Loading RAG resources...")
        
        # Paths
        index_path = os.path.join(self.vector_store_dir, 'faiss_index.bin')
        metadata_path = os.path.join(self.vector_store_dir, 'metadata.pkl')
        config_path = os.path.join(self.vector_store_dir, 'config.pkl')
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Vector store not found at {index_path}. Did you run Task 2?")
            
        # Load FAISS
        self.index = faiss.read_index(index_path)
        
        # Load Metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        # Load Config
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
                
        # Load Embedding Model
        # (Should match the one used in Task 2)
        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        print("✓ Resources loaded successfully.")
        
        # Setup Chain
        self._setup_chain()

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k similar chunks for a query.
        """
        if not self.index or not self.embedding_model:
            raise ValueError("Resources not loaded. Call load_resources() first.")
            
        # Generate embedding
        query_vector = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item['distance'] = float(distances[0][i])
                results.append(item)
                
        return results

    def _setup_chain(self):
        """Set up the LangChain Loop."""
        if not self.llm:
            return

        # Prompt Template
        # "Instruct the LLM to use only provided context. Say 'I don’t know' if context is insufficient."
        template = """You are an intelligent assistant analyzing consumer financial complaints.
        
        Use the following pieces of retrieved context to answer the question at the end.
        If the context does not contain enough information to answer the question, just say "I don't know" or "The provided context is insufficient to answer this question". Do not try to make up an answer.
        Keey your answer concise and professional.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        self.chain = prompt | self.llm | StrOutputParser()

    def format_docs(self, docs: List[Dict]) -> str:
        """Format retrieved chunks into a context string."""
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(f"Complaint {i+1} (Product: {doc.get('product')}, Issue: {doc.get('issue')}):\n{doc.get('chunk_text')}")
        return "\n\n".join(formatted)

    def answer_question(self, question: str, k: int = 5):
        """
        End-to-end RAG: Retrieve + Generate.
        Returns dictionary with answer and sources.
        """
        if not self.chain:
            return {"answer": "LLM not initialized (missing API key?)", "sources": []}
            
        # 1. Retrieve
        retrieved_docs = self.retrieve(question, k=k)
        
        # 2. Format Context
        context_str = self.format_docs(retrieved_docs)
        
        # 3. Generate
        response = self.chain.invoke({
            "context": context_str,
            "question": question
        })
        
        return {
            "question": question,
            "answer": response,
            "context_used": context_str,
            "sources": retrieved_docs
        }

if __name__ == "__main__":
    # Simple test
    rag = ComplaintRAG()
    try:
        rag.load_resources()
        print("\nTest Question: 'What are common issues with credit cards?'")
        result = rag.answer_question("What are common issues with credit cards?")
        print("\nAnswer:", result['answer'])
    except Exception as e:
        print(f"Skipping run: {e}")
