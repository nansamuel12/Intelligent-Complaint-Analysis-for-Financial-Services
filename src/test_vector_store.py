"""
Helper script to test the vector store and perform similarity search.
"""

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os

class VectorStoreLoader:
    """Load and query the FAISS vector store."""
    
    def __init__(self, vector_store_dir: str = 'vector_store'):
        """
        Initialize the vector store loader.
        
        Args:
            vector_store_dir: Directory containing the vector store files
        """
        self.vector_store_dir = vector_store_dir
        self.index = None
        self.metadata = None
        self.config = None
        self.model = None
        
    def load(self):
        """Load the vector store and metadata."""
        print("Loading vector store...")
        
        # Load FAISS index
        index_path = os.path.join(self.vector_store_dir, 'faiss_index.bin')
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded FAISS index: {self.index.ntotal} vectors")
        
        # Load metadata
        metadata_path = os.path.join(self.vector_store_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"✓ Loaded metadata: {len(self.metadata)} entries")
        
        # Load config
        config_path = os.path.join(self.vector_store_dir, 'config.pkl')
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)
        print(f"✓ Loaded config")
        
        # Load embedding model
        model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded")
        
        return self
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar complaints.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing results and metadata
        """
        if self.model is None:
            raise ValueError("Vector store not loaded. Call load() first.")
        
        # Encode query
        query_vector = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = {
                    'rank': i + 1,
                    'distance': float(distances[0][i]),
                    'similarity_score': 1 / (1 + float(distances[0][i])),  # Convert distance to similarity
                    **self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def print_results(self, results: List[Dict]):
        """Pretty print search results."""
        print("\n" + "="*80)
        print("SEARCH RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\nRank #{result['rank']} (Similarity: {result['similarity_score']:.4f})")
            print(f"Product: {result['product']}")
            print(f"Issue: {result['issue']}")
            print(f"Date: {result['date_received']}")
            print(f"Complaint ID: {result['complaint_id']}")
            print(f"Chunk {result['chunk_index'] + 1}/{result['total_chunks']}")
            print(f"\nText Preview:")
            print(f"{result['chunk_text'][:300]}...")
            print("-" * 80)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if not self.metadata or not self.config:
            raise ValueError("Vector store not loaded. Call load() first.")
        
        products = {}
        for item in self.metadata:
            prod = item.get('product', 'Unknown')
            products[prod] = products.get(prod, 0) + 1
        
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'total_metadata_entries': len(self.metadata),
            'embedding_dimension': self.config.get('embedding_dim'),
            'chunk_size': self.config.get('chunk_size'),
            'chunk_overlap': self.config.get('chunk_overlap'),
            'products': products,
            'sample_size': self.config.get('sample_size'),
            'total_chunks': self.config.get('total_chunks')
        }


def main():
    """Example usage."""
    print("="*80)
    print("VECTOR STORE TEST")
    print("="*80)
    
    # Load vector store
    loader = VectorStoreLoader()
    loader.load()
    
    # Get stats
    stats = loader.get_stats()
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total Vectors: {stats['total_vectors']:,}")
    print(f"Embedding Dimension: {stats['embedding_dimension']}")
    print(f"Sample Size: {stats['sample_size']:,} complaints")
    print(f"Total Chunks: {stats['total_chunks']:,}")
    print(f"Chunk Size: {stats['chunk_size']} chars")
    print(f"Chunk Overlap: {stats['chunk_overlap']} chars")
    
    print("\nProduct Distribution:")
    for prod, count in sorted(stats['products'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {prod}: {count:,} chunks")
    
    # Example searches
    test_queries = [
        "I was charged an unfair fee on my credit card",
        "They closed my account without warning",
        "I can't access my savings account online"
    ]
    
    print("\n" + "="*80)
    print("EXAMPLE SEARCHES")
    print("="*80)
    
    for query in test_queries:
        print(f"\n📍 Query: \"{query}\"")
        results = loader.search(query, top_k=3)
        loader.print_results(results)
        print("\n")


if __name__ == "__main__":
    main()
