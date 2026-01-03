# %% [markdown]
# # Task 2: Chunking, Embedding & Vector Indexing
# 
# This script performs:
# 1. Stratified sampling of complaints (10,000-15,000)
# 2. Text chunking with overlap
# 3. Embedding generation using sentence-transformers
# 4. Vector store creation using FAISS
# 5. Persistence to disk

# %% [code]
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import re
from typing import List, Dict
from tqdm import tqdm

# Configuration
INPUT_FILE = r'data/filtered_complaints.csv'
VECTOR_STORE_DIR = r'vector_store'
SAMPLE_SIZE = 12000
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50  # characters
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
RANDOM_SEED = 42

# %% [markdown]
# ## 1. Load Filtered Data

# %% [code]
print("Loading filtered complaints...")
df = pd.read_csv(INPUT_FILE)
print(f"Total filtered complaints: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(2))

# %% [markdown]
# ## 2. Stratified Sampling
# 
# **Sampling Logic:**
# - We maintain the product distribution from the filtered dataset
# - Using stratified sampling ensures each product category is proportionally represented
# - This prevents bias towards more common products while ensuring rare products are included
# - Sample size: 12,000 complaints (middle of 10k-15k range)

# %% [code]
print("\n" + "="*60)
print("STRATIFIED SAMPLING")
print("="*60)

# Check product distribution
print("\nOriginal Product Distribution:")
product_dist = df['Product'].value_counts()
print(product_dist)
print(f"\nTotal: {len(df)}")

# Stratified sampling
# Calculate fraction to sample
sample_fraction = min(SAMPLE_SIZE / len(df), 1.0)

print(f"\nSample fraction: {sample_fraction:.4f}")
print(f"Target sample size: {SAMPLE_SIZE}")

# Perform stratified sampling
df_sample = df.groupby('Product', group_keys=False).apply(
    lambda x: x.sample(frac=sample_fraction, random_state=RANDOM_SEED)
)

print(f"\nActual sample size: {len(df_sample)}")
print("\nSampled Product Distribution:")
print(df_sample['Product'].value_counts())

# %% [markdown]
# ## 3. Text Chunking Function
# 
# **Chunking Strategy:**
# - **Chunk Size**: 500 characters - optimal for sentence-transformers which have ~512 token limit
# - **Overlap**: 50 characters - ensures context continuity across chunks
# - **Method**: Character-based sliding window for consistent chunk sizes

# %% [code]
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    if not isinstance(text, str) or len(text) == 0:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk.strip())
        
        # Move start position
        start += chunk_size - overlap
        
        # Prevent infinite loop for very short texts
        if start <= 0:
            break
    
    return chunks

# Test chunking
sample_text = "This is a sample complaint narrative. " * 50
test_chunks = chunk_text(sample_text)
print(f"\nChunking Test:")
print(f"Original length: {len(sample_text)} chars")
print(f"Number of chunks: {len(test_chunks)}")
print(f"First chunk length: {len(test_chunks[0]) if test_chunks else 0}")
print(f"Last chunk length: {len(test_chunks[-1]) if test_chunks else 0}")

# %% [markdown]
# ## 4. Load Embedding Model
# 
# **Model Choice: sentence-transformers/all-MiniLM-L6-v2**
# 
# **Rationale:**
# - **Lightweight**: Only 80MB, fast inference
# - **Semantic Performance**: Strong performance on semantic similarity tasks
# - **Speed**: Can process thousands of texts quickly
# - **Proven**: Widely used in production RAG systems
# - **Balanced**: Good trade-off between accuracy and computational cost

# %% [code]
print("\n" + "="*60)
print("LOADING EMBEDDING MODEL")
print("="*60)
print(f"\nModel: {EMBEDDING_MODEL}")

model = SentenceTransformer(EMBEDDING_MODEL)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Embedding dimension: {embedding_dim}")

# %% [markdown]
# ## 5. Generate Chunks and Metadata

# %% [code]
print("\n" + "="*60)
print("CHUNKING NARRATIVES")
print("="*60)

chunks_data = []

# Use 'cleaned_narrative' if available, otherwise use original
narrative_col = 'cleaned_narrative' if 'cleaned_narrative' in df_sample.columns else 'Consumer complaint narrative'

for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Chunking"):
    narrative = row[narrative_col]
    
    # Skip if no narrative
    if pd.isna(narrative) or narrative == "":
        continue
    
    # Chunk the narrative
    chunks = chunk_text(str(narrative))
    
    # Store each chunk with metadata
    for chunk_idx, chunk in enumerate(chunks):
        chunks_data.append({
            'chunk_text': chunk,
            'complaint_id': row.get('Complaint ID', idx),
            'product': row.get('Product', 'Unknown'),
            'issue': row.get('Issue', 'Unknown'),
            'date_received': row.get('Date received', 'Unknown'),
            'chunk_index': chunk_idx,
            'total_chunks': len(chunks)
        })

print(f"\nTotal chunks created: {len(chunks_data)}")
print(f"Average chunks per complaint: {len(chunks_data) / len(df_sample):.2f}")

# Convert to DataFrame for easier handling
chunks_df = pd.DataFrame(chunks_data)
print("\nChunk DataFrame Info:")
print(chunks_df.info())
print("\nSample chunks:")
print(chunks_df.head(3))

# %% [markdown]
# ## 6. Generate Embeddings

# %% [code]
print("\n" + "="*60)
print("GENERATING EMBEDDINGS")
print("="*60)

# Extract texts
texts = chunks_df['chunk_text'].tolist()

# Generate embeddings in batches
batch_size = 32
embeddings_list = []

print(f"Processing {len(texts)} chunks in batches of {batch_size}...")

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    embeddings_list.append(batch_embeddings)

# Combine all embeddings
embeddings = np.vstack(embeddings_list)
print(f"\nEmbeddings shape: {embeddings.shape}")

# %% [markdown]
# ## 7. Create FAISS Index

# %% [code]
print("\n" + "="*60)
print("CREATING FAISS VECTOR STORE")
print("="*60)

# Create FAISS index
# Using IndexFlatL2 for exact search (suitable for our dataset size)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to index
index.add(embeddings.astype('float32'))

print(f"FAISS Index created")
print(f"Total vectors: {index.ntotal}")
print(f"Dimension: {dimension}")

# %% [markdown]
# ## 8. Persist Vector Store

# %% [code]
print("\n" + "="*60)
print("PERSISTING VECTOR STORE")
print("="*60)

# Create directory if it doesn't exist
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Save FAISS index
index_path = os.path.join(VECTOR_STORE_DIR, 'faiss_index.bin')
faiss.write_index(index, index_path)
print(f"FAISS index saved to: {index_path}")

# Save metadata
metadata_path = os.path.join(VECTOR_STORE_DIR, 'metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(chunks_df.to_dict('records'), f)
print(f"Metadata saved to: {metadata_path}")

# Save configuration
config = {
    'sample_size': len(df_sample),
    'total_chunks': len(chunks_data),
    'chunk_size': CHUNK_SIZE,
    'chunk_overlap': CHUNK_OVERLAP,
    'embedding_model': EMBEDDING_MODEL,
    'embedding_dim': embedding_dim,
    'index_type': 'IndexFlatL2'
}

config_path = os.path.join(VECTOR_STORE_DIR, 'config.pkl')
with open(config_path, 'wb') as f:
    pickle.dump(config, f)
print(f"Configuration saved to: {config_path}")

# %% [markdown]
# ## 9. Summary Report

# %% [code]
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

report = f"""
TASK 2 COMPLETION SUMMARY
========================

1. SAMPLING
   - Original dataset: {len(df):,} complaints
   - Sample size: {len(df_sample):,} complaints
   - Sampling method: Stratified by product

2. CHUNKING
   - Total chunks: {len(chunks_data):,}
   - Chunk size: {CHUNK_SIZE} characters
   - Chunk overlap: {CHUNK_OVERLAP} characters
   - Avg chunks/complaint: {len(chunks_data) / len(df_sample):.2f}

3. EMBEDDING
   - Model: {EMBEDDING_MODEL}
   - Embedding dimension: {embedding_dim}
   - Total embeddings: {embeddings.shape[0]:,}

4. VECTOR STORE
   - Type: FAISS IndexFlatL2
   - Location: {VECTOR_STORE_DIR}
   - Total vectors: {index.ntotal:,}

FILES CREATED:
   - {index_path}
   - {metadata_path}
   - {config_path}

READY FOR RAG QUERYING!
"""

print(report)

# Save report to file
report_path = os.path.join(VECTOR_STORE_DIR, 'build_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"\nReport saved to: {report_path}")

print("\n✓ Task 2 Complete!")

# %% [code]
if __name__ == "__main__":
    print("Script completed successfully!")
