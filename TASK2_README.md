# Task 2: Chunking, Embedding & Vector Indexing

## Overview
This task converts complaint narratives into embeddings and stores them in a vector database for efficient semantic search.

## Implementation

### 1. Stratified Sampling
**Script**: `src/create_embeddings.py`

**Logic**:
- Sample size: 12,000 complaints (middle of 10k-15k range)
- Method: Stratified sampling by product category
- Rationale: Maintains proportional representation of each product type, prevents bias toward common products while ensuring rare products are included

### 2. Text Chunking
**Parameters**:
- `chunk_size`: 500 characters
- `chunk_overlap`: 50 characters

**Strategy**:
- Character-based sliding window approach
- Ensures context continuity across chunk boundaries
- Optimized for sentence-transformers token limits (~512 tokens)

### 3. Embedding Model
**Model**: `sentence-transformers/all-MiniLM-L6-v2`

**Why This Model?**
- ✅ **Lightweight**: Only 80MB, minimal resource requirements
- ✅ **Fast**: Quick inference, can process thousands of texts efficiently
- ✅ **Semantic Performance**: Strong performance on semantic similarity benchmarks
- ✅ **Production-Ready**: Widely used in real-world RAG systems
- ✅ **Balanced**: Optimal trade-off between accuracy and computational cost
- ✅ **Embedding Dimension**: 384 dimensions (compact yet expressive)

### 4. Metadata Structure
Each chunk stores:
- `chunk_text`: The actual text chunk
- `complaint_id`: Unique complaint identifier
- `product`: Product category
- `issue`: Issue type
- `date_received`: Complaint receive date
- `chunk_index`: Position in the original narrative
- `total_chunks`: Total chunks from this complaint

### 5. Vector Store
**Technology**: FAISS (Facebook AI Similarity Search)
- **Index Type**: IndexFlatL2 (exact search)
- **Location**: `vector_store/`
- **Files**:
  - `faiss_index.bin`: Vector index
  - `metadata.pkl`: Chunk metadata
  - `config.pkl`: Configuration parameters
  - `build_report.txt`: Build summary

## Usage

### Build Vector Store
```bash
# Activate virtual environment
venv\Scripts\activate

# Run the embedding script
python src/create_embeddings.py
```

### Expected Output
```
vector_store/
├── faiss_index.bin      # FAISS vector index
├── metadata.pkl         # Chunk metadata
├── config.pkl          # Build configuration
└── build_report.txt    # Summary report
```

## Performance Metrics
- **Sample Size**: ~12,000 complaints
- **Total Chunks**: ~60,000-80,000 (depending on narrative lengths)
- **Average Chunks/Complaint**: ~5-7
- **Embedding Dimension**: 384
- **Processing Time**: ~10-15 minutes (depending on hardware)

## Next Steps
After building the vector store, you can:
1. Load the index for similarity search
2. Query using natural language
3. Retrieve relevant complaint chunks
4. Build the RAG chatbot interface

## Dependencies
```
sentence-transformers
faiss-cpu
pandas
numpy
tqdm
```
