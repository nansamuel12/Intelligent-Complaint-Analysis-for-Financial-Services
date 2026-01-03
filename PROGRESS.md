# RAG Complaint Chatbot - Progress Summary

## Project Status

### ✅ Task 1: Exploratory Data Analysis & Preprocessing (COMPLETED)
- **Script**: `src/data_processing.py`
- **Output**: `data/filtered_complaints.csv` (454K rows)
- **Status**: Done & Merged.

### 🔄 Task 2: Chunking, Embedding & Vector Indexing (RUNNING)
- **Script**: `src/create_embeddings.py`
- **Status**: **Generating Embeddings (In Progress)**.
- **Output**: `vector_store/` (FAISS index being built).

### 📝 Task 3: RAG Pipeline & Evaluation (CODE READY)
- **Modules Created**:
  - `src/rag_pipeline.py`: Handles loading vector store, retrieval, and LLM generation.
  - `src/evaluate_rag.py`: Automation script for running 10 test questions.
  - `.env.example`: Configuration template for API keys.
- **Next Steps**:
  1. Wait for Task 2 to finish.
  2. Add OpenAI API Key to `.env`.
  3. Run `python src/evaluate_rag.py`.
  4. Manually score the results in `data/rag_evaluation.csv`.

---

## Project Structure

```
rag-complaint-chatbot/
├── .vscode/
├── .github/
├── data/
│   ├── raw/
│   ├── filtered_complaints.csv
│   └── rag_evaluation.csv (To be generated)
├── vector_store/          (Generating...)
├── notebooks/
├── src/
│   ├── data_processing.py      # Task 1
│   ├── create_embeddings.py    # Task 2
│   ├── test_vector_store.py    # Task 2
│   ├── rag_pipeline.py         # Task 3 (New)
│   └── evaluate_rag.py         # Task 3 (New)
├── app.py
├── requirements.txt
├── .gitignore
└── PROGRESS.md
```

## How to Run Evaluation (Once Task 2 Completes)

1. **Set up API Key**:
   Copy `.env.example` to `.env` and add your key:
   ```bash
   cp .env.example .env
   # Edit .env with your key
   ```

2. **Run Evaluation**:
   ```bash
   python src/evaluate_rag.py
   ```

3. **Analyze Results**:
   Open `data/rag_evaluation.csv` and rate the answers.
