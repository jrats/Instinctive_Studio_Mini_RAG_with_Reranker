# Mini RAG System with Hybrid Reranking

A small question-answering service built over industrial safety documents that implements both baseline vector similarity search and hybrid reranking to improve retrieval quality.

## Directory Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── questions.txt            # Test questions for evaluation
├── src/
│   ├── config.py            # Configuration constants
│   ├── utils.py             # Database and NLP utilities
│   ├── ingest.py            # PDF ingestion and chunking
│   ├── build_index.py       # FAISS vector index creation
│   ├── reranker.py          # Hybrid reranking implementation
│   ├── search.py            # Search logic (baseline + hybrid modes)
│   ├── api.py               # FastAPI REST endpoint
│   └── evaluate.py          # Evaluation script for comparing modes
├── data/
│   ├── sources.json         # Document metadata (titles + URLs)
│   └── industrial-safety-pdfs/  # PDF documents (not in git)
└── artifacts/               # Generated files (not in git)
    ├── chunks.sqlite        # Chunked documents database
    └── faiss_index.index    # Vector similarity index
```

## Setup and Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Prepare data:**
   - Extract the provided `industrial-safety-pdfs.zip` to `data/industrial-safety-pdfs/`
   - Ensure `data/sources.json` contains document metadata

## How to Run

### 1. Ingest Documents
Process PDFs into chunks and store in SQLite:
```bash
cd src
python ingest.py
```

### 2. Build Vector Index
Create FAISS index from document chunks:
```bash
python build_index.py
```

### 3. Start API Server
Launch the FastAPI service:
```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

### 4. Run Evaluation
Compare baseline vs hybrid reranking performance:
```bash
python evaluate.py
```

## API Usage

The system exposes a single endpoint:

**POST `/ask`**
```json
{
  "q": "What is a Performance Level in functional safety?",
  "k": 3,
  "mode": "baseline"  // or "hybrid"
}
```

**Response:**
```json
{
  "answer": "Performance Level (PL) The performance level is a discrete level...",
  "contexts": [
    {
      "title": "ABB — Safety in Control Systems according to EN ISO 13849-1",
      "url": "https://...",
      "score": 0.8429,
      "text": "Full chunk text...",
      "reranker_used": false
    }
  ],
  "reranker_used": false,
  "top_score": 0.8429
}
```

## Example cURL Requests

### Easy Question (Baseline Mode):
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "What is SISTEMA used for in evaluating machinery safety functions?",
    "k": 3,
    "mode": "baseline"
  }'
```

### Complex Question (Hybrid Mode):
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "How do EN ISO 13849-1 and IEC 62061 differ in their approaches to safety-related control systems?",
    "k": 3,
    "mode": "hybrid"
  }'
```

## Results Comparison

| Question | Baseline Score | Hybrid Score | Baseline Abstained | Hybrid Abstained |
|----------|----------------|--------------|-------------------|------------------|
| Q1: ISO 10218 safety objectives | 0.8406 | 0.7646 | No | No |
| Q2: Performance Level (PL) definition | 0.8429 | 0.8429 | No | No |
| Q3: OSHA machine guarding methods | 0.7454 | 0.7454 | No | No |
| Q4: SISTEMA evaluation tool | 0.7928 | 0.7928 | No | No |
| Q5: EN ISO 13849-1 vs IEC 62061 | 0.8573 | 0.8463 | No | No |
| Q6: EU Machinery Regulation safety components | 0.8104 | 0.8104 | No | No |
| Q7: ISO 10218 food safety (irrelevant) | 0.6673 | 0.6673 | **Yes** | **Yes** |
| Q8: Sci-fi robots (irrelevant) | 0.4916 | 0.4537 | **Yes** | **Yes** |





## Learnings

* Baseline vector search reliably provided direct, grounded answers on this technical dataset.
* Hybrid reranker adds modest improvements for queries with specific technical terms.
* Even if hybrid scores are slightly lower, it surfaces more contextually relevant content that the baseline might miss.
* BM25 keyword matching helps capture nuanced context (e.g., ISO 10218 Q1 – references ANSI/RIA R15.06 and both Parts 1 & 2).
* Abstention logic prevents hallucinations on out-of-scope questions.
* BM25 ensures rare technical terms aren't overlooked, improving coverage and precision for nuanced, terminology-heavy queries.

## Notes

1. `evaluate.py` is provided to compare the hybrid reranker with the baseline approach and was used to generate the results table.
2. The question set was structured to cover a variety of cases: easy, medium, technically hard/specific, and some to test abstention logic.
3. The threshold for abstention was determined after testing and tuning.
4. Different alpha values were tested to balance the contributions of semantic similarity and BM25 in the hybrid reranker.

