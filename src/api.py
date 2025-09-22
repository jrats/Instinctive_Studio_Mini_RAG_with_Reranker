"""
FastAPI RAG API
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel 
from typing import Optional 
import os
import search
import config

SQLITE_PATH = os.environ.get("SQLITE_PATH", "artifacts/chunks.sqlite")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "artifacts/faiss_index.index")

app = FastAPI(title="Mini RAG API")

class AskRequest(BaseModel):
    q: str
    k: Optional[int] = None
    mode: Optional[str] = "baseline"


@app.post("/ask")
def ask_post(req: AskRequest):
    """Handle POST /ask with JSON body."""
    k = req.k if req.k and req.k > 0 else config.TOP_K
    if not req.q:
        raise HTTPException(status_code=400, detail="query is required and k must be > 0")

    results = search.search(req.q, k)

    if not results:
        return {"answer": None, "contexts": [], "reranker_used": False, "reason": "No chunk crossed similarity threshold"}

    top = results[0]
    return {
        "answer": top["answer"],
        "contexts": results,
        "reranker_used": False,
        "top_score": top["score"]
    }


# @app.get("/ask")
# def ask_get(q: str = Query(..., description="Question string"), k: Optional[int] = Query(None, description="Top-k results")):
#     """Handle GET /ask?q=...&k=..."""
#     k_val = k if k and k > 0 else config.TOP_K
#     if not q:
#         raise HTTPException(status_code=400, detail="query parameter q is required")

#     results = search.search(q, k_val)

#     if not results:
#         return {"answer": None, "contexts": [], "reranker_used": False, "reason": "No chunk crossed similarity threshold"}

#     top = results[0]
#     return {
#         "answer": top["answer"],
#         "contexts": results,
#         "reranker_used": False,
#         "top_score": top["score"]
#     }
