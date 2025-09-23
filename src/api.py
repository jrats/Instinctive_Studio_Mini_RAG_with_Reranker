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

    results = search.search(req.q, k, mode=req.mode)

    if not results:
        return {"answer": None, "contexts": [], "reranker_used": False, "reason": "No chunk crossed similarity threshold"}

    top = results[0]

    # In case abstained
    reason = None 
    if top.get("answer") is None:
        reason = "Top similarity score below threshold, not confident in answer"
    return {
        "answer": top.get("answer"),
        "contexts": results,
        "reranker_used": top.get("reranker_used",False),
        "top_score": top.get("score"),
        "reason_to_abstain": reason
    }


