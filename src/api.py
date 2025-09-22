"""
FastAPI RAG API
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from typing import Optional 
import os
import search

SQLITE_PATH = os.environ.get("SQLITE_PATH", "artifacts/chunks.sqlite")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "artifacts/faiss_index.index")

app = FastAPI(title="Mini RAG API")

class AskRequest(BaseModel):
    q: str
    k: Optional[int] = 3
    mode: Optional[str] = "baseline"

@app.post("/ask")
def ask(req: AskRequest):
    if not req.q or req.k <= 0:
        raise HTTPException(status_code=400, detail="query is required and k must be > 0")
    
    results = search.search(req.q, SQLITE_PATH, FAISS_INDEX_PATH, req.k)

    if not results:
        return {"answer": None, "contexts": [], "reranker_used": False, "reason": "No chunk crossed similarity threshold"}
    
    top = results[0]
    return {
        "answer": top["answer"],
        "contexts": results,
        "reranker_used": False,
        "top_score": top["score"]
    }
