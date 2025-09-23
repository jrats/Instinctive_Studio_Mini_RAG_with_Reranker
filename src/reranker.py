"""
Hybrid reranker: vector similarity + lexical keyword score (BM25-style) to reorder chunks
"""
import re
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi


def normalize_scores(scores: List[float]) -> List[float]:
    """Min - max normalization from range 0-1"""
    if not scores:
        return scores
    arr = np.array(scores, dtype=float)
    min_v, max_v = arr.min(), arr.max()
    if max_v == min_v:
        return [1.0 for _ in scores]
    return ((arr-min_v) / (max_v - min_v)).tolist()

def rerank(query: str, candidates: List[Dict], alpha: float = 0.6):
    """
    Rerank candidates using a hybrid of FAISS vector score + BM25 keyword score
    """

    if not candidates:
        return []
    
    ## vector scores
    vector_scores = [c["score"] for c in candidates]
    vector_scores = normalize_scores(vector_scores)

    ## BM25 keyword based scores
    corpus = [c["text"] for c in candidates]
    tokenized_corpus = [re.findall(r"\w+", text.lower()) for text in corpus]
    bm25  = BM25Okapi(tokenized_corpus)

    qs = re.findall(r"\w+", query.lower())
    bm25_scores = bm25.get_scores(qs)
    keyword_scores = normalize_scores(bm25_scores.tolist())

    #combine scores
    for c, v, k in zip(candidates, vector_scores, keyword_scores):
        c["final_score"] = alpha * v + (1-alpha) * k

    #sort using the hybrid score calculated
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    return candidates
