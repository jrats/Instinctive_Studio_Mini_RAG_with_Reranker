"""
Search most similar vector using the FAISS index and chunks stored in SQLite
"""
import faiss
from sentence_transformers import SentenceTransformer
import utils, config

def search(query: str, k: int = config.TOP_K):
    model = SentenceTransformer(config.EMBED_MODEL)
    q_vec = model.encode([query]).astype("float32")

    #Normalise the query embedding
    faiss.normalize_L2(q_vec)

    index = faiss.read_index(config.FAISS_INDEX_PATH)
    D, I = index.search(q_vec, k)

    conn = utils.connect_db(config.SQLITE_PATH)
    ids = [int(idx) for idx in I[0] if idx != -1]
    rows = utils.fetch_chunks_by_ids(conn, ids)
    rowmap = {r["id"]: r for r in rows}

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx not in rowmap:
            continue
        chunk = rowmap[idx]
        sentences = utils.simple_sentence_split(chunk["chunk_text"])

        best_sentence, best_overlap = None, 0.0
        for s in sentences:
            ov = utils.token_overlap_score(query, s)
            if ov > best_overlap:
                best_overlap, best_sentence = ov, s
        
        if score < config.TOP_SIM_THRESHOLD:
            answer = None
        elif best_sentence and best_overlap >= config.MIN_SENTENCE_OVERLAP:
            answer = best_sentence.strip()
        else:
            answer = chunk["chunk_text"].strip()[:400].rsplit(".", 1)[0]

        results.append({
            "id": chunk["id"],
            "title": chunk["title"],
            "url": chunk["url"],
            "text": chunk["chunk_text"],
            "score": float(score),
            "answer": answer
        })
    
    conn.close()
    return results





