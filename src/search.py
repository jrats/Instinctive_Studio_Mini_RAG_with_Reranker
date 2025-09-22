"""
Search most similar vector using the FAISS index and chunks stored in SQLite
"""
import argparse
import faiss
from sentence_transformers import SentenceTransformer
import utils

TOP_SIM_THRESHOLD = 0.62
MIN_SENTENCE_OVERLAP = 0.15

def search(query: str, db_path: str, index_path: str, k: int = 3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_vec = model.encode([query]).astype("float32")

    #Normalise the query embedding
    faiss.normalize_L2(q_vec)

    index = faiss.read_index(index_path)
    D, I = index.search(q_vec, k)

    conn = utils.connect_db(db_path)
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
        
        if score < TOP_SIM_THRESHOLD:
            answer = None
        elif best_sentence and best_overlap >= MIN_SENTENCE_OVERLAP:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="artifacts/chunks.sqlite")
    parser.add_argument("--index", default="artifacts/faiss_index.index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    res = search(args.query, args.db, args.index, args.k)
    




