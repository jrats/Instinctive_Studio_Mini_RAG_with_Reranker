"""
Build FAISS vector index from chunks stored in SQLite database
"""
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import utils

def build_faiss(db_path: str, out_path: str, embed_model: str = "all-MiniLM-L6-v2"):
    conn = utils.connect_db(db_path)
    rows = conn.execute("SELECT id, chunk_text FROM chunks").fetchall()

    model = SentenceTransformer(embed_model)
    texts = [row["chunk_text"] for row in rows]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32") #faiss takes float32 vectors

    #Normalising embeddings for cosine similarity
    faiss.mormalize_L2(embeddings)

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
    index.add_with_ids(embeddings, np.array([row["id"] for row in rows]))

    faiss.write_index(index, out_path)
    print(f"FAISS index built at {out_path}, {len(rows)} vectors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="artifacts/chunks.sqlite")
    parser.add_argument("--out", default="artifacts/faiss_index.index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    build_faiss(args.db, args.out, args.model)
    

