"""
Build FAISS vector index from chunks stored in SQLite database
"""
import os
import faiss
import random
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import utils, config

# seeting seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

def build_faiss():
    conn = utils.connect_db(config.SQLITE_PATH)
    rows = conn.execute("SELECT id, chunk_text FROM chunks").fetchall()

    if not rows:
        raise RuntimeError("No chunks found in database. Did you run ingest.py?")

    model = SentenceTransformer(config.EMBED_MODEL)
    texts = [row["chunk_text"] for row in rows]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
    index.add_with_ids(embeddings, np.array([row["id"] for row in rows]))

    # Ensure artifacts folder exists
    index_dir = os.path.dirname(config.FAISS_INDEX_PATH)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(index, config.FAISS_INDEX_PATH)

    print(f"FAISS index built at {config.FAISS_INDEX_PATH}, {len(rows)} vectors.")


if __name__ == "__main__":
    try:
        build_faiss()
    except Exception as e:
        print(f"Error during index build: {e}")
        raise



