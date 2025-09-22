# paths
PDF_DIR = "../data/industrial-safety-pdfs"
SOURCES_PATH = "../data/sources.json"
SQLITE_PATH = "../artifacts/chunks.sqlite"
FAISS_INDEX_PATH = "../artifacts/faiss_index.index"
EMBED_MODEL = "all-MiniLM-L6-v2"

# chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# search
TOP_K = 3
TOP_SIM_THRESHOLD = 0.62
MIN_SENTENCE_OVERLAP = 0.15

# setting seed to make answers deterministic
SEED = 42