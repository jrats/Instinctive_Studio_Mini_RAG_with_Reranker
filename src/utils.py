"""
Utility functions: sqlite database management and small nlp helper functions
"""
import sqlite3
import re
import json
from typing import List, Dict, Any

## Database Helpers

def connect_db(path: str):
    """Connect to SQLite database and return connection"""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row #to access columns with ease
    return conn

def create_schema(conn):
    """Creates database schema"""
    c = conn.cursor()

    #Documents Table
    c.execute("""
    CREATE TABLE IF NOT EXISTS documents(
            doc_index INTEGER PRIMARY KEY,
            filename TEXT,
            title TEXT,
            url TEXT
        )
    """)

    #Chunks Table
    c.execute("""
    CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY AUTOCORRECT,
            doc_index INTEGER,
            title TEXT,
            url TEXT,
            chunk TEXT,
            page_n INTEGER
        )
    """)

def load_sources(sources_path: str) -> List[Dict[str, Any]]:
    """Load sources.json file as a list of {title, url}"""
    with open(sources_path, 'r', encoding ='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("sources.json must be a list of objects containing title and url")
    return data

def insert_document(conn, doc_index: int, filename: str, title: str, url: str):
    """Inserts a document record in the documents table"""
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO documents (doc_index, filename, title, url) VALUES (?, ?, ?, ?)",
              (doc_index, filename, title, url))
    conn.commit()

def insert_chunk(conn, doc_index: int, title: str, url: str, chunk_text: str, page_n: int = None):
    """Inserts a text chunk in the chunks table"""
    c = conn.cursor()
    c.execute("INSERT INTO chunks (doc_index, title, url, chunk_text, page_n) VALUES (?, ?, ?, ?, ?)",
              (doc_index, title, url, chunk_text, page_n))
    rowid = c.lastrowid
    conn.commit()
    return rowid

def fetch_chunks_by_ids(conn, ids: List[int]):
    """Fetch chunk records by their ids"""
    if not ids:
        return []
    placeholder = ",".join("?" for _ in ids) #to accomodate multiple ids if entered
    c = conn.cursor()
    q = f"SELECT id, doc_index, title, url, chunk_text, FROM chunks WHERE id IN ({placeholder})"
    rows = c.execute(q, ids).fetchall()
    rowmap = {r["id"]: r for r in rows}
    return [rowmap[i] for i in ids if i in rowmap]

## NLP Helper functions

def simple_sentence_split(text: str):
    """Sentence splitting for a refined answer(handles e.g., i.e., etc.)"""
    sentences = re.split(r'(?<!\b(?:e\.g|i\.e)\.)(?<=[\.\?\!])\s+', text.strip())
    if len(sentences) == 1 and len(sentences[0]) > 400:
        s = sentences[0]
        pieces = [s[i:i+200] for i in range(0, len(s), 200)]
        return pieces
    return [s for s in sentences if s.strip()]

def token_overlap_score(query: str, sentence: str) -> float:
    """Returns lexical overlap score between query and candidate sentence"""
    qs = set(re.findall(r"\w+", query.lower()))
    ss = set(re.findall(r"\w+", sentence.lower()))
    if not qs:
        return 0.0
    return len(qs & ss) / len(qs)
