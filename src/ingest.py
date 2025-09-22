"""
Ingests PDFS and stores the chunks generated in an SQLite database
"""
import os
import argparse
import glob
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import utils, config

#Initialising paths for all data, citations and sqlite database
# PDF_DIR = os.getenv('PDF_DIR','data/industrial-safety-pdfs')
# DB_PATH = os.getenv('DB_PATH','data/sqlite.db')
# SOURCES_PATH = os.getenv('SOURCES_PATH','data/sources.json')


def ingest_pdfs(pdf_dir: str, sources_path, db_path: str, chunk_size: int = 300, chunk_overlap: int = 50):
    conn = utils.connect_db(db_path)
    utils.create_schema(conn)
    sources = utils.load_sources(sources_path)

    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

    if len(pdf_files) != len(sources):
        print(f"Warning: mismatch between number of PDFs ({len(pdf_files)}) and sources.json entries ({len(sources)})")

    #Initialising text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, 
                                            chunk_overlap=config.chunk_overlap, 
                                            length_function=len(), 
                                            separators=['\n\n', '\n','.','?','!',' ',''])   
    
    for idx, (filename, source) in enumerate(zip(pdf_files, sources)):
        loader = PyMuPDFLoader(filename)
        docs = loader.load()
        utils.insert_document(conn, idx+1, filename, source["title"], source["url"])

        for doc in docs:
            for chunk in splitter.split_text(doc.page_content):
                utils.insert_chunk(conn, idx+1, source["title"], source["url"], chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="data/industrial-safety-pdfs")
    parser.add_argument("--sources", default="data/sources.json")
    parser.add_argument("--out", default="artifacts/chunks.sqlite")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    args = parser.parse_args()

    ingest_pdfs(args.pdf_dir, args.sources, args.out, args.chunk_size, args.chunk_overlap)




