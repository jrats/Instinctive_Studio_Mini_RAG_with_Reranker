"""
Ingests PDFS and stores the chunks generated in an SQLite database
"""
import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import utils, config


def ingest_pdfs():
    # Ensure folder for SQLite database exists
    db_dir = os.path.dirname(config.SQLITE_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = utils.connect_db(config.SQLITE_PATH)
    utils.create_schema(conn)
    sources = utils.load_sources(config.SOURCES_PATH)

    pdf_files = sorted(glob.glob(os.path.join(config.PDF_DIR, "*.pdf")))

    if len(pdf_files) != len(sources):
        print(f"Warning: mismatch between number of PDFs ({len(pdf_files)}) and sources.json entries ({len(sources)})")

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP, 
        length_function=len, 
        separators=['\n\n', '\n', '.', '?', '!', ' ', '']
    )   
    
    for idx, (filename, source) in enumerate(zip(pdf_files, sources)):
        loader = PyMuPDFLoader(filename)
        docs = loader.load()
        utils.insert_document(conn, idx+1, filename, source["title"], source["url"])

        for doc in docs:
            for chunk in splitter.split_text(doc.page_content):
                utils.insert_chunk(conn, idx+1, source["title"], source["url"], chunk)

    print(f"Ingestion complete. {len(pdf_files)} documents processed.")


if __name__ == "__main__":
    try:
        ingest_pdfs()
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise


