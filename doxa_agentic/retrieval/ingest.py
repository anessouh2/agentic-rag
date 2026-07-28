# ═══════════════════════════════════════════════════════════════════
# ingest.py – PDF loading, recursive chunking, and ChromaDB ingestion
# ═══════════════════════════════════════════════════════════════════
# This script reads ALL PDF files from the retrieval/ folder,
# splits them into overlapping chunks using RecursiveCharacterTextSplitter,
# and stores the chunks in ChromaDB with their embeddings.
#
# Run this script ONCE before using the multi-agent system:
#   python doxa_agentic/retrieval/ingest.py
# ═══════════════════════════════════════════════════════════════════

import os                                                       # for file path operations
import sys                                                      # to modify Python import path
import glob                                                     # to find all PDF files with a pattern
from pathlib import Path                                        # cross-platform path manipulation

# ── Add parent directory to path so we can import config ────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # adds doxa_agentic/ to sys.path

from dotenv import load_dotenv                                  # load .env file for API keys
load_dotenv()                                                   # execute the loading right away

from langchain_community.document_loaders import PyPDFLoader    # loads PDF files page by page
from langchain_text_splitters import RecursiveCharacterTextSplitter
 # recursive chunking strategy
from vectorstore import get_vectorstore                         # our ChromaDB singleton
from config import RETRIEVAL_DIR, CHUNK_SIZE, CHUNK_OVERLAP     # import chunking parameters


def load_all_pdfs():
    """
    Find and load ALL PDF files in the retrieval/ directory.
    Returns a flat list of LangChain Document objects (one per PDF page).
    """
    pdf_dir = str(RETRIEVAL_DIR)                                # convert Path to string for glob
    pdf_pattern = os.path.join(pdf_dir, "*.pdf")                # build glob pattern: retrieval/*.pdf
    pdf_files = glob.glob(pdf_pattern)                          # find all matching PDF file paths

    print(f"📂 Found {len(pdf_files)} PDF files in {pdf_dir}") # log how many PDFs we found

    all_documents = []                                          # accumulator for all loaded pages

    for pdf_path in pdf_files:                                  # iterate over each PDF file
        file_name = os.path.basename(pdf_path)                  # extract just the filename
        print(f"  📄 Loading: {file_name}")                     # log which file we're loading

        loader = PyPDFLoader(pdf_path)                          # create a loader for this PDF
        pages = loader.load()                                   # load all pages as Document objects

        for page in pages:                                      # iterate over each page Document
            page.metadata["source_file"] = file_name            # tag each page with its source filename

        all_documents.extend(pages)                             # add all pages to the accumulator
        print(f"    → {len(pages)} pages loaded")               # log how many pages this PDF had

    print(f"\n📊 Total pages loaded: {len(all_documents)}")     # log total across all PDFs
    return all_documents                                        # return the complete list of Documents


def recursive_chunk_documents(documents):
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    This splitter tries to split on these separators IN ORDER:
      1. "\n\n"  (paragraph breaks)
      2. "\n"    (line breaks)
      3. " "     (spaces / words)
      4. ""      (individual characters, last resort)

    It recursively tries the biggest separator first, then falls back
    to smaller ones to keep chunks under the size limit while preserving
    as much semantic coherence as possible.
    """
    text_splitter = RecursiveCharacterTextSplitter(             # create the recursive splitter
        chunk_size=CHUNK_SIZE,                                  # max characters per chunk (1000)
        chunk_overlap=CHUNK_OVERLAP,                            # overlap between chunks (200 chars)
        length_function=len,                                    # use Python's len() to measure size
        separators=["\n\n", "\n", " ", ""],                     # split hierarchy: paragraphs → lines → words → chars
        is_separator_regex=False,                               # treat separators as literal strings
    )

    chunks = text_splitter.split_documents(documents)           # split all documents into chunks

    print(f"🔪 Split into {len(chunks)} chunks")               # log how many chunks were created
    print(f"   Chunk size: {CHUNK_SIZE} chars")                 # log the configured chunk size
    print(f"   Chunk overlap: {CHUNK_OVERLAP} chars")           # log the configured overlap

    return chunks                                               # return the list of chunk Documents


def ingest_into_vectorstore(chunks):
    """
    Embed all chunks and store them in ChromaDB.
    Each chunk is embedded using MistralAI embeddings and stored
    with its text + metadata for later cosine similarity retrieval.
    """
    vectorstore = get_vectorstore()                             # get the ChromaDB singleton instance

    print(f"\n💾 Ingesting {len(chunks)} chunks into ChromaDB...")  # log the start of ingestion

    texts = [chunk.page_content for chunk in chunks]            # extract raw text from each chunk
    metadatas = [chunk.metadata for chunk in chunks]            # extract metadata from each chunk

    vectorstore.add_texts(                                      # add all texts to the vector store
        texts=texts,                                            # the text content to embed and store
        metadatas=metadatas,                                    # metadata (source_file, page number, etc.)
    )

    print(f"✅ Successfully ingested {len(chunks)} chunks!")     # log completion
    print(f"   Persisted to: {vectorstore._persist_directory}") # log where the data is stored


# ── Main execution block ───────────────────────────────────────
if __name__ == "__main__":                                      # only run when script is executed directly
    print("=" * 60)                                             # visual separator
    print("🚀 DOXA Knowledge Base Ingestion Pipeline")          # header
    print("=" * 60)                                             # visual separator

    documents = load_all_pdfs()                                 # STEP 1: load all PDFs

    if not documents:                                           # guard: check if any docs were found
        print("❌ No documents found! Check the retrieval/ folder.")  # error message
        sys.exit(1)                                             # exit with error code

    chunks = recursive_chunk_documents(documents)               # STEP 2: recursive chunking

    ingest_into_vectorstore(chunks)                             # STEP 3: embed + store in ChromaDB

    print("\n" + "=" * 60)                                      # visual separator
    print("🎉 Ingestion complete! Vector store is ready.")      # success message
    print("=" * 60)                                             # visual separator
