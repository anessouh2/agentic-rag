# ═══════════════════════════════════════════════════════════════════
# vectorstore.py – ChromaDB vector store setup with MistralAI embeddings
# ═══════════════════════════════════════════════════════════════════
# This module provides a singleton function `get_vectorstore()` that
# returns a ready-to-use Chroma vector store instance. All other
# modules (ingest.py, solution_finder.py) import from here.
# ═══════════════════════════════════════════════════════════════════

import os                                                       # access environment variables (API key)
import sys                                                      # used to modify Python path
from pathlib import Path                                        # cross-platform path handling

# ── Add parent directory to Python path so we can import config ─
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # adds doxa_agentic/ to sys.path

from dotenv import load_dotenv                                  # loads .env file into environment
load_dotenv()                                                   # execute the .env loading immediately

from langchain_mistralai import MistralAIEmbeddings             # Mistral embedding model wrapper
from langchain_community.vectorstores import Chroma             # ChromaDB vector store integration
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL          # our central config values
from config import COLLECTION_NAME                              # name of the ChromaDB collection

# ── Module-level variable to cache the vector store instance ───
_vectorstore = None                                             # starts as None, set on first call


def get_embeddings():
    """
    Create and return a MistralAI embeddings instance.
    Uses the model name from config and the API key from .env.
    """
    embeddings = MistralAIEmbeddings(                           # instantiate the Mistral embeddings
        model=EMBEDDING_MODEL,                                  # e.g. "mistral-embed"
        api_key=os.getenv("MISTRAL_API_KEY")                    # read API key from environment
    )
    return embeddings                                           # return the ready-to-use embeddings object


def get_vectorstore():
    """
    Singleton function: returns the ChromaDB vector store.
    On the first call it creates the instance and caches it.
    On subsequent calls it returns the cached instance.
    """
    global _vectorstore                                         # reference the module-level cache variable

    if _vectorstore is None:                                    # only create if not already cached
        embeddings = get_embeddings()                           # get the Mistral embeddings instance

        _vectorstore = Chroma(                                  # create the ChromaDB vector store
            collection_name=COLLECTION_NAME,                    # name of the collection in ChromaDB
            embedding_function=embeddings,                      # function used to embed queries
            persist_directory=CHROMA_PERSIST_DIR                # directory where ChromaDB saves its data
        )

    return _vectorstore                                         # return the (possibly cached) vector store
