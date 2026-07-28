# ═══════════════════════════════════════════════════════════════════
# config.py – Central configuration for the Doxa multi-agent system
# ═══════════════════════════════════════════════════════════════════

import os                                       # used to build file paths dynamically
from pathlib import Path                         # cross-platform path manipulation

# ── Project paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent       # absolute path to doxa_agentic/
RETRIEVAL_DIR = BASE_DIR / "retrieval"           # folder containing PDF knowledge base
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db") # where ChromaDB stores its index on disk
PROMPTS_DIR = BASE_DIR / "prompts"               # folder containing prompt template files

# ── LLM model settings ─────────────────────────────────────────
LLM_MODEL_NAME = "mistral-small-latest"          # Mistral chat model used by all agents
LLM_TEMPERATURE = 0                              # temperature=0 for deterministic outputs
EMBEDDING_MODEL = "mistral-embed"                # Mistral embedding model for vectorization

# ── Chunking parameters (RecursiveCharacterTextSplitter) ───────
CHUNK_SIZE = 1000                                # max characters per chunk
CHUNK_OVERLAP = 200                              # overlap between consecutive chunks (context continuity)

# ── Retrieval settings ──────────────────────────────────────────
TOP_K_RESULTS = 5                                # number of top documents to retrieve from vector store
SIMILARITY_THRESHOLD = 0.3                       # minimum cosine similarity score to keep a result

# ── Evaluator / Decision thresholds ────────────────────────────
CONFIDENCE_THRESHOLD = 0.6                       # if LLM confidence >= this → compose response
                                                 # if LLM confidence <  this → escalate to human

# ── ChromaDB collection name ───────────────────────────────────
COLLECTION_NAME = "doxa_knowledge_base"          # name of the Chroma collection storing all chunks
