# ═══════════════════════════════════════════════════════════════════
# solution_finder.py – Node 2: Searches the vector store for solutions
# ═══════════════════════════════════════════════════════════════════
# This node takes the summary + keywords from the query_analyzer
# and performs a COSINE SIMILARITY search against the ChromaDB
# vector store. Results are then reranked using keyword overlap.
# ═══════════════════════════════════════════════════════════════════

import sys                                                      # modify Python import path
from pathlib import Path                                        # cross-platform path handling
from typing import List                                         # type hints

# ── Add parent directory to Python path for imports ────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # adds doxa_agentic/ to sys.path

from dotenv import load_dotenv                                  # loads .env file
load_dotenv()                                                   # execute .env loading immediately

from state import TicketState                                   # our shared state TypedDict
from schemas import RetrievedDoc                                # Pydantic schema for a retrieved chunk
from config import TOP_K_RESULTS, SIMILARITY_THRESHOLD          # retrieval config constants
from retrieval.vectorstore import get_vectorstore               # ChromaDB singleton accessor
from retrieval.reranker import rerank_documents                 # keyword-overlap reranker


def solution_finder(state: TicketState) -> dict:
    """
    LangGraph node function: searches the knowledge base for relevant docs.

    Uses COSINE SIMILARITY (built into ChromaDB) to find chunks
    that are semantically close to the ticket summary + keywords.
    Then reranks results using keyword overlap for better precision.

    Reads from state:
        - summary:  the concise ticket summary from query_analyzer
        - keywords: the extracted keywords from query_analyzer

    Writes to state:
        - retrieved_docs: list of dicts with doc_id, snippet, score
    """
    summary = state["summary"]                                  # get the ticket summary from state
    keywords = state["keywords"]                                # get the keywords from state

    # ── Build the search query ─────────────────────────────────
    keyword_string = ", ".join(keywords)                         # join keywords into a comma-separated string
    search_query = f"{summary} {keyword_string}"                # combine summary + keywords into one query

    print(f"\n🔎 [Solution Finder] Searching knowledge base...")  # log that search is starting
    print(f"   Query: {search_query[:100]}...")                  # log first 100 chars of the query

    # ── Get the vector store instance ──────────────────────────
    vectorstore = get_vectorstore()                             # get the ChromaDB singleton

    # ── Perform cosine similarity search ───────────────────────
    # ChromaDB uses cosine similarity by default when you call
    # similarity_search_with_score(). It returns tuples of
    # (Document, score) where lower score = more similar.
    raw_results = vectorstore.similarity_search_with_score(     # execute the similarity search
        query=search_query,                                     # the text query to search for
        k=TOP_K_RESULTS,                                        # number of top results to retrieve (5)
    )

    print(f"   📊 Got {len(raw_results)} raw results from ChromaDB")  # log raw result count

    # ── Convert raw results to our standard format ─────────────
    retrieved_docs = []                                         # accumulator for formatted results

    for i, (doc, distance) in enumerate(raw_results):           # iterate over each (Document, distance) tuple
        # ChromaDB returns L2 distance by default; we convert to
        # a similarity score: similarity = 1 / (1 + distance)
        # This gives us a value between 0 (far) and 1 (identical)
        similarity_score = 1 / (1 + distance)                   # convert distance to similarity (0-1 range)

        if similarity_score < SIMILARITY_THRESHOLD:             # skip results below our threshold
            print(f"   ⚠️ Skipping result {i+1}: score {similarity_score:.4f} < threshold {SIMILARITY_THRESHOLD}")
            continue                                            # move to the next result

        doc_entry = {                                           # build a dict matching RetrievedDoc schema
            "doc_id": doc.metadata.get("source_file", f"chunk_{i}"),  # use source filename as ID
            "snippet": doc.page_content,                        # the actual text content of the chunk
            "score": round(similarity_score, 4),                # the cosine similarity score (rounded)
        }
        retrieved_docs.append(doc_entry)                        # add to our results list

        print(f"   📄 Result {i+1}: score={similarity_score:.4f} | source={doc_entry['doc_id']}")  # log each result

    # ── Rerank the results using keyword overlap ───────────────
    if retrieved_docs and keywords:                             # only rerank if we have docs AND keywords
        print(f"\n   🔄 Reranking {len(retrieved_docs)} documents with keyword overlap...")
        retrieved_docs = rerank_documents(                      # call the reranker
            documents=retrieved_docs,                           # pass the retrieved docs
            keywords=keywords,                                  # pass the keywords for overlap scoring
        )
        print(f"   ✅ Reranking complete. Top score: {retrieved_docs[0]['score']}")  # log top score

    # ── Handle case where no documents were found ──────────────
    if not retrieved_docs:                                      # if no docs passed the threshold
        print(f"   ❌ No documents found above threshold {SIMILARITY_THRESHOLD}")
        retrieved_docs = [{                                     # return a placeholder indicating no results
            "doc_id": "none",                                   # ID indicating no match
            "snippet": "No relevant documents found in the knowledge base.",  # informative message
            "score": 0.0,                                       # zero score
        }]

    print(f"   ✅ Returning {len(retrieved_docs)} documents")    # log final count

    return {                                                    # return dict to update the state
        "retrieved_docs": retrieved_docs,                       # write retrieved docs to state
    }
