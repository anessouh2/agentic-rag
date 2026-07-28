# ═══════════════════════════════════════════════════════════════════
# reranker.py – Simple keyword-overlap reranker for retrieved chunks
# ═══════════════════════════════════════════════════════════════════
# After the vector store returns candidates via cosine similarity,
# this reranker boosts documents that share more keywords with the
# query. It combines the original similarity score with a keyword
# overlap score to produce a final ranking.
# ═══════════════════════════════════════════════════════════════════

from typing import List                                         # type hint for list parameters


def compute_keyword_overlap(text: str, keywords: List[str]) -> float:
    """
    Calculate what fraction of the given keywords appear in the text.

    Args:
        text:     the chunk text to search within
        keywords: list of keywords from the query analyzer

    Returns:
        a float between 0.0 and 1.0 representing the keyword match ratio
    """
    if not keywords:                                            # guard: if no keywords provided
        return 0.0                                              # return zero overlap

    text_lower = text.lower()                                   # lowercase the text for case-insensitive matching
    matches = 0                                                 # counter for how many keywords are found

    for keyword in keywords:                                    # iterate over each keyword
        if keyword.lower() in text_lower:                       # check if keyword exists in text (case-insensitive)
            matches += 1                                        # increment match counter

    overlap_score = matches / len(keywords)                     # ratio of matched keywords to total keywords
    return overlap_score                                        # return the overlap score (0.0 to 1.0)


def rerank_documents(
    documents: List[dict],                                      # list of retrieved docs with score + snippet
    keywords: List[str],                                        # keywords from the query analyzer
    similarity_weight: float = 0.7,                             # how much weight to give cosine similarity
    keyword_weight: float = 0.3,                                # how much weight to give keyword overlap
) -> List[dict]:
    """
    Rerank retrieved documents by combining cosine similarity score
    with keyword overlap score.

    The final score = (similarity_weight × cosine_score) + (keyword_weight × keyword_overlap)

    Args:
        documents:         list of dicts with keys: doc_id, snippet, score
        keywords:          keywords extracted by the query analyzer
        similarity_weight: weight for the original cosine similarity score
        keyword_weight:    weight for the keyword overlap score

    Returns:
        the same list of documents, sorted by final_score descending
    """
    reranked = []                                               # accumulator for reranked results

    for doc in documents:                                       # iterate over each retrieved document
        cosine_score = doc["score"]                             # get the original cosine similarity score
        snippet = doc["snippet"]                                # get the text content of the chunk

        kw_overlap = compute_keyword_overlap(snippet, keywords) # compute keyword overlap for this chunk

        # ── Combine scores with weighted average ───────────────
        final_score = (                                         # calculate the blended final score
            (similarity_weight * cosine_score) +                # weighted cosine similarity component
            (keyword_weight * kw_overlap)                       # weighted keyword overlap component
        )

        reranked_doc = {                                        # build the reranked document dict
            "doc_id": doc["doc_id"],                            # preserve the original document ID
            "snippet": snippet,                                 # preserve the original snippet text
            "score": round(final_score, 4),                     # store the new blended score (rounded)
            "cosine_score": round(cosine_score, 4),             # keep the original cosine score for debugging
            "keyword_overlap": round(kw_overlap, 4),            # keep the keyword overlap for debugging
        }
        reranked.append(reranked_doc)                           # add to the reranked list

    # ── Sort by final score, highest first ─────────────────────
    reranked.sort(key=lambda d: d["score"], reverse=True)       # sort descending by blended score

    return reranked                                             # return the reranked list
