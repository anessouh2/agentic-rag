# ═══════════════════════════════════════════════════════════════════
# schemas.py – Pydantic models for structured LLM outputs
# ═══════════════════════════════════════════════════════════════════
# These schemas are passed to `llm.with_structured_output(...)` so
# the LLM is forced to return JSON matching these exact fields.
# ═══════════════════════════════════════════════════════════════════

from pydantic import BaseModel, Field            # BaseModel = schema class, Field = field metadata


# ── Schema for the Query Analyzer node ──────────────────────────
class QueryAnalysis(BaseModel):
    """What the query_analyzer LLM must return."""
    summary: str = Field(                         # concise ticket summary
        description="A concise summary of the customer support ticket, under 100 words"
    )
    keywords: list[str] = Field(                  # relevant search keywords
        description="5 to 10 relevant keywords extracted from the ticket"
    )


# ── Schema for a single retrieved document chunk ───────────────
class RetrievedDoc(BaseModel):
    """Represents one chunk returned by the vector store."""
    doc_id: str = Field(                          # unique ID of the chunk in ChromaDB
        description="The unique identifier of the document chunk"
    )
    snippet: str = Field(                         # actual text content of the chunk
        description="The text content of the retrieved chunk"
    )
    score: float = Field(                         # cosine similarity score (0 to 1)
        description="Cosine similarity score between query and this chunk"
    )


# ── Schema for the Evaluator / Decider node ────────────────────
class EvaluationResult(BaseModel):
    """What the evaluator_decider LLM must return."""
    confidence_score: float = Field(              # confidence that the docs answer the ticket (0-1)
        description="Confidence score from 0.0 to 1.0 indicating how well the retrieved documents answer the ticket"
    )
    decision: str = Field(                        # routing: "respond" if confident, "escalate" if not
        description="Decision: 'respond' if the documents are sufficient, 'escalate' if a human agent is needed"
    )
    reasoning: str = Field(                       # short explanation of why the LLM chose this decision
        description="Brief explanation of why this decision was made"
    )


# ── Schema for the Response Composer node ──────────────────────
class ComposedResponse(BaseModel):
    """What the response_composer LLM must return."""
    response: str = Field(                        # the polished customer-facing response
        description="A professional, helpful customer-facing response that addresses the ticket"
    )