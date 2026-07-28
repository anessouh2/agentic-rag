# ═══════════════════════════════════════════════════════════════════
# state.py – Shared state definition for the entire LangGraph pipeline
# ═══════════════════════════════════════════════════════════════════
# This TypedDict is the "shared memory" that every node reads from
# and writes to. LangGraph passes it automatically between nodes.
# ═══════════════════════════════════════════════════════════════════

from typing import TypedDict, List, Optional     # typing imports for state field annotations

class TicketState(TypedDict):
    """
    Full state that flows through the multi-agent graph.
    Each node reads what it needs and writes its own output fields.
    """

    # ── INPUT fields (provided by the user / ticket system) ─────
    ticket_id: str                                # unique identifier for the support ticket
    subject: str                                  # subject line of the customer ticket
    description: str                              # full body / description of the ticket

    # ── OUTPUT of query_analyzer node ───────────────────────────
    summary: str                                  # concise summary of the ticket (under 100 words)
    keywords: List[str]                           # 5-10 relevant keywords extracted from the ticket

    # ── OUTPUT of solution_finder node ──────────────────────────
    retrieved_docs: List[dict]                    # list of retrieved document chunks (doc_id, snippet, score)

    # ── OUTPUT of evaluator_decider node ────────────────────────
    confidence_score: float                       # how confident the system is that docs answer the ticket (0-1)
    decision: str                                 # routing decision: "respond" or "escalate"

    # ── OUTPUT of response_composer node ────────────────────────
    response: str                                 # the final customer-facing response text

    # ── OUTPUT of escalation_handler node ───────────────────────
    escalation_reason: str                        # why the ticket was escalated to a human agent
