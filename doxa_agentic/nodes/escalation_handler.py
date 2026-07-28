# ═══════════════════════════════════════════════════════════════════
# escalation_handler.py – Node 5: Handles ticket escalation to humans
# ═══════════════════════════════════════════════════════════════════
# This node is called when the evaluator decides that the retrieved
# documents are NOT sufficient to answer the ticket. It flags the
# ticket for review by a human support agent and records the reason.
# ═══════════════════════════════════════════════════════════════════

import sys                                                      # modify Python import path
from pathlib import Path                                        # cross-platform path handling

# ── Add parent directory to Python path for imports ────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # adds doxa_agentic/ to sys.path

from state import TicketState                                   # our shared state TypedDict
from config import CONFIDENCE_THRESHOLD                         # the threshold that was not met


def escalation_handler(state: TicketState) -> dict:
    """
    LangGraph node function: escalates the ticket to a human agent.

    This is the "fallback" path when the evaluator decides the
    knowledge base does not have enough information to answer.

    Reads from state:
        - ticket_id:        the ticket identifier
        - subject:          the ticket subject line
        - summary:          the ticket summary
        - confidence_score: the evaluator's confidence score
        - retrieved_docs:   what was found (for context)

    Writes to state:
        - escalation_reason: why the ticket was escalated
        - response:          a message indicating escalation
    """
    ticket_id = state["ticket_id"]                              # get the ticket ID from state
    subject = state["subject"]                                  # get the ticket subject from state
    summary = state["summary"]                                  # get the ticket summary from state
    confidence = state.get("confidence_score", 0.0)             # get the confidence score (default 0.0)
    retrieved_docs = state.get("retrieved_docs", [])            # get retrieved docs (default empty list)

    # ── Build the escalation reason ────────────────────────────
    num_docs = len(retrieved_docs)                              # count how many docs were retrieved
    reason = (                                                  # construct a detailed escalation reason
        f"Ticket '{subject}' (ID: {ticket_id}) escalated to human agent. "  # which ticket
        f"Confidence score: {confidence:.2f} "                  # what the confidence was
        f"(threshold: {CONFIDENCE_THRESHOLD}). "                # what the threshold is
        f"{num_docs} document(s) were retrieved but deemed insufficient."   # how many docs were found
    )

    # ── Build a customer-facing escalation message ─────────────
    escalation_response = (                                     # message to show the customer
        "Thank you for contacting Doxa support. "               # opening acknowledgement
        "Your request has been forwarded to a specialized agent "  # inform about escalation
        "who will review your case and respond shortly. "       # set expectations
        f"Your ticket reference is: {ticket_id}."              # provide reference number
    )

    print(f"\n🚨 [Escalation Handler] Ticket escalated!")        # log that escalation happened
    print(f"   📋 Ticket ID: {ticket_id}")                       # log the ticket ID
    print(f"   📊 Confidence was: {confidence:.2f}")             # log the confidence score
    print(f"   📝 Reason: {reason}")                             # log the full reason

    return {                                                    # return dict to update the state
        "escalation_reason": reason,                            # write the escalation reason to state
        "response": escalation_response,                        # write the escalation message to state
    }
