# ═══════════════════════════════════════════════════════════════════
# main.py – Entry point / test runner for the Doxa multi-agent system
# ═══════════════════════════════════════════════════════════════════
# This script creates a sample support ticket and runs it through
# the entire multi-agent graph to demonstrate the full pipeline:
#   query_analyzer → solution_finder → evaluator_decider
#     → response_composer (or escalation_handler)
# ═══════════════════════════════════════════════════════════════════

import sys                                                      # modify Python import path
from pathlib import Path                                        # cross-platform path handling

# ── Add this directory to Python path ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))        # adds doxa_agentic/ to sys.path

from dotenv import load_dotenv                                  # loads .env file for API keys
load_dotenv()                                                   # execute .env loading immediately

from graph import build_graph                                   # import the graph builder function


def run_ticket(ticket_id: str, subject: str, description: str):
    """
    Run a single support ticket through the entire multi-agent pipeline.

    Args:
        ticket_id:   unique identifier for the ticket
        subject:     the ticket subject line
        description: the full ticket body/description

    Returns:
        The final state dict after the graph has finished executing.
    """

    # ── Build and compile the graph ────────────────────────────
    graph = build_graph()                                       # create the compiled LangGraph

    # ── Prepare the initial state ──────────────────────────────
    initial_state = {                                           # create the starting state dict
        "ticket_id": ticket_id,                                 # set the ticket ID
        "subject": subject,                                     # set the ticket subject
        "description": description,                             # set the ticket description
        # All other fields will be populated by the nodes:
        "summary": "",                                          # will be set by query_analyzer
        "keywords": [],                                         # will be set by query_analyzer
        "retrieved_docs": [],                                   # will be set by solution_finder
        "confidence_score": 0.0,                                # will be set by evaluator_decider
        "decision": "",                                         # will be set by evaluator_decider
        "response": "",                                         # will be set by response_composer or escalation_handler
        "escalation_reason": "",                                # will be set by escalation_handler (if triggered)
    }

    print("=" * 70)                                             # visual separator
    print("🚀 DOXA MULTI-AGENT SUPPORT SYSTEM")                 # header
    print("=" * 70)                                             # visual separator
    print(f"📋 Ticket ID: {ticket_id}")                          # log the ticket ID
    print(f"📌 Subject: {subject}")                              # log the subject
    print(f"📝 Description: {description[:100]}...")             # log first 100 chars of description
    print("=" * 70)                                             # visual separator

    # ── Invoke the graph ───────────────────────────────────────
    final_state = graph.invoke(initial_state)                   # run the full pipeline

    # ── Display the results ────────────────────────────────────
    print("\n" + "=" * 70)                                      # visual separator
    print("📊 FINAL RESULTS")                                    # results header
    print("=" * 70)                                             # visual separator
    print(f"🎯 Decision: {final_state.get('decision', 'N/A')}") # show the routing decision
    print(f"📊 Confidence: {final_state.get('confidence_score', 'N/A')}")  # show confidence
    print(f"\n📝 Response:\n{final_state.get('response', 'N/A')}")  # show the final response

    if final_state.get("escalation_reason"):                    # if the ticket was escalated
        print(f"\n🚨 Escalation Reason:\n{final_state['escalation_reason']}")  # show why

    print("=" * 70)                                             # visual separator

    return final_state                                          # return the full final state


# ── Main execution block ───────────────────────────────────────
if __name__ == "__main__":                                      # only run when executed directly

    # ── Sample test ticket ─────────────────────────────────────
    test_ticket_id = "DOXA-2024-001"                            # sample ticket ID
    test_subject = "Impossible de me connecter à mon compte"    # sample subject (French: "Can't login")
    test_description = (                                        # sample ticket description
        "Bonjour, depuis ce matin je n'arrive plus à me connecter "  # "Hello, since this morning I can't login"
        "à mon compte Doxa. J'ai essayé de réinitialiser mon mot "   # "I tried to reset my password"
        "de passe mais je ne reçois pas l'email de réinitialisation. "  # "but I don't receive the reset email"
        "Mon adresse email est correcte. Pouvez-vous m'aider ?"  # "My email is correct. Can you help?"
    )

    # ── Run the test ticket through the pipeline ───────────────
    result = run_ticket(                                        # invoke the full multi-agent pipeline
        ticket_id=test_ticket_id,                               # pass the test ticket ID
        subject=test_subject,                                   # pass the test subject
        description=test_description,                           # pass the test description
    )
