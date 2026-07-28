# ═══════════════════════════════════════════════════════════════════
# graph.py – Builds and compiles the LangGraph StateGraph
# ═══════════════════════════════════════════════════════════════════
# This is the BRAIN of the multi-agent system. It defines:
#   1. The graph with all 5 nodes
#   2. The edges connecting them in sequence
#   3. A CONDITIONAL edge from evaluator → response_composer or escalation_handler
#   4. The compiled, runnable graph
#
# Flow:
#   START → query_analyzer → solution_finder → evaluator_decider
#     ├── (confidence >= 0.6) → response_composer → END
#     └── (confidence <  0.6) → escalation_handler → END
# ═══════════════════════════════════════════════════════════════════

import sys                                                      # modify Python import path
from pathlib import Path                                        # cross-platform path handling

# ── Add this directory to Python path ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))        # adds doxa_agentic/ to sys.path

from langgraph.graph import StateGraph, END                     # StateGraph builder and END sentinel
from state import TicketState                                   # our shared state TypedDict

# ── Import all 5 node functions ────────────────────────────────
from nodes.query_analyzer import query_analyzer                 # Node 1: ticket analysis
from nodes.solution_finder import solution_finder               # Node 2: vector store search
from nodes.evaluator_decider import evaluator_decider           # Node 3: confidence evaluation
from nodes.response_composer import response_composer           # Node 4: draft response
from nodes.escalation_handler import escalation_handler         # Node 5: human escalation


def route_after_evaluation(state: TicketState) -> str:
    """
    Conditional routing function called after the evaluator_decider node.

    Reads the 'decision' field from state and returns the name of
    the next node to execute:
      - "respond"  → go to "response_composer"
      - "escalate" → go to "escalation_handler"

    This function is passed to `add_conditional_edges()` in the graph.
    """
    decision = state.get("decision", "escalate")                # read the decision (default: escalate)

    if decision == "respond":                                   # if evaluator said docs are sufficient
        print(f"\n🔀 [Router] Decision: RESPOND → routing to Response Composer")
        return "response_composer"                              # route to response_composer node
    else:                                                       # if evaluator said docs are insufficient
        print(f"\n🔀 [Router] Decision: ESCALATE → routing to Escalation Handler")
        return "escalation_handler"                             # route to escalation_handler node


def build_graph():
    """
    Construct the complete LangGraph StateGraph with all nodes and edges.
    Returns a compiled graph ready to be invoked with a TicketState.
    """

    # ── Step 1: Create the StateGraph builder ──────────────────
    graph_builder = StateGraph(TicketState)                      # initialize graph with our state schema

    # ── Step 2: Add all 5 nodes to the graph ───────────────────
    graph_builder.add_node("query_analyzer", query_analyzer)    # register Node 1: ticket analysis
    graph_builder.add_node("solution_finder", solution_finder)  # register Node 2: vector search
    graph_builder.add_node("evaluator_decider", evaluator_decider)  # register Node 3: evaluation
    graph_builder.add_node("response_composer", response_composer)  # register Node 4: response draft
    graph_builder.add_node("escalation_handler", escalation_handler)  # register Node 5: escalation

    # ── Step 3: Set the entry point ────────────────────────────
    graph_builder.set_entry_point("query_analyzer")             # the graph starts at query_analyzer

    # ── Step 4: Add sequential edges (linear flow) ─────────────
    graph_builder.add_edge("query_analyzer", "solution_finder")     # after analysis → search docs
    graph_builder.add_edge("solution_finder", "evaluator_decider")  # after search → evaluate results

    # ── Step 5: Add the CONDITIONAL edge (branching) ───────────
    graph_builder.add_conditional_edges(                         # add branching after evaluator
        "evaluator_decider",                                    # source node: the evaluator
        route_after_evaluation,                                 # routing function to call
        {                                                       # mapping of return values → node names
            "response_composer": "response_composer",           # if function returns "response_composer"
            "escalation_handler": "escalation_handler",         # if function returns "escalation_handler"
        }
    )

    # ── Step 6: Add terminal edges (both paths lead to END) ────
    graph_builder.add_edge("response_composer", END)            # after composing response → finish
    graph_builder.add_edge("escalation_handler", END)           # after escalation → finish

    # ── Step 7: Compile the graph ──────────────────────────────
    compiled_graph = graph_builder.compile()                     # compile into a runnable graph

    print("✅ [Graph] Multi-agent graph compiled successfully!") # log success

    return compiled_graph                                       # return the compiled graph
