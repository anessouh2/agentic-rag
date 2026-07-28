# ═══════════════════════════════════════════════════════════════════
# evaluator_decider.py – Node 3: Evaluates retrieved docs & decides routing
# ═══════════════════════════════════════════════════════════════════
# This node receives the ticket info + retrieved documents and uses
# a Mistral LLM to judge whether the docs are sufficient to answer.
# It outputs a confidence score (0-1) and a routing decision:
#   - "respond"  → route to response_composer (docs are sufficient)
#   - "escalate" → route to escalation_handler (need human help)
# ═══════════════════════════════════════════════════════════════════

import os                                                       # access environment variables
import sys                                                      # modify Python import path
from pathlib import Path                                        # cross-platform path handling

# ── Add parent directory to Python path for imports ────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # adds doxa_agentic/ to sys.path

from dotenv import load_dotenv                                  # loads .env file
load_dotenv()                                                   # execute .env loading immediately

from langchain_mistralai import ChatMistralAI                   # Mistral AI chat model wrapper
from state import TicketState                                   # our shared state TypedDict
from schemas import EvaluationResult                            # Pydantic schema for structured output
from config import LLM_MODEL_NAME, LLM_TEMPERATURE              # model config values
from config import PROMPTS_DIR, CONFIDENCE_THRESHOLD            # paths and thresholds


def _load_prompt_template():
    """
    Load the evaluator prompt template from the prompts/ folder.
    Returns the template string with placeholders.
    """
    prompt_path = PROMPTS_DIR / "evaluator.txt"                 # build path to the evaluator prompt
    with open(prompt_path, "r", encoding="utf-8") as f:         # open the file in read mode
        template = f.read()                                     # read the entire file content
    return template                                             # return the template string


def _format_retrieved_docs(docs: list) -> str:
    """
    Format the retrieved documents into a readable string for the LLM.
    Each doc is numbered and shows its source and similarity score.
    """
    if not docs:                                                # guard: if no docs provided
        return "No documents were retrieved."                   # return informative message

    formatted_parts = []                                        # accumulator for formatted doc strings

    for i, doc in enumerate(docs, 1):                           # iterate with 1-based index
        part = (                                                # build a formatted string for this doc
            f"Document {i}:\n"                                  # document number header
            f"  Source: {doc.get('doc_id', 'unknown')}\n"       # source file name
            f"  Score: {doc.get('score', 0.0)}\n"               # similarity score
            f"  Content: {doc.get('snippet', '')}\n"            # the actual text content
        )
        formatted_parts.append(part)                            # add to the accumulator

    return "\n".join(formatted_parts)                           # join all parts with newlines


# ── Initialize the Mistral LLM ─────────────────────────────────
llm = ChatMistralAI(                                            # create the Mistral chat model
    model=LLM_MODEL_NAME,                                       # e.g. "mistral-small-latest"
    temperature=LLM_TEMPERATURE,                                # temperature=0 for deterministic output
    api_key=os.getenv("MISTRAL_API_KEY"),                        # API key from .env
)

# ── Wrap the LLM with structured output ────────────────────────
structured_llm = llm.with_structured_output(EvaluationResult)   # forces LLM to return EvaluationResult schema


def evaluator_decider(state: TicketState) -> dict:
    """
    LangGraph node function: evaluates whether retrieved docs can answer the ticket.

    Reads from state:
        - subject:        the ticket subject line
        - summary:        the concise ticket summary
        - keywords:       the extracted keywords
        - retrieved_docs: the documents from solution_finder

    Writes to state:
        - confidence_score: float (0-1) indicating answer quality
        - decision:         "respond" or "escalate"
    """
    subject = state["subject"]                                  # get the ticket subject
    summary = state["summary"]                                  # get the ticket summary
    keywords = state["keywords"]                                # get the keywords
    retrieved_docs = state["retrieved_docs"]                    # get the retrieved documents

    template = _load_prompt_template()                          # load the evaluator prompt template

    formatted_docs = _format_retrieved_docs(retrieved_docs)     # format docs into readable string

    prompt = template.format(                                   # fill in all placeholders
        subject=subject,                                        # insert ticket subject
        summary=summary,                                        # insert ticket summary
        keywords=", ".join(keywords),                           # insert keywords as comma-separated string
        retrieved_docs=formatted_docs,                          # insert formatted documents
    )

    print(f"\n⚖️ [Evaluator] Evaluating {len(retrieved_docs)} retrieved documents...")  # log start

    result = structured_llm.invoke(prompt)                      # call the LLM for structured evaluation

    # ── Enforce the confidence threshold ───────────────────────
    decision = result.decision.lower().strip()                  # normalize the decision string
    if result.confidence_score >= CONFIDENCE_THRESHOLD:         # if confidence is above threshold
        decision = "respond"                                    # override to "respond" (compose answer)
    else:                                                       # if confidence is below threshold
        decision = "escalate"                                   # override to "escalate" (human needed)

    print(f"   📊 Confidence: {result.confidence_score}")        # log the confidence score
    print(f"   🎯 Decision: {decision}")                         # log the routing decision
    print(f"   💭 Reasoning: {result.reasoning}")                # log the LLM's reasoning

    return {                                                    # return dict to update the state
        "confidence_score": result.confidence_score,            # write confidence score to state
        "decision": decision,                                   # write decision to state
    }
