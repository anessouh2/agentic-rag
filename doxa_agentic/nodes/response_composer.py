# ═══════════════════════════════════════════════════════════════════
# response_composer.py – Node 4: Composes the customer-facing response
# ═══════════════════════════════════════════════════════════════════
# This node is called when the evaluator decides that the retrieved
# documents are sufficient to answer the ticket. It uses a Mistral
# LLM to draft a professional, helpful response for the customer.
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
from schemas import ComposedResponse                            # Pydantic schema for structured output
from config import LLM_MODEL_NAME, LLM_TEMPERATURE              # model config values
from config import PROMPTS_DIR                                  # path to prompt templates


def _load_prompt_template():
    """
    Load the response composer prompt template from the prompts/ folder.
    Returns the template string with placeholders.
    """
    prompt_path = PROMPTS_DIR / "response_composer.txt"         # build path to the prompt file
    with open(prompt_path, "r", encoding="utf-8") as f:         # open the file in read mode
        template = f.read()                                     # read the entire file content
    return template                                             # return the template string


def _format_retrieved_docs(docs: list) -> str:
    """
    Format retrieved documents into a readable context block for the LLM.
    Each doc shows its source and full text content.
    """
    if not docs:                                                # guard: if no docs provided
        return "No relevant documents available."               # return fallback message

    formatted_parts = []                                        # accumulator for formatted sections

    for i, doc in enumerate(docs, 1):                           # iterate with 1-based index
        part = (                                                # build formatted string for this doc
            f"[Source: {doc.get('doc_id', 'unknown')}]\n"       # document source header
            f"{doc.get('snippet', '')}\n"                       # the actual text content
        )
        formatted_parts.append(part)                            # add to accumulator

    return "\n---\n".join(formatted_parts)                      # join with horizontal separators


# ── Initialize the Mistral LLM ─────────────────────────────────
llm = ChatMistralAI(                                            # create the Mistral chat model
    model=LLM_MODEL_NAME,                                       # e.g. "mistral-small-latest"
    temperature=0.3,                                            # slightly creative for natural responses
    api_key=os.getenv("MISTRAL_API_KEY"),                        # API key from .env
)

# ── Wrap the LLM with structured output ────────────────────────
structured_llm = llm.with_structured_output(ComposedResponse)   # forces LLM to return ComposedResponse


def response_composer(state: TicketState) -> dict:
    """
    LangGraph node function: composes a customer-facing response.

    Reads from state:
        - subject:        the ticket subject line
        - description:    the ticket body
        - summary:        the concise ticket summary
        - retrieved_docs: the relevant knowledge base documents

    Writes to state:
        - response: the final customer-facing response text
    """
    subject = state["subject"]                                  # get the ticket subject from state
    description = state["description"]                          # get the ticket description from state
    summary = state["summary"]                                  # get the ticket summary from state
    retrieved_docs = state["retrieved_docs"]                    # get the retrieved documents from state

    template = _load_prompt_template()                          # load the response composer prompt template

    formatted_docs = _format_retrieved_docs(retrieved_docs)     # format docs into readable context

    prompt = template.format(                                   # fill in all placeholders
        subject=subject,                                        # insert ticket subject
        description=description,                                # insert ticket description
        summary=summary,                                        # insert ticket summary
        retrieved_docs=formatted_docs,                          # insert formatted documents
    )

    print(f"\n✍️ [Response Composer] Drafting customer response...")  # log start

    result = structured_llm.invoke(prompt)                      # call the LLM to compose the response

    print(f"   ✅ Response composed ({len(result.response)} chars)")  # log response length
    print(f"   📝 Preview: {result.response[:150]}...")           # log first 150 chars as preview

    return {                                                    # return dict to update the state
        "response": result.response,                            # write the composed response to state
    }
