# ═══════════════════════════════════════════════════════════════════
# query_analyzer.py – Node 1: Analyzes the support ticket
# ═══════════════════════════════════════════════════════════════════
# This node receives a raw customer ticket (subject + description)
# and uses a Mistral LLM with structured output to produce:
#   - A concise summary (under 100 words)
#   - A list of 5-10 relevant keywords for knowledge base search
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
from schemas import QueryAnalysis                               # Pydantic schema for structured output
from config import LLM_MODEL_NAME, LLM_TEMPERATURE              # model config values
from config import PROMPTS_DIR                                  # path to prompt templates


def _load_prompt_template():
    """
    Load the query analyzer prompt template from the prompts/ folder.
    Returns the template string with {subject} and {description} placeholders.
    """
    prompt_path = PROMPTS_DIR / "query_analyzer.txt"            # build path to the prompt file
    with open(prompt_path, "r", encoding="utf-8") as f:         # open the file in read mode
        template = f.read()                                     # read the entire file content
    return template                                             # return the template string


# ── Initialize the Mistral LLM ─────────────────────────────────
llm = ChatMistralAI(                                            # create the Mistral chat model
    model=LLM_MODEL_NAME,                                       # e.g. "mistral-small-latest"
    temperature=LLM_TEMPERATURE,                                # temperature=0 for deterministic output
    api_key=os.getenv("MISTRAL_API_KEY"),                        # API key from .env
)

# ── Wrap the LLM with structured output ────────────────────────
structured_llm = llm.with_structured_output(QueryAnalysis)      # forces LLM to return QueryAnalysis schema


def query_analyzer(state: TicketState) -> dict:
    """
    LangGraph node function: analyzes the customer support ticket.

    Reads from state:
        - subject:     the ticket subject line
        - description: the ticket body/description

    Writes to state:
        - summary:  concise summary of the ticket
        - keywords: list of relevant keywords
    """
    subject = state["subject"]                                  # extract the ticket subject from state
    description = state["description"]                          # extract the ticket description from state

    template = _load_prompt_template()                          # load the prompt template from file

    prompt = template.format(                                   # fill in the placeholders in the template
        subject=subject,                                        # insert the ticket subject
        description=description,                                # insert the ticket description
    )

    print(f"\n🔍 [Query Analyzer] Analyzing ticket...")         # log that analysis is starting
    print(f"   Subject: {subject}")                             # log the subject being analyzed

    result = structured_llm.invoke(prompt)                      # call the LLM and get structured output

    print(f"   ✅ Summary: {result.summary[:80]}...")            # log first 80 chars of summary
    print(f"   ✅ Keywords: {result.keywords}")                  # log extracted keywords

    return {                                                    # return a dict to update the state
        "summary": result.summary,                              # write the summary to state
        "keywords": result.keywords,                            # write the keywords to state
    }
