# query_analyzer node


from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from state import TicketState
from schemas import QueryAnalysis


llm = ChatMistralAI(
    model="mistral-small-latest" , 
    temperature=0,
)
structured_llm = llm.with_structured_output(QueryAnalysis)

system_prompt = (
    "You are analyzing a customer support ticket for Doxa." , 
     "Write a concise summary under 100 words, and extract 5 to 10 relevant keywords."
)
#\


def query_analyzer(state : TicketState) -> dict :
    subject = state["subject"]
    description = state["descreption"]

    prompt = f"{system_prompt}\n\n Subject : {subject} \ndescription : {description} "

    result = structured_llm.invoke(prompt)

    return{
        "summary": result.summary,
        "keywords": result.keywords,
    }

