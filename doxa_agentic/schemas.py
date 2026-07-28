# Pydantic models for structured LLM outputs
from pydantic import Field , BaseModel 

class QueryAnalysis(BaseModel):
    summary : str = Field(description="A concise summary of the ticket, under 100 words")
    keywords :list[str] = Field(description="5 to 10 relevant keywords extracted from the ticket")
class RetrievedDoc(BaseModel):
    doc_id: str
    snippet: str
    score: float