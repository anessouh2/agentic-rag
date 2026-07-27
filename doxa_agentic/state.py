# shared TicketState definition
from typing import TypedDict , Annotated , Literal , Optional , List

class TicketState(TypedDict):
#this are the input 
    ticket_id   : str
    subject     : str
    descreption : str

  #this states will be the output of the query analyzer
    summary : str
    keywords : List[str]


