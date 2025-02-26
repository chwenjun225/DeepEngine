from pydantic import BaseModel 
from typing_extensions import (Annotated, TypedDict, Literal)
from langchain_core.documents import Document
from langgraph.graph.message import add_messages, BaseMessage

class State(TypedDict):
	messages: Annotated[list[BaseMessage], add_messages]
	selected_tools: list[str]

class SupervisorDecision(BaseModel):
	next: Literal["researcher", "coder", "mechanic_engineer", "FINISH"]

class AnswerWithJustification(BaseModel):
	answer: str
	justification: str

class Input(TypedDict):
	user_query: str

class Output(TypedDict):
	documents: list[Document]
	answer: str
