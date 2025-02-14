from typing_extensions import (Annotated, TypedDict, Literal)
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class State(TypedDict):
	messages: Annotated[list, add_messages]
	user_query: str
	domain: Literal["records", "insurance"]
	documents: list[Document]
	answer: str

class Input(TypedDict):
	user_query: str

class Output(TypedDict):
	documents: list[Document]
	answer: str