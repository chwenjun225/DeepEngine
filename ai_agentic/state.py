from langgraph.graph import add_messages 
from langchain_core.messages import AnyMessage
from typing_extensions import (TypedDict, Annotated)


class State(TypedDict):
	"""Respond to the user in this format."""
	messages: Annotated[list[AnyMessage], add_messages]