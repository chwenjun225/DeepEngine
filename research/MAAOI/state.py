from typing_extensions import (Annotated, TypedDict)



from langgraph.graph.message import add_messages 



class State(TypedDict):
	"""Lưu trữ ngữ cảnh trò chuyện."""
	messages: Annotated[list[dict], add_messages]
