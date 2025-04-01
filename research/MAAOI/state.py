from typing import TypedDict, Annotated



from langgraph.graph.message import add_messages, BaseMessage



class State(TypedDict):
    """Lưu trữ trạng thái hội thoại giữa các agents."""
    messages: Annotated[list[BaseMessage], add_messages]
