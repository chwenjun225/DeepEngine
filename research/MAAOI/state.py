from typing import TypedDict, Annotated



from langgraph.graph.message import (add_messages, BaseMessage)



class State(TypedDict):
	"""Lưu trữ trạng thái hội thoại giữa các agents."""
	VISION_AGENT_MSGS				: Annotated[list[BaseMessage], add_messages]
	TEMPORAL_PATTERN_AGENT_MSGS		: Annotated[list[BaseMessage], add_messages]
	DEFECT_REASONING_AGENT_MSGS		: Annotated[list[BaseMessage], add_messages]
	CRITICAL_ASSESSMENT_AGENT_MSGS	: Annotated[list[BaseMessage], add_messages]
	REPORT_GENERATOR_AGENT_MSGS		: Annotated[list[BaseMessage], add_messages]
	VISUAL_AGENT_MSGS				: Annotated[list[BaseMessage], add_messages]
