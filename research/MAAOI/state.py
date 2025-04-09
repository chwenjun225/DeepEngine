from typing import TypedDict, Annotated



from langgraph.graph.message import (
	add_messages, 
	BaseMessage
)



class State(TypedDict):
	"""Lưu trữ trạng thái hội thoại giữa các agents."""
	VISION_AGENT_MSGS: 											Annotated[list[BaseMessage], add_messages] 	### YOLOv11
	TEMPORAL_PATTERN_AGENT_MSGS: 						Annotated[list[BaseMessage], add_messages]	### NON-LLM 
	DEFECT_REASONING_AGENT_MSGS: 						Annotated[list[BaseMessage], add_messages]	### LLM
	QUALITY_CONTROL_JUDGEMENT_AGENT_MSGS: 	Annotated[list[BaseMessage], add_messages]	### LLM 
	VISUAL_AGENT_MSGS: 											Annotated[list[BaseMessage], add_messages] 	### NON-LLM
