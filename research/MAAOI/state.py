# from collections import defaultdict
from typing_extensions import (Annotated, TypedDict, List)



from langgraph.graph.message import add_messages 



# DEFAULT_AGENTS: Dict[str, List[Dict]] = {
# 	agent: [] for agent in [
# 		"SYSTEM_AGENT"			,
# 		"ORCHESTRATE_AGENTS"	,
# 		"REASONING_AGENT"		,
# 		"RESEARCH_AGENT"		,
# 		"PLANNING_AGENT"		,
# 		"EXECUTION_AGENT"		,
# 		"COMMUNICATION_AGEN"	,
# 		"EVALUATION_AGENT"		,
# 		"DEBUGGING_AGENT"		, 
# 	]}



# def default_messages() -> Dict[str, List[Dict]]:
# 	"""Trả về dictionary mặc định chứa danh sách tin nhắn theo agent."""
# 	return defaultdict(lambda: [], DEFAULT_AGENTS.copy())



# class State(TypedDict):
# 	user_query: Annotated[Dict, "The current user query."]
# 	messages: Annotated[Dict[str, List[Dict]], "The state storage messages of multi-agents."]


class State(TypedDict):
	messages: Annotated[List, add_messages, "Lưu trữ ngữ cảnh trò chuyện cho các Agents."]
