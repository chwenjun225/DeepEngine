from collections import defaultdict
from typing_extensions import (Annotated, TypedDict, List, Dict)



from langchain_core.messages import (HumanMessage, AIMessage, SystemMessage, BaseMessage)



MSG_TYPES = {SystemMessage: "SYS", HumanMessage: "HUMAN", AIMessage: "AI"}
DEFAULT_AGENTS: Dict[str, Dict[str, List[BaseMessage]]] = {
	agent: {"SYS": [], "HUMAN": [], "AI": []} for agent in [
		"SYSTEM_AGENT", "REASONING_AGENT", "RESEARCH_AGENT", "PLANNING_AGENT", 
		"EXECUTION_AGENT", "EVALUATION_AGENT", "DEBUGGING_AGENT"
	]
}



def default_messages() -> Dict[str, Dict[str, List[BaseMessage]]]:
	"""Trả về dictionary mặc định chứa danh sách tin nhắn theo agent và loại tin nhắn."""
	return defaultdict(lambda: {"SYS": [], "HUMAN": [], "AI": []}, DEFAULT_AGENTS.copy())



class State(TypedDict):
	"""Lưu trạng thái hội thoại trong hệ thống đa agent."""
	user_query: Annotated[List[HumanMessage], "The list of user query."]
	messages: Annotated[Dict[str, Dict[str, List[BaseMessage]]], "The state storage messages of multi-agents."] 
