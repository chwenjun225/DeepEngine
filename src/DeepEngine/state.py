from pydantic import BaseModel, Field, ValidationError, TypeAdapter
from collections import defaultdict
from typing_extensions import (Annotated, TypedDict, Sequence, Union, Optional, Literal, List, Dict, Iterator, Any, Type)



from langgraph.graph.message import add_messages



from langchain_core.messages import (HumanMessage, AIMessage, SystemMessage, BaseMessage)



MSG_TYPES = {SystemMessage: "SYS", HumanMessage: "HUMAN", AIMessage: "AI"}



DEFAULT_AGENTS: Dict[str, Dict[str, List[BaseMessage]]] = {
	"MANAGER_AGENT": 	{	"SYS": [], "HUMAN": [], "AI": []	},
	"REQUEST_VERIFY": 	{	"SYS": [], "HUMAN": [], "AI": []	},
	"PROMPT_AGENT": 	{	"SYS": [], "HUMAN": [], "AI": []	},
	"RAP": 				{	"SYS": [], "HUMAN": [], "AI": []	},
	"DATA_AGENT": 		{	"SYS": [], "HUMAN": [], "AI": []	},
	"MODEL_AGENT": 		{	"SYS": [], "HUMAN": [], "AI": []	},
	"OP_AGENT": 		{	"SYS": [], "HUMAN": [], "AI": []	},
}



def default_messages() -> Dict[str, Dict[str, List[BaseMessage]]]:
	"""Tạo dictionary mặc định cho `messages`, giữ nguyên danh sách các Agent.

	- Sử dụng `defaultdict` để tránh lỗi KeyError nếu truy cập Agent chưa tồn tại.
	- `lambda: {"SYS": [], "HUMAN": [], "AI": []}` đảm bảo mỗi Agent có đủ 3 loại tin nhắn.
	- `DEFAULT_AGENTS.copy()` giúp giữ nguyên cấu trúc ban đầu mà không bị ghi đè.

	Returns:
		Dict[str, Dict[str, List[BaseMessage]]]: Cấu trúc lưu trữ tin nhắn theo Agent và loại tin nhắn.
	"""
	return defaultdict(lambda: {"SYS": [], "HUMAN": [], "AI": []}, DEFAULT_AGENTS.copy())



class State(BaseModel):
	"""Manages structured conversation state in a multi-agent system.

	Attributes:
		human_query (List[HumanMessage]): List of user queries.
		messages (Dict[str, Dict[str, List[BaseMessage]]]): 
			Stores categorized messages by agent type and message type:
			- **Agent types**: MANAGER_AGENT, REQUEST_VERIFY, PROMPT_AGENT, DATA_AGENT, MODEL_AGENT, OP_AGENT.
			- **Message types**: SYSTEM, HUMAN, AI.
		is_last_step (bool): Indicates if this is the final step.
		remaining_steps (int): Number of steps left.

	Methods:
		get_all_msgs(): Retrieves all messages across all agents in chronological order.
		get_latest_msg(agent_type, msg_type): Returns the latest message from a given agent and type.
		get_msgs_by_agent_type_and_msg_type(agent_type, msg_type): Retrieves all messages from a specific agent and type.
		add_unique_msgs(node, msgs_type, msg): Adds a unique message to a specified node.
	"""
	human_query: Annotated[List[HumanMessage], add_messages] = Field(default_factory=list)
	messages: Dict[str, Dict[str, List[BaseMessage]]] = Field(
		default_factory=default_messages, 
		description="Categorized messages by agent type and message type."
	)
	is_last_step: bool = False
	remaining_steps: int = 3

	def get_all_msgs(self) -> List[BaseMessage]:
		"""Retrieves all messages from all agents in chronological order."""
		all_messages = []
		for agent_messages in self.messages.values():
			for msg_list in agent_messages.values():
				all_messages.extend(msg_list)
		return all_messages
		
	def get_latest_msg(self, agent_type: str, msg_type: str) -> BaseMessage:
		"""Returns the latest message from a specified agent and type.

		Raises:
			ValueError: If the agent type or message type is invalid.
		"""
		if agent_type not in self.messages: raise ValueError(f"[ERROR]: Invalid agent category '{agent_type}'. Must be one of {list(self.messages.keys())}.")
		if msg_type not in self.messages[agent_type]: raise ValueError(f"[ERROR]: Invalid message type '{msg_type}'. Must be 'SYSTEM', 'HUMAN', or 'AI'.")
		return self.messages[agent_type][msg_type][-1] if self.messages[agent_type][msg_type] else None

	def get_msgs_by_node_and_msgs_type(self, node: str, msgs_type: str) -> List[BaseMessage]:
		"""Retrieves all messages from a specified agent and type.

		Raises:
			ValueError: If the node or message type is invalid.
		"""
		if node not in self.messages: raise ValueError(f"[ERROR]: Invalid agent category '{node}'. Must be one of {list(self.messages.keys())}.")
		if msgs_type not in self.messages[node]: raise ValueError(f"[ERROR]: Invalid message type '{msgs_type}'. Must be 'SYSTEM', 'HUMAN', or 'AI'.")
		return self.messages[node][msgs_type]

	def add_unique_msgs(self, node: str, msgs_type: str, msg: BaseMessage) -> None:
		"""Adds a unique message to a specific node in the State.
		
		Args:
			node (str): The agent node (e.g., "MANAGER_AGENT", "REQUEST_VERIFY").
			msgs_type (str): The message type (one of "AI", "HUMAN", "SYS").
			msg (BaseMessage): The message object to be stored.

		Raises:
			ValueError: If the message type is invalid.
		"""
		if node not in self.messages: 
			self.messages[node] = {"SYS": [], "HUMAN": [], "AI": []}
		if msgs_type not in {"SYS", "HUMAN", "AI"}: 
			raise ValueError(f"[ERROR]: Invalid message type '{msgs_type}'. Must be 'SYS', 'HUMAN', or 'AI'.")
		if node == "REQUEST_VERIFY": 
			self.messages[node][msgs_type] = [msg]
		else:
			if msg.content not in {m.content for m in self.messages[node][msgs_type]}:
				self.messages[node][msgs_type].append(msg)
