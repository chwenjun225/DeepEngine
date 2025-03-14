from collections import defaultdict
from pydantic import BaseModel, Field, ValidationError, TypeAdapter
from typing_extensions import Annotated, TypedDict, Sequence, Union, \
	Optional, Literal, List, Dict, Iterator, Any, Type


from langgraph.graph.message import add_messages



from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage



MSG_TYPES = {SystemMessage: "SYS", HumanMessage: "HUMAN", AIMessage: "AI"}



DEFAULT_AGENTS: Dict[str, Dict[str, List[BaseMessage]]] = {
	agent: {"SYS": [], "HUMAN": [], "AI": []} for agent in [
		"MANAGER_AGENT", "REQUEST_VERIFY", "PROMPT_AGENT", 
		"RAP", "DATA_AGENT", "MODEL_AGENT", "OP_AGENT"
	]
}



def default_messages() -> Dict[str, Dict[str, List[BaseMessage]]]:
	"""Trả về dictionary mặc định chứa danh sách tin nhắn theo agent và loại tin nhắn."""
	return defaultdict(lambda: {"SYS": [], "HUMAN": [], "AI": []}, DEFAULT_AGENTS.copy())



class State(TypedDict):
	"""Lưu trạng thái hội thoại trong hệ thống đa agent."""
	human_query: List[HumanMessage]  
	messages: Dict[str, Dict[str, List[BaseMessage]]] 



class ReAct(TypedDict):
	user_query: Annotated[str, ..., "The original question provided by the user."]
	thought: str 
	action: Annotated[str, ..., "The action to be taken, chosen from available tools: {tools_name}."]
	action_input: Annotated[str, ..., "The required input for the action."]
	observation: Optional[str]
	final_answer: Annotated[str, ..., "The final answer to the original input question."]



class Conversation(TypedDict):
	"""You are an AI assistant. Respond in a conversational manner. Be kind and helpful."""
	response:		Annotated[str, ..., "A conversational response to the user's query"			]
	justification: 	Annotated[str, ..., "A brief explanation or reasoning behind the response."	]



class Prompt2JSON(TypedDict):
	"""Parses user requirements related to AI project potential into structured JSON."""
	problem_area: 	Annotated[str, ..., "Problem domain (e.g., tabular data analysis)."			]
	task: 			Annotated[str, ..., "Type of ML task (e.g., classification, regression)."	]
	application: 	Annotated[str, ..., "Application field (e.g., agriculture, healthcare)."	]
	dataset_name: 	Annotated[str, ..., "Dataset name (e.g., banana_quality)."					]
	data_modality: 	Annotated[List[str], ..., "Data modality (e.g., ['tabular', 'image'])."		]
	model_name: 	Annotated[str, ..., "Model name (e.g., XGBoost, ResNet)."					]
	model_type: 	Annotated[str, ..., "Model type (e.g., vision, text, tabular)."				]
	cuda: 			Annotated[bool, ..., "Requires CUDA? (True/False)."							]
	vram: 			Annotated[str, ..., "GPU's VRAM required (e.g., '6GB')."					]
	cpu_cores: 		Annotated[int, ..., "Number of CPU cores required."							]
	ram: 			Annotated[str, ..., "RAM required (e.g., '16GB')."							]
