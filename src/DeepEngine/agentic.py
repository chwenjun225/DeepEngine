import re
import uuid
import random
import tqdm
import requests
import json
import json5
import fire 
import streamlit as st 
from PIL import Image
from io import BytesIO



from pydantic import BaseModel, Field, ValidationError
from typing_extensions import (Annotated, TypedDict, Sequence, Union, Optional, Literal, List, Dict, Iterator, Any)



from langchain_core.tools import InjectedToolCallId, BaseTool
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import (HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage)
from langchain_core.prompts import PromptTemplate



from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.managed import IsLastStep
from langgraph.graph import (MessagesState, StateGraph, START, END)
from langgraph.prebuilt import (create_react_agent, ToolNode, tools_condition)



from tools import (add, subtract, multiply, divide, power, square_root)
from prompts_lib import Prompts 



BEGIN_OF_TEXT		=	"<|begin_of_text|>"
END_OF_TEXT			= 	"<|end_of_text|>"
START_HEADER_ID		= 	"<|start_header_id|>"
END_HEADER_ID		= 	"<|end_header_id|>"
END_OF_MESSAGE_ID	= 	"<|eom_id|>"
END_OF_TURN_ID		= 	"<|eot_id|>"



MSG_TYPES = {SystemMessage: "SYSTEM", HumanMessage: "HUMAN", AIMessage: "AI"}



def add_unique_msgs(
		msgs: Dict[str, Dict[str, List[BaseMessage]]],
		agent_type: str, 
		msg: BaseMessage
	) -> None:
	"""Adds a message to the appropriate agent category if it does not already exist."""
	msg_type = MSG_TYPES.get(type(msg))
	if not msg_type or agent_type not in msgs: return
	if agent_type == "REQUEST_VERIFY":
		msgs[agent_type][msg_type] = [msg]
	elif msg.content not in {m.content for m in msgs[agent_type][msg_type]}:
		msgs[agent_type][msg_type].append(msg)



class State(BaseModel):
	"""Manages structured conversation state in a multi-agent system.

	Attributes:
		human_query (List[HumanMessage]): List of user queries.
		messages (Dict[str, Dict[str, List[BaseMessage]]]): 
			Stores categorized messages by agent type and message type:
				- Agents types: MANAGER_AGENT, REQUEST_VERIFY, PROMPT_AGENT, DATA_AGENT, MODEL_AGENT, OP_AGENT.
				- Message types: SYSTEM, HUMAN, AI.
		is_last_step (bool): Indicates if this is the final step.
		remaining_steps (int): Number of steps left.

	Methods:
		get_all_msgs(): Returns all messages in chronological order.
		get_latest_msg(agent_type, msg_type): Gets the latest message of a given agent and type.
		get_msgs_by_agent_type_and_msg_type(agent_type, msgs_type): Retrieves all messages from a specific agent and type.
	"""
	human_query: Annotated[List[HumanMessage], add_messages] = Field(default_factory=list)
	messages: Dict[str, Dict[str, Annotated[List[BaseMessage], add_unique_msgs]]] = Field(
		default_factory = lambda: {
			"MANAGER_AGENT"		: 	{	"SYSTEM": [], "HUMAN": [], "AI": []		}, 
			"REQUEST_VERIFY"	: 	{	"SYSTEM": [], "HUMAN": [], "AI": []		}, 
			"PROMPT_AGENT"		: 	{	"SYSTEM": [], "HUMAN": [], "AI": []		}, 
			"DATA_AGENT"		: 	{	"SYSTEM": [], "HUMAN": [], "AI": []		}, 
			"MODEL_AGENT"		: 	{	"SYSTEM": [], "HUMAN": [], "AI": []		}, 
			"OP_AGENT"			: 	{	"SYSTEM": [], "HUMAN": [], "AI": []		}
		}, 
		description="Categorized Multi-Agent messages: MANAGER_AGENT, REQUEST_VERIFY, PROMPT_AGENT, DATA_AGENT, MODEL_AGENT, OP_AGENT."
	)
	is_last_step: bool = False
	remaining_steps: int = 3

	def get_all_msgs(self) -> List[BaseMessage]:
		"""Returns all messages across all agents in chronological order."""
		all_messages = []
		for agent_messages in self.messages.values():
			for msg_list in agent_messages.values():
				all_messages.extend(msg_list)
		return all_messages

	def get_latest_msg(self, agent_type: str, msg_type: str) -> BaseMessage:
		"""Returns the latest message from a given agent category and message type."""
		if agent_type not in self.messages:
			raise ValueError(
				f"[ERROR]: Invalid agent category '{agent_type}'. Must be one of {list(self.messages.keys())}."
			)
		if msg_type not in self.messages[agent_type]:
			raise ValueError(
				f"[ERROR]: Invalid message type '{msg_type}'. Must be 'SYSTEM', 'HUMAN', or 'AI'."
			)
		return self.messages[agent_type][msg_type][-1] if self.messages[agent_type][msg_type] else None

	def get_msgs_by_agent_type_and_msg_type(self, agent_type: str, msgs_type: str) -> List[BaseMessage]:
		"""Returns all messages from a specific agent and type."""
		if agent_type not in self.messages:
			raise ValueError(
				f"[ERROR]: Invalid agent category '{agent_type}'. Must be one of {list(self.messages.keys())}."
			)
		if msgs_type not in self.messages[agent_type]:
			raise ValueError(
				f"[ERROR]: Invalid message type '{msgs_type}'. Must be 'SYSTEM', 'HUMAN', or 'AI'."
			)
		return self.messages[agent_type][msgs_type]


class PerformanceMetric(BaseModel):
	name: str = Field(..., description="accuracy")
	value: float = Field(..., description=0.98)

class Problem(BaseModel):
	area: str = Field(..., description="tabular data analysis")
	downstream_task: str = Field(..., description="tabular classification")
	application_domain: str = Field(..., description="agriculture")
	description: str = Field(..., description="""E.g,. Build a machine learning model, potentially XGBoost 
			or LightGBM, to classify banana quality as Good or Bad based on their numerical 
			information about bananas of different quality (size, weight, sweetness, softness, 
			harvest time, ripeness, and acidity). The model must achieve at least 0.98 accuracy.""")
	performance_metrics: List[PerformanceMetric]
	complexity_metrics: List[str] = []

class Dataset(BaseModel):
	name: str = Field(..., description="banana_quality")
	modality: List[str] = Field(..., description=["tabular"])
	target_variables: List[str] = Field(..., description=["quality"])
	specification: Optional[str] = None
	description: str = Field(..., description="""A dataset containing numerical information about bananas of different quality, including size, weight, sweetness, softness, harvest time, ripeness, and acidity.""")
	preprocessing: List[str] = []
	augmentation: List[str] = []
	visualization: List[str] = []
	source: str 

class Model(BaseModel):
	name: str 
	family: str 
	type: str 
	specification: Optional[str] = None
	description: str 

class HardwareRequirements(BaseModel):
	cuda: bool 
	cpu_cores: int 
	memory: str 

class User(BaseModel):
	intent: str 
	expertise: str 

class PromptParsingJSON(BaseModel):
	user: User
	problem: Problem
	dataset: List[Dataset]
	model: List[Model]



def validate_json(data):
	"""Validation parsed JSON output."""
	try:
		validated_data = PromptParsingJSON(**data)
		print(">>> JSON hợp lệ!")
		return validated_data
	except ValidationError as e:
		print(">>> JSON không hợp lệ:", e)
		return None



def enhance_human_query(state: State) -> str:
	"""Enhances the human query by formatting it with special tokens of LLama 3 series models.

	This function constructs a structured prompt including:
	- `sys_msg (from context)
	- `human_msg (latest human query)
	- `ai_msg (space for AI response)

	Args:
		state (State): The current conversation state.

	Returns:
		HumanMessage: A formatted human query wrapped with special tokens.

	Example:
		>>> state.user_query = [HumanMessage(content="What is AI?")]
		>>> enhance_human_query(state)
		HumanMessage(content='<|start_header_id|>HUMAN<|end_header_id|>What is AI?<|end_of_turn_id|><|start_header_id|>AI<|end_header_id|>')
	"""
	human_query = state.human_query[-1].content if state.human_query else ""
	formatted_query = (
		f"{START_HEADER_ID}HUMAN{END_HEADER_ID}"
		f"{human_query}"
		f"{END_OF_TURN_ID}"
		f"{START_HEADER_ID}AI{END_HEADER_ID}"
	)
	return formatted_query



def add_eotext_eoturn_to_ai_msg(
	ai_msg: AIMessage, 
	end_of_turn_id_token: str = END_OF_TURN_ID,
	end_of_text_token: str = END_OF_TEXT
) -> AIMessage:
	"""
	Ensures AIMessage content ends with required special tokens.

	This function appends `<|end_of_text|>` and `<|eot_id|>` at the end of 
	the message content if they are not already present.

	Args:
		ai_msg (AIMessage): The AI-generated message.
		end_of_text_token (str, optional) = "<|end_of_text|>": Special token indicating the end of the text.
		end_of_turn_id_token (str, optional) = "<|eot_id|>": Special token marking the end of a conversation turn.

	Returns:
		AIMessage: The updated AI message with the required tokens.

	Example:
		>>> message = AIMessage(content="Hello, how can I assist you?")
		>>> add_eotext_eoturn_to_ai_msg(message, "<|end_of_text|>", "<|eot_id|>")
		AIMessage(content="Hello, how can I assist you?<|end_of_text|><|eot_id|>")
	"""
	content = ai_msg.content.strip()
	if not content.endswith(end_of_turn_id_token):
		content += end_of_turn_id_token
	if not content.endswith(end_of_turn_id_token + end_of_text_token):
			content = content.replace(end_of_turn_id_token, end_of_turn_id_token + end_of_text_token)
	return AIMessage(content=content)



def build_react_sys_msg_prompt(tool_desc_prompt: str, react_prompt: str, tools: List[BaseTool])-> str:
	"""Builds a formatted system prompt with tool descriptions.

	Args:
		tool_desc_prompt (PromptTemplate): Template for tool descriptions.
		react_prompt (PromptTemplate): Template for constructing the final system prompt.
		tools (List[BaseTool]): List of tool objects.

	Returns:
		str: A fully formatted system prompt with tool descriptions.
	"""
	list_tool_desc = [
		tool_desc_prompt.format(
			name_for_model=(tool_info := getattr(tool.args_schema, "model_json_schema", lambda: {})()).get("title", "Unknown Tool"),
			name_for_human=tool_info.get("title", "Unknown Tool"),
			description_for_model=tool_info.get("description", "No description available."),
			type=tool_info.get("type", "N/A"),
			properties=json.dumps(tool_info.get("properties", {}), ensure_ascii=False),
			required=json.dumps(tool_info.get("required", []), ensure_ascii=False),
		) + " Format the arguments as a JSON object."
		for tool in tools
	]
	prompt = react_prompt.format(
		BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
		START_HEADER_ID=START_HEADER_ID, 
		END_HEADER_ID=END_HEADER_ID, 
		END_OF_TURN_ID=END_OF_TURN_ID, 
		tools_desc="\n\n".join(list_tool_desc), 
		tools_name=", ".join(tool.name for tool in tools),
	)
	return prompt



MGR_SYS_MSG_PROMPT = Prompts.AGENT_MANAGER_PROMPT
REQ_VER_MSG_PROMPT = Prompts.REQUEST_VERIFY_RELEVANCY



CONFIG = {"configurable": {"thread_id": str(uuid.uuid4())}}
CHECKPOINTER = MemorySaver()
STORE = InMemoryStore()



MODEL_HIGH_TEMP = ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0.8, num_predict=128_000)
MODEL_LOW_TEMP = ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0.1, num_predict=128_000)
MODEL_STRUCTURE_OUTPUT = MODEL_LOW_TEMP.with_structured_output(schema=HumanQueryParseJSON)



def data_agent(state: State) -> State:
	"""Data Agent."""
	human_msg = ""
	ai_msg = add_eotext_eoturn_to_ai_msg(
		ai_msg=ai_msg, 
		end_of_turn_id_token=END_OF_TURN_ID, 
		end_of_text_token=END_OF_TEXT 
	)
	return {"messages": {
		"DATA_AGENT": {
			"SYSTEM": [MGR_SYS_MSG_PROMPT], 
			"HUMAN": [human_msg], 
			"AI": [ai_msg]
		}}}



def model_agent(state: State) -> State:
	"""Model Agent."""
	sys_msg = ""
	human_msg = ""
	ai_msg = add_eotext_eoturn_to_ai_msg(
		ai_msg=ai_msg, 
		end_of_turn_id_token=END_OF_TURN_ID, 
		end_of_text_token=END_OF_TEXT,
	)
	return {"messages": {
		"MODEL_AGENT": {
			"SYSTEM": [sys_msg], 
			"HUMAN": [human_msg], 
			"AI": [ai_msg]
		}}}



def op_agent(state: State) -> State:
	"""Operation Agent."""
	human_msg = ""
	ai_msg = add_eotext_eoturn_to_ai_msg(
		ai_msg=ai_msg, 
		end_of_turn_id_token=END_OF_TURN_ID, 
		end_of_text_token=END_OF_TEXT,
	)
	return {"messages": {
		"OP_AGENT": {
			"SYSTEM": [MGR_SYS_MSG_PROMPT], 
			"HUMAN": [human_msg], 
			"AI": [ai_msg]
		}}}



def manager_agent(state: State) -> State:
	"""Manager Agent.

	Example:
		>>> Human query: I need a very accurate model 
				to classify images in the Butterfly Image 
				Classification dataset into their respective 
				categories. 
				The dataset has been uploaded with its label 
				information in the labels.csv file.
		>>> AI response: ...
	"""
	sys_msg = SystemMessage(content=MGR_SYS_MSG_PROMPT.format(
		BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
		START_HEADER_ID=START_HEADER_ID, 
		END_HEADER_ID=END_HEADER_ID, 
		END_OF_TURN_ID=END_OF_TURN_ID 
	))
	human_msg = HumanMessage(
		content=enhance_human_query(state=state)
	)
	human_msg_json = MODEL_STRUCTURE_OUTPUT.invoke([human_msg]) # TODO: biến input đầu vào của user thành chuỗi JSON
	ai_msg = MODEL_LOW_TEMP.invoke([
		sys_msg, 
		human_msg
	])
	if not isinstance(ai_msg, AIMessage):
		ai_msg = AIMessage(
			content=ai_msg.strip() 
			if isinstance(ai_msg, str) 
			else "At node_manager_agent, I'm unable to generate a response."
		)
	ai_msg = add_eotext_eoturn_to_ai_msg(
		ai_msg=ai_msg, 
		end_of_turn_id_token=END_OF_TURN_ID, 
		end_of_text_token=END_OF_TEXT,
	)
	return {"messages": {
		"MANAGER_AGENT": {
			"SYSTEM": [sys_msg], 
			"HUMAN": [human_msg], 
			"AI": [ai_msg]
		}}}



def check_contain_yes_or_no(ai_msg: str) -> str:
	"""Checks if the AI response contains 'Yes' or 'No'."""
	match = re.search(r"<\|end_header_id\|>\s*(Yes|No)\s*<\|eot_id\|>", ai_msg, re.IGNORECASE)
	return match.group(1) if match else f"[ERROR]: Can't found ('Yes' or 'No') in this {ai_msg}"



def request_verify(state: State):
	"""Request verification output of Agent Manager."""
	human_msg = state.human_query[-1].content
	sys_msg = SystemMessage(content=REQ_VER_MSG_PROMPT.format(
		instruction=human_msg, 
		begin_of_text=BEGIN_OF_TEXT, 
		start_header_id=START_HEADER_ID, 
		end_header_id=END_HEADER_ID, 
		end_of_turn_id=END_OF_TURN_ID
	))
	ai_msg = MODEL_LOW_TEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage):
		ai_msg = AIMessage(
			content=ai_msg.strip() 
			if isinstance(ai_msg, str) 
			else "At node_request_verify, I'm unable to generate a response."
		)
	ai_msg = add_eotext_eoturn_to_ai_msg(
		ai_msg=ai_msg, 
		end_of_turn_id_token=END_OF_TURN_ID, 
		end_of_text_token=END_OF_TEXT
	)
	ai_msg_yes_or_no = AIMessage(check_contain_yes_or_no(ai_msg=ai_msg.content))
	return {"messages": {
		"REQUEST_VERIFY": {
			"SYSTEM": [sys_msg], 
			"HUMAN": [human_msg], 
			"AI": [ai_msg_yes_or_no] 
		}}}



def req_ver_yes_or_no(state: State):
	"""Determines the next step based on the AI response from REQUEST_VERIFY."""
	resp_map = {
		"YES": "PROMPT_AGENT", 
		"NO": END
	}
	ai_msg = state.get_latest_msg(
		agent_type="REQUEST_VERIFY", 
		msg_type="AI"
	)
	if not ai_msg or not hasattr(ai_msg, "content"):
		raise ValueError("[ERROR]: No valid AI message found in REQUEST_VERIFY.")
	resp = ai_msg.content.strip().upper()
	return resp_map.get(resp, ValueError(f"[ERROR]: Unexpected response '{resp}'"))



def prompt_agent(state: State) -> State:
	"""Prompt Agent."""
	human_msg = ""
	ai_msg = ""
	return {"messages": {
		"PROMPT_AGENT": {
			"SYSTEM": [MGR_SYS_MSG_PROMPT], 
			"HUMAN": [human_msg], 
			"AI": [ai_msg]
		}}}



workflow = StateGraph(State)

workflow.add_node("MANAGER_AGENT", manager_agent)
workflow.add_node("REQUEST_VERIFY", request_verify)
workflow.add_node("PROMPT_AGENT", prompt_agent)
workflow.add_node("DATA_AGENT", data_agent)
workflow.add_node("MODEL_AGENT", model_agent)
workflow.add_node("OP_AGENT", op_agent)

workflow.add_edge(START, "MANAGER_AGENT")
workflow.add_edge("MANAGER_AGENT", "REQUEST_VERIFY")
workflow.add_conditional_edges("REQUEST_VERIFY", req_ver_yes_or_no)
workflow.add_edge("PROMPT_AGENT", "REQUEST_VERIFY")

app = workflow.compile(
	checkpointer=CHECKPOINTER, 
	store=STORE, debug=True, 
	name="FOXCONN-AI Research"
)



def main() -> None:
	"""Handles user queries and displays AI responses."""
	for user_query in QUERIES:
		user_query = user_query.strip().lower()
		if user_query == "exit":
			print(">>> [System Exit] Goodbye! Have a great day! 😊")
			break
		streamlit_user_interface(
			app.stream(
				input={"human_query": [user_query]}, 
				stream_mode="values", 
				config=CONFIG
			))



def streamlit_user_interface(stream: Iterator[Dict[str, Dict[str, Dict[str, List[BaseMessage]]]] | Any]) -> None:
	"""Hiển thị kết quả hội thoại trên Streamlit."""
	st.title("FOXCONN-AI Research")
	for s in stream:
		if len(list(s.keys())) == 2:
			msgs = s["messages"]
			for agent, messages in msgs.items():
				with st.expander(f"🔹 {agent}", expanded=False):
					for msg_type, msg_list in messages.items():
						if msg_list: 
							st.subheader(f"{msg_type} Messages")
							for msg in msg_list:
								st.markdown(f"```{msg.content}```")



QUERIES = [
	"""I need a highly accurate machine learning model developed to classify images within the Butterfly Image Classification dataset into their correct species categories. 
The dataset has been uploaded with its label information in the labels.csv file. 
Please use a convolutional neural network (CNN) architecture for this task, leveraging transfer learning from a pre-trained ResNet-50 model to improve accuracy. 
Optimize the model using cross-validation on the training split to fine-tune hyperparameters, and aim for an accuracy of at least 0.95 on the test split. 
Provide the final trained model, a detailed report of the training process, hyperparameter settings, accuracy metrics, and a confusion matrix to evaluate performance across different categories.""",
	
	"""Please provide a classification model that categorizes images into one of four clothing categories. 
The image path, along with its label information, can be found in the files train labels.csv and test labels.csv. 
The model should achieve at least 85% accuracy on the test set and be implemented using PyTorch. 
Additionally, please include data augmentation techniques and a confusion matrix in the evaluation."""	
	
	"""Hello, What is heavier a kilo of feathers or a kilo of steel?""", 
	
	"""exit"""
]



if __name__ == "__main__":
	fire.Fire(main)
