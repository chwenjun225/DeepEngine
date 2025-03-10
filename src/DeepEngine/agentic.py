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



from pydantic import BaseModel, Field, ValidationError, TypeAdapter
from typing_extensions import (Annotated, TypedDict, Sequence, Union, Optional, Literal, List, Dict, Iterator, Any, Type)



from langchain_core.output_parsers import PydanticOutputParser
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
from prompts import Prompts 



DEBUG = True



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
	if not msg_type or agent_type not in msgs: 
		return
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
			raise ValueError(f"[ERROR]: Invalid agent category '{agent_type}'. Must be one of {list(self.messages.keys())}.")
		if msg_type not in self.messages[agent_type]:
			raise ValueError(f"[ERROR]: Invalid message type '{msg_type}'. Must be 'SYSTEM', 'HUMAN', or 'AI'.")
		return self.messages[agent_type][msg_type][-1] if self.messages[agent_type][msg_type] else None

	def get_msgs_by_agent_type_and_msg_type(self, agent_type: str, msgs_type: str) -> List[BaseMessage]:
		"""Returns all messages from a specific agent and type."""
		if agent_type not in self.messages:
			raise ValueError(f"[ERROR]: Invalid agent category '{agent_type}'. Must be one of {list(self.messages.keys())}.")
		if msgs_type not in self.messages[agent_type]:
			raise ValueError(f"[ERROR]: Invalid message type '{msgs_type}'. Must be 'SYSTEM', 'HUMAN', or 'AI'.")
		return self.messages[agent_type][msgs_type]



class ConversationalResponse(TypedDict):
	"""Respond in a conversational manner. Be kind and helpful."""
	response: str = Field(description="A conversational response to the user's query")



class UserRequirementsToJSON(TypedDict):
	"""Parses user requirements related to AI project potential into structured JSON."""
	problem_area: Annotated[str, ..., "Problem domain (e.g., tabular data analysis)."]
	task: Annotated[str, ..., "Type of ML task (e.g., classification, regression)."]
	application: Annotated[str, ..., "Application field (e.g., agriculture, healthcare)."]
	dataset_name: Annotated[str, ..., "Dataset name (e.g., banana_quality)."]
	data_modality: Annotated[List[str], ..., "Data modality (e.g., ['tabular', 'image'])."]
	model_name: Annotated[str, ..., "Model name (e.g., XGBoost, ResNet)."]
	model_type: Annotated[str, ..., "Model type (e.g., vision, text, tabular)."]
	hardware_cuda: Annotated[bool, ..., "Requires CUDA? (True/False)."]
	hardware_cpu_cores: Annotated[int, ..., "Number of CPU cores required."]
	hardware_memory: Annotated[str, ..., "RAM required (e.g., '32GB')."]



class TheFinalAnswer(TypedDict):
	"""Final answer of LLM response to user."""
	final_output: Union[UserRequirementsToJSON, ConversationalResponse]



MGR_SYS_MSG_PROMPT = Prompts.AGENT_MANAGER_PROMPT
REQ_VER_RELEVANCY_MSG_PROMPT = Prompts.REQUEST_VERIFY_RELEVANCY
REQ_VER_ADEQUACY_MSG_PROMPT = Prompts.REQUEST_VERIFY_ADEQUACY
PAR_JSON_MSG_PROMPT = Prompts.PROMPT_PARSE_JSON_AGENT_PROMPT



CONFIG = {"configurable": {"thread_id": str(uuid.uuid4())}}
CHECKPOINTER = MemorySaver()
STORE = InMemoryStore()



MODEL_HIGH_TEMP = ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0.8, num_predict=100_000)
MODEL_LOW_TEMP = ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0, num_predict=100_000)
MODEL_STRUCTURE_OUTPUT = MODEL_LOW_TEMP.with_structured_output(UserRequirementsToJSON, method="json_schema")



def enhance_human_query(human_msg: str) -> str:
	"""Enhances the human query by formatting it with special tokens of LLama 3 series models.

	This function constructs a structured prompt including:
	- `sys_msg (from context)
	- `human_msg (latest human query)
	- `ai_msg (space for AI response)

	Args:
		human_msg (str): User's query.

	Returns:
		formatted_query: A formatted human query wrapped with special tokens.

	Example:
		>>> state.user_query = [HumanMessage(content="What is AI?")]
		>>> enhance_human_query(state)
		formatted_query="<|start_header_id|>HUMAN<|end_header_id|>What is AI?<|end_of_turn_id|><|start_header_id|>AI<|end_header_id|>"
	"""
	formatted_query = (
		f"{START_HEADER_ID}HUMAN{END_HEADER_ID}"
		f"{human_msg}{END_OF_TURN_ID}"
		f"{START_HEADER_ID}AI{END_HEADER_ID}"
	)
	return formatted_query



def add_eotext_eoturn_to_ai_msg(
	ai_msg: AIMessage, 
	end_of_turn_id_token: str = END_OF_TURN_ID,
	end_of_text_token: str = END_OF_TEXT
) -> AIMessage:
	"""Ensures AIMessage content ends with required special tokens.

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



def model_parse_json(human_msg: HumanMessage, schema: UserRequirementsToJSON) -> json:
	"""LLM tạo JSON đúng theo định dạng Pydantic.

	Args:
		human_msg (HumanMessage): Tin nhắn đầu vào của người dùng.
		schema (Type[BaseModel]): Pydantic model dùng để kiểm tra JSON.

	Returns:
		dict: JSON đã được kiểm tra và xác nhận hợp lệ.
	"""
	json_data = MODEL_STRUCTURE_OUTPUT.invoke([human_msg])
	if not json_data:
		json_schema = str(TypeAdapter(UserRequirementsToJSON).json_schema()["properties"])
		sys_msg = SystemMessage(content=PAR_JSON_MSG_PROMPT.format(
			BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
			START_HEADER_ID=START_HEADER_ID, 
			END_HEADER_ID=END_HEADER_ID, 
			json_specification=json_schema, 
			human_msg=human_msg.content, 
			END_OF_TURN_ID=END_OF_TURN_ID
		))
		ai_msg_json = MODEL_LOW_TEMP.invoke([sys_msg])
		pattern = r"```json\n(.*?)\n```"
		match = re.search(pattern=pattern, string=ai_msg_json.content, flags=re.DOTALL)
		if not match:
			raise ValueError(">>> Không tìm thấy JSON hợp lệ trong phản hồi của mô hình.")
		json_string = match.group(1).strip()
		try:
			json_data = json.loads(json_string)
			if DEBUG: 
				print(">>> JSON hợp lệ:")
				print(json.dumps(json_data, indent=2, ensure_ascii=False))
		except json.JSONDecodeError as e:
			raise ValueError(f">>> JSON không hợp lệ (DecodeError): {e}")
	return json_data 



def manager_agent(state: State) -> State:
	"""Manager Agent.

	Example:
		>>> Human query: I need a very accurate model to classify images in the 
				Butterfly Image Classification dataset into their respective 
				categories. The dataset has been uploaded with its label 
				information in the labels.csv file.
		>>> AI response: ...
	"""
	sys_msg = SystemMessage(content=MGR_SYS_MSG_PROMPT.format(
		BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
		START_HEADER_ID=START_HEADER_ID, 
		END_HEADER_ID=END_HEADER_ID, 
		END_OF_TURN_ID=END_OF_TURN_ID 
	))
	human_msg = HumanMessage(content=enhance_human_query(
			human_msg=state.human_query[-1].content if state.human_query else ""
		))
	ai_msg_json = HumanMessage(str(model_parse_json(human_msg=human_msg, schema=UserRequirementsToJSON)))
	ai_msg = MODEL_LOW_TEMP.invoke([sys_msg, human_msg, ai_msg_json])
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
			"AI": [AIMessage("<|parse_json|>" + ai_msg_json.content + "<|end_parse_json|>" + ai_msg.content)]
		}}}



def check_contain_yes_or_no(ai_msg: str) -> str:
	"""Checks if the AI response contains 'Yes' or 'No'."""
	pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\s*(yes|no)\b"
	match = re.search(pattern=pattern, string=ai_msg, flags=re.IGNORECASE)
	if match:
		return match.group(1).upper()
	else:
		return "[ERROR]: Không tìm thấy 'Yes' hoặc 'No' trong phản hồi AI!"



def req_ver_relevancy(state: State) -> List[BaseMessage]:
	"""Check request verification of human_query."""
	human_msg = state.human_query[-1]
	sys_msg = SystemMessage(content=REQ_VER_RELEVANCY_MSG_PROMPT.format(
		instruction=human_msg.content, 
		BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
		START_HEADER_ID=START_HEADER_ID, 
		END_HEADER_ID=END_HEADER_ID, 
		END_OF_TURN_ID=END_OF_TURN_ID
	))
	ai_msg = MODEL_LOW_TEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage):
		ai_msg = AIMessage(content=ai_msg.strip() if isinstance(ai_msg, str) else "At node_request_verify-REQUEST_VERIFY_RELEVANCY, I'm unable to generate a response.")
	ai_msg = add_eotext_eoturn_to_ai_msg(
		ai_msg=ai_msg, 
		end_of_turn_id_token=END_OF_TURN_ID, 
		end_of_text_token=END_OF_TEXT
	)
	return [sys_msg, human_msg, ai_msg]



def req_ver_adequacy(state: State) -> List[BaseMessage]:
	"""Check request verification of AIMessage response with JSON object."""
	pattern = r"<\|parse_json\|>(.*?)<\|end_parse_json\|>"
	human_msg = state.messages['MANAGER_AGENT']['HUMAN'][-1]
	ai_msg = state.messages['MANAGER_AGENT']['AI'][-1]
	json_obj_from_ai_msg = re.findall(pattern=pattern, string=ai_msg.content, flags=re.DOTALL)[-1]
	sys_msg = SystemMessage(content=REQ_VER_ADEQUACY_MSG_PROMPT.format(
		BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
		START_HEADER_ID=START_HEADER_ID, 
		END_HEADER_ID=END_HEADER_ID, 
		parsed_user_requirements=json_obj_from_ai_msg, 
		END_OF_TURN_ID=END_OF_TURN_ID
	))
	ai_msg = MODEL_LOW_TEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage):
		ai_msg = AIMessage(content=ai_msg.strip() if isinstance(ai_msg, str) else "At node_request_verify-REQUEST_VERIFY_ADEQUACY, I'm unable to generate a response.")
	ai_msg = add_eotext_eoturn_to_ai_msg(
		ai_msg=ai_msg, 
		end_of_turn_id_token=END_OF_TURN_ID, 
		end_of_text_token=END_OF_TEXT
	)
	return [sys_msg, human_msg, ai_msg]



def request_verify(state: State) -> State:
	"""Request verification output of Agent Manager."""
	ai_msg_relevancy = req_ver_relevancy(state=state)[2]
	ai_msg_adequacy = req_ver_adequacy(state=state)[2]
	yes_no_relevancy = check_contain_yes_or_no(ai_msg=ai_msg_relevancy.content)
	yes_no_adequacy  = check_contain_yes_or_no(ai_msg=ai_msg_adequacy.content )
	ai_msg_yes_no = "YES" if "YES" in (yes_no_relevancy, yes_no_adequacy) else "NO"
	return {"messages": {
		"REQUEST_VERIFY": {
			"AI": [AIMessage(content=ai_msg_yes_no)] 
		}}}



def req_ver_determine_yes_or_no(state: State) -> State:
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
	return resp_map.get(resp, ValueError(f">>> [ERROR]: Unexpected response '{resp}'"))



def prompt_agent(state: State) -> State:
	"""Prompt Agent."""
	human_msg = "" # Tiến đến pipeline prompt-agent 
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

workflow.add_edge(START, "MANAGER_AGENT")
workflow.add_edge("MANAGER_AGENT", "REQUEST_VERIFY")
workflow.add_conditional_edges("REQUEST_VERIFY", req_ver_determine_yes_or_no)
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
Optimize the model using cross-validation on the training split to fine-tune hyperparameters, and aim for an accuracy of at least 0.95 (95%) on the test split. 
Provide the final trained model, a detailed report of the training process, hyperparameter settings, accuracy metrics, and a confusion matrix to evaluate performance across different categories.""",

	"""Please provide a classification model that categorizes images into one of four clothing categories. 
The image path, along with its label information, can be found in the files train labels.csv and test labels.csv. 
The model should achieve at least 0.95 (95%) accuracy on the test set and be implemented using PyTorch. 
Additionally, please include data augmentation techniques and a confusion matrix in the evaluation."""	
	
	"""Hello, What is heavier a kilo of feathers or a kilo of steel?""", 
	
	"""exit"""
]



if __name__ == "__main__":
	fire.Fire(main)
