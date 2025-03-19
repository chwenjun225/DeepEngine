import re 
import json 
from typing_extensions import Type, List, Dict
from pydantic import BaseModel, TypeAdapter



from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool 
from langchain_core.language_models import LanguageModelInput



from utils import check_contain_yes_or_no
from state import State 
from const_vars import (
	DEBUG, SPECIAL_TOKENS_LLAMA_MODELS, 
	LLM_LTEMP, VER_RELEVANCY_MSG_PROMPT, 
	VER_ADEQUACY_MSG_PROMPT
)



def get_latest_msg(state: State, node: str, msgs_type: str) -> BaseMessage:
	"""Lấy tin nhắn mới nhất từ một node (agent), đảm bảo tốc độ O(1)."""
	return state["messages"][node][msgs_type][-1]



def add_unique_msg(state: State, node: str, msgs_type: str, msg: BaseMessage) -> None:
	"""Thêm tin nhắn nếu chưa có, tối ưu hóa O(1)."""
	if node == "REQUEST_VERIFY":
		state["messages"][node][msgs_type] = [msg]
	else:
		existing_msgs = state["messages"][node][msgs_type] 
		if not existing_msgs or existing_msgs[-1].content != msg.content:
			existing_msgs.append(msg)



def add_special_token_to_human_msg(human_msg: str) -> str:
	"""Format human message with LLama3's special tokens.

	Args:
		human_msg (str): User's query.

	Returns:
		formatted_query (str): A formatted human query wrapped with special tokens.

	Example:
		>>> state.user_query = [HumanMessage(content="What is AI?")]
		>>> enhance_human_query(state)
		formatted_query="<|start_header_id|>HUMAN<|end_header_id|>What is AI?<|end_of_turn_id|><|start_header_id|>AI<|end_header_id|>"
	"""
	formatted_query = (
		f"{SPECIAL_TOKENS_LLAMA_MODELS["START_HEADER_ID"]}HUMAN{SPECIAL_TOKENS_LLAMA_MODELS["END_HEADER_ID"]}"
		f"{human_msg}{SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"]}"
		f"{SPECIAL_TOKENS_LLAMA_MODELS["START_HEADER_ID"]}AI{SPECIAL_TOKENS_LLAMA_MODELS["END_HEADER_ID"]}"
	)
	return formatted_query



def add_special_token_ai_msg(
		ai_msg: AIMessage, 
		end_of_turn_id_token: str = SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"], 
		end_of_text_token: str = SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TEXT"]
	) -> AIMessage:
	"""Appends `<|end_of_text|>` and `<|eot_id|>` at the end of the message content if they are not already present.

	Args:
		ai_msg (AIMessage): The AI-generated message.
		end_of_text_token (str, optional) = "<|end_of_text|>": LLama3's special tokens.
		end_of_turn_id_token (str, optional) = "<|eot_id|>": LLama3's special tokens.

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



def build_react_sys_msg_prompt(tool_desc_prompt: str, react_prompt: str, tools: List[BaseTool]) -> str:
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
		BEGIN_OF_TEXT=SPECIAL_TOKENS_LLAMA_MODELS["BEGIN_OF_TEXT"], 
		START_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["START_HEADER_ID"], 
		END_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_HEADER_ID"], 
		END_OF_TURN_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"], 
		tools_desc="\n\n".join(list_tool_desc), 
		tools_name=", ".join(tool.name for tool in tools)
	)
	return prompt



def conversation2json(
		msg_prompt: str, 
		llm_structure_output: Runnable[LanguageModelInput, Dict | BaseModel], 
		human_msg: HumanMessage, 
		schema: Type[Dict]
	) -> json:
	"""Parses user's query into structured JSON for manager_agent.

	Args:
		llm_structure_output (Runnable[LanguageModelInput, Dict | BaseModel]): LLM function to generate structured output.
		human_msg (HumanMessage): User's input message.
		schema (Type[Dict]): TypedDict schema to validate JSON.

	Returns:
		Dict: Validated JSON containing:
			{
				"response": "A conversational response to the user's query.",
				"justification": "A brief explanation or reasoning behind the response."
			}

	Raises:
		ValueError: If the response does not contain valid JSON or is missing required fields.
	"""
	json_data = llm_structure_output.invoke([human_msg])
	if not json_data:
		json_schema = json.dumps(TypeAdapter(schema).json_schema()["properties"], indent=2).strip()
		sys_msg = SystemMessage(content=msg_prompt.format(
			BEGIN_OF_TEXT=SPECIAL_TOKENS_LLAMA_MODELS["BEGIN_OF_TEXT"], 
			START_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["START_HEADER_ID"], 
			END_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_HEADER_ID"], 
			json_schema=json_schema, 
			human_msg=human_msg.content, 
			END_OF_TURN_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"]
		))
		ai_msg_json = LLM_LTEMP.invoke([sys_msg])
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
	missing_keys = set(schema.__annotations__.keys()) - json_data.keys()
	extra_keys = json_data.keys() - set(schema.__annotations__.keys())
	if missing_keys:
		raise ValueError(f">>> JSON thiếu các trường bắt buộc: {missing_keys}")
	if extra_keys: 
		raise ValueError(f">>> JSON có các trường không hợp lệ: {extra_keys}")
	return json_data



def check_msg_yes_or_no(ai_msg: str) -> str:
	"""Checks if the AI response contains 'Yes' or 'No'."""
	match = re.search(
		r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\s*(yes|no)\b", 
		ai_msg, re.IGNORECASE
	)
	return match.group(1).upper() \
		if match \
			else "[ERROR]: Không tìm thấy 'Yes' hoặc 'No' trong phản hồi AIMessage!"




def relevancy(state: State) -> List[BaseMessage]:
	"""Check request verification-relevancy of human_query."""
	human_msg = state["human_query"][-1]
	sys_msg = SystemMessage(content=VER_RELEVANCY_MSG_PROMPT.format(
		instruction=human_msg.content, 
		BEGIN_OF_TEXT=SPECIAL_TOKENS_LLAMA_MODELS["BEGIN_OF_TEXT"], 
		START_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["START_HEADER_ID"], 
		END_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_HEADER_ID"], 
		END_OF_TURN_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"]
	))
	ai_msg = LLM_LTEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage(content=ai_msg.strip() if isinstance(ai_msg, str) else "At node_request_verify-REQUEST_VERIFY_RELEVANCY, I'm unable to generate a response.")
	ai_msg = add_special_token_ai_msg(ai_msg=ai_msg, end_of_turn_id_token=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"], end_of_text_token=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TEXT"])
	return [sys_msg, human_msg, ai_msg]



def adequacy(state: State) -> List[BaseMessage]:
	"""Check request verification-adequacy of AIMessage response with JSON object."""
	pattern = r"<\|json\|>(.*?)<\|end_json\|>"
	human_msg = state["messages"]['MANAGER_AGENT']['HUMAN'][-1]
	ai_msg = get_latest_msg(state=state, node="MANAGER_AGENT", msgs_type="AI")
	json_obj_from_ai_msg = re.findall(pattern=pattern, string=ai_msg.content, flags=re.DOTALL)[-1]
	sys_msg = SystemMessage(content=VER_ADEQUACY_MSG_PROMPT.format(
		BEGIN_OF_TEXT=SPECIAL_TOKENS_LLAMA_MODELS["BEGIN_OF_TEXT"], 
		START_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["START_HEADER_ID"], 
		END_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_HEADER_ID"], 
		parsed_user_requirements=json_obj_from_ai_msg, 
		END_OF_TURN_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"]
	))
	ai_msg = LLM_LTEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage(content=ai_msg.strip() if isinstance(ai_msg, str) else "At node_request_verify-REQUEST_VERIFY_ADEQUACY, I'm unable to generate a response.")
	ai_msg = add_special_token_ai_msg(ai_msg=ai_msg, end_of_turn_id_token=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"], end_of_text_token=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TEXT"])
	return [sys_msg, human_msg, ai_msg]



def request_verify_adequacy_relevancy(state: State) -> State:
	"""Request verification output of Agent Manager."""
	ai_msg_relevancy = relevancy(state=state)[2]
	ai_msg_adequacy = adequacy(state=state)[2]
	yes_no_relevancy = check_contain_yes_or_no(ai_msg=ai_msg_relevancy.content)
	yes_no_adequacy  = check_contain_yes_or_no(ai_msg=ai_msg_adequacy.content )
	yes_or_no_answer = "YES" if "YES" in (yes_no_relevancy, yes_no_adequacy) else "NO"
	ai_msg = AIMessage(content=yes_or_no_answer)
	add_unique_msg(state=state, node="REQUEST_VERIFY", msgs_type="AI", msg=ai_msg)
	return state