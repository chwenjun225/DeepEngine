
import re 
import json 



from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool 



from langchain_core.messages import BaseMessage, BaseMessage, SystemMessage, AIMessage



from state import State 
from const_vars import (
	LLAMA_TOKENS, 
	ENCODING, 
	MAX_TOKENS
)



def passthrough() -> RunnableLambda:
	"""Trả về một Runnable không thay đổi state."""
	return RunnableLambda(lambda x: x)



def trim_context(messages: list[dict]) -> list[dict]:
	"""Cắt bớt context nếu vượt token limit (cắt từ đầu)."""
	while count_tokens(messages) > MAX_TOKENS and len(messages) > 1:
		messages = messages[1:]
	return messages



def has_system_prompt(messages: list[dict], agent_name: str) -> bool:
	"""Kiểm tra đã có system message cho agent này chưa, hỗ trợ OpenAI format và BaseMessage."""
	for msg in messages:
		if isinstance(msg, dict):
			if msg.get("role") == "system" and msg.get("name") == agent_name:
				return True
		elif isinstance(msg, SystemMessage):
			if getattr(msg, "name", None) == agent_name:
				return True
	return False



def count_tokens(messages: list[dict]) -> int:
	"""Đếm tổng số tokens trong messages theo model."""
	num_tokens = 0
	for msg in messages:
		num_tokens += 4
		for _, value in msg.items():
			num_tokens += len(ENCODING.encode(str(value)))
	num_tokens += 2
	return num_tokens



def estimate_tokens(text: str) -> int:
	"""Ước lượng token trong prompt."""
	return len(ENCODING.encode(text))



def get_safe_num_predict(prompt: str, max_context: int = 131072, buffer: int = 512) -> int:
	"""Hàm tự động tính num_predict

	Args:
		prompt: nội dung bạn muốn gửi vào mô hình
		max_context: tổng context window của model (token)
		model_name: để chọn đúng tokenizer
		buffer: token chừa ra để tránh cắt hoặc lỗi
	"""
	prompt_tokens = estimate_tokens(prompt)
	available_tokens = max_context - prompt_tokens - buffer
	return max(256, min(available_tokens, 128000))



def get_latest_user_query(state: State) -> dict:
	"""Lấy truy vấn người dùng mới nhất, O(1)."""
	return state["user_query"]



def get_msgs(state: State, node: str, type_msgs) -> list[dict]:
	"""Lấy danh sách tin nhắn của một node theo loại tin nhắn."""
	return state["messages"][node][type_msgs]



def get_latest_msg(state: State, node: str, type_msgs: str) -> dict|None:
	"""Lấy tin nhắn mới nhất từ một node theo loại tin nhắn, O(1)."""
	return state["messages"][node][type_msgs][-1] if \
		get_msgs(state=state, node=node, type_msgs=type_msgs) \
			else None 



def add_unique_msg(state: State, node: str, type_msgs: str, msg: dict) -> None:
	"""Chỉ thêm nếu khác với tin nhắn cuối, O(1)."""
	existing_msgs = get_msgs(state=state, node=node, type_msgs=type_msgs)
	if not existing_msgs or existing_msgs[-1]["content"] != msg["content"]:
		existing_msgs.append(msg)



def build_react_sys_msg_prompt(tool_desc_prompt: str, react_prompt: str, tools: list[BaseTool]) -> str:
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
		BEGIN_OF_TEXT=LLAMA_TOKENS["BEGIN_OF_TEXT"], 
		START_HEADER_ID=LLAMA_TOKENS["START_HEADER_ID"], 
		END_HEADER_ID=LLAMA_TOKENS["END_HEADER_ID"], 
		END_OF_TURN_ID=LLAMA_TOKENS["END_OF_TURN_ID"], 
		tools_desc="\n\n".join(list_tool_desc), 
		tools_name=", ".join(tool.name for tool in tools)
	)
	return prompt



# def conversation2json(
# 		msg_prompt: str, 
# 		llm_structure_output: Runnable[LanguageModelInput, Dict | BaseModel], 
# 		human_msg: Dict, 
# 		schema: Type[Dict]
# 	) -> json:
# 	"""Parses user's query into structured JSON for manager_agent."""
# 	json_data = llm_structure_output.invoke([human_msg])
# 	if not json_data:
# 		json_schema = json.dumps(TypeAdapter(schema).json_schema()["properties"], indent=2).strip()
# 		sys_msg = SystemMessage(content=msg_prompt.format(
# 			BEGIN_OF_TEXT=LLAMA_TOKENS["BEGIN_OF_TEXT"], 
# 			START_HEADER_ID=LLAMA_TOKENS["START_HEADER_ID"], 
# 			END_HEADER_ID=LLAMA_TOKENS["END_HEADER_ID"], 
# 			json_schema=json_schema, 
# 			human_msg=human_msg.content, 
# 			END_OF_TURN_ID=LLAMA_TOKENS["END_OF_TURN_ID"]
# 		))
# 		ai_msg_json = LLM_LTEMP.invoke([sys_msg])
# 		pattern = r"```json\n(.*?)\n```"
# 		match = re.search(pattern=pattern, string=ai_msg_json.content, flags=re.DOTALL)
# 		if not match: 
# 			raise ValueError(">>> Không tìm thấy JSON hợp lệ trong phản hồi của mô hình.")
# 		json_string = match.group(1).strip()
# 		try:
# 			json_data = json.loads(json_string)
# 			if DEBUG: 
# 				print(">>> JSON hợp lệ:")
# 				print(json.dumps(json_data, indent=2, ensure_ascii=False))
# 		except json.JSONDecodeError as e:
# 			raise ValueError(f">>> JSON không hợp lệ (DecodeError): {e}")
# 	missing_keys = set(schema.__annotations__.keys()) - json_data.keys()
# 	extra_keys = json_data.keys() - set(schema.__annotations__.keys())
# 	if missing_keys:
# 		raise ValueError(f">>> JSON thiếu các trường bắt buộc: {missing_keys}")
# 	if extra_keys: 
# 		raise ValueError(f">>> JSON có các trường không hợp lệ: {extra_keys}")
# 	return json_data



def check_msg_yes_or_no(ai_msg: str) -> str:
	"""Checks if the AI response contains 'Yes' or 'No'."""
	match = re.search(
		r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\s*(\byes\b|\bno\b)",  
		ai_msg, re.IGNORECASE
	)
	return match.group(1).upper() \
		if match \
			else "[ERROR]: Không tìm thấy 'Yes' hoặc 'No' trong phản hồi AIMessage!"
