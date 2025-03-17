import re
import json
import fire 
import logging
import streamlit as st 



from pydantic import BaseModel, TypeAdapter
from typing_extensions import List, Dict, Type



from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.messages import (HumanMessage, AIMessage, SystemMessage, BaseMessage)



from langgraph.graph import StateGraph, START, END



from state import State, Conversation, Prompt2JSON, ReAct, default_messages
from const_params import *



logging.basicConfig(level=logging.CRITICAL)



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



def add_special_token_to_human_query(human_msg: str) -> str:
	"""Enhances the human query by formatting it with special tokens of LLama 3 series models.

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



def add_eoturn_eotext_to_ai_msg(ai_msg: AIMessage, end_of_turn_id_token: str = END_OF_TURN_ID, end_of_text_token: str = END_OF_TEXT) -> AIMessage:
	"""Ensures AIMessage content ends with required special tokens.

	This function appends `<|end_of_text|>` and `<|eot_id|>` at the end of the message content if they are not already present.

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
	if not content.endswith(end_of_turn_id_token): content += end_of_turn_id_token
	if not content.endswith(end_of_turn_id_token + end_of_text_token): content = content.replace(end_of_turn_id_token, end_of_turn_id_token + end_of_text_token)
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
		BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
		START_HEADER_ID=START_HEADER_ID, 
		END_HEADER_ID=END_HEADER_ID, 
		END_OF_TURN_ID=END_OF_TURN_ID, 
		tools_desc="\n\n".join(list_tool_desc), 
		tools_name=", ".join(tool.name for tool in tools)
	)
	return prompt



def conversation2json(msg_prompt: str, llm_structure_output: Runnable[LanguageModelInput, Dict | BaseModel], human_msg: HumanMessage, schema: Type[Dict]) -> json:
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
			BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
			START_HEADER_ID=START_HEADER_ID, 
			END_HEADER_ID=END_HEADER_ID, 
			json_schema=json_schema, 
			human_msg=human_msg.content, 
			END_OF_TURN_ID=END_OF_TURN_ID
		))
		ai_msg_json = LLM_LTEMP.invoke([sys_msg])
		pattern = r"```json\n(.*?)\n```"
		match = re.search(pattern=pattern, string=ai_msg_json.content, flags=re.DOTALL)
		if not match: raise ValueError(">>> Không tìm thấy JSON hợp lệ trong phản hồi của mô hình.")
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
	if missing_keys: raise ValueError(f">>> JSON thiếu các trường bắt buộc: {missing_keys}")
	if extra_keys: raise ValueError(f">>> JSON có các trường không hợp lệ: {extra_keys}")
	return json_data



def manager_agent(state: State) -> State:
	"""Manager Agent.

	Example:
		>>> Human query: I need a very accurate model to classify images in the 
				Butterfly Image Classification dataset into their respective 
				categories. The dataset has been uploaded with its label 
				information in the labels.csv file.
		>>> AI response: Here is a sample code that uses the Keras library to develop and train a convolutional neural network (CNN) model for ...
	"""
	sys_msg = SystemMessage(content=MGR_SYS_MSG_PROMPT.format(
			BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
			START_HEADER_ID=START_HEADER_ID, 
			END_HEADER_ID=END_HEADER_ID, 
			END_OF_TURN_ID=END_OF_TURN_ID
		))
	human_msg = HumanMessage(content=add_special_token_to_human_query(human_msg=state["human_query"][-1].content))
	human_msg_json = HumanMessage(json.dumps((conversation2json(
			msg_prompt=CONVERSATION_2_JSON_MSG_PROMPT, llm_structure_output=LLM_STRUC_OUT_CONVERSATION, human_msg=human_msg, 
			schema=Conversation)), indent=2))
	ai_msg = LLM_LTEMP.invoke([sys_msg, human_msg, human_msg_json])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage( 
			content=ai_msg.strip() 
			if isinstance(ai_msg, str) 
			else "At node_manager_agent, I'm unable to generate a response."
		)
	ai_msg = add_eoturn_eotext_to_ai_msg(ai_msg=ai_msg, end_of_turn_id_token=END_OF_TURN_ID, end_of_text_token=END_OF_TEXT)
	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="SYS", msg=sys_msg)
	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN", msg=human_msg)
	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="AI", msg=AIMessage("<|json|>" + human_msg_json.content + "<|end_json|>" + ai_msg.content))
	return state



def check_contain_yes_or_no(ai_msg: str) -> str:
	"""Checks if the AI response contains 'Yes' or 'No'."""
	match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\s*(yes|no)\b", ai_msg, re.IGNORECASE)
	return match.group(1).upper() if match else "[ERROR]: Không tìm thấy 'Yes' hoặc 'No' trong phản hồi AIMessage!"



def relevancy(state: State) -> List[BaseMessage]:
	"""Check request verification-relevancy of human_query."""
	human_msg = state["human_query"][-1]
	sys_msg = SystemMessage(content=VER_RELEVANCY_MSG_PROMPT.format(
		instruction=human_msg.content, 
		BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
		START_HEADER_ID=START_HEADER_ID, 
		END_HEADER_ID=END_HEADER_ID, 
		END_OF_TURN_ID=END_OF_TURN_ID
	))
	ai_msg = LLM_LTEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage(content=ai_msg.strip() if isinstance(ai_msg, str) else "At node_request_verify-REQUEST_VERIFY_RELEVANCY, I'm unable to generate a response.")
	ai_msg = add_eoturn_eotext_to_ai_msg(ai_msg=ai_msg, end_of_turn_id_token=END_OF_TURN_ID, end_of_text_token=END_OF_TEXT)
	return [sys_msg, human_msg, ai_msg]



def adequacy(state: State) -> List[BaseMessage]:
	"""Check request verification-adequacy of AIMessage response with JSON object."""
	pattern = r"<\|json\|>(.*?)<\|end_json\|>"
	human_msg = state["messages"]['MANAGER_AGENT']['HUMAN'][-1]
	ai_msg = get_latest_msg(state=state, node="MANAGER_AGENT", msgs_type="AI")
	json_obj_from_ai_msg = re.findall(pattern=pattern, string=ai_msg.content, flags=re.DOTALL)[-1]
	sys_msg = SystemMessage(content=VER_ADEQUACY_MSG_PROMPT.format(
		BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
		START_HEADER_ID=START_HEADER_ID, 
		END_HEADER_ID=END_HEADER_ID, 
		parsed_user_requirements=json_obj_from_ai_msg, 
		END_OF_TURN_ID=END_OF_TURN_ID
	))
	ai_msg = LLM_LTEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage(content=ai_msg.strip() if isinstance(ai_msg, str) else "At node_request_verify-REQUEST_VERIFY_ADEQUACY, I'm unable to generate a response.")
	ai_msg = add_eoturn_eotext_to_ai_msg(ai_msg=ai_msg, end_of_turn_id_token=END_OF_TURN_ID, end_of_text_token=END_OF_TEXT)
	return [sys_msg, human_msg, ai_msg]



def request_verify(state: State) -> State:
	"""Request verification output of Agent Manager."""
	ai_msg_relevancy = relevancy(state=state)[2]
	ai_msg_adequacy = adequacy(state=state)[2]
	yes_no_relevancy = check_contain_yes_or_no(ai_msg=ai_msg_relevancy.content)
	yes_no_adequacy  = check_contain_yes_or_no(ai_msg=ai_msg_adequacy.content )
	yes_or_no_answer = "YES" if "YES" in (yes_no_relevancy, yes_no_adequacy) else "NO"
	ai_msg = AIMessage(content=yes_or_no_answer)
	add_unique_msg(state=state, node="REQUEST_VERIFY", msgs_type="AI", msg=ai_msg)
	return state



def req_ver_yes_or_no_control_flow(state: State) -> State:
	"""Determines the next step based on the AI response from REQUEST_VERIFY.

	Args:
		state (State): The current conversation state.

	Returns:
		str: The next agent ("PROMPT_AGENT" or END).

	Raises:
		ValueError: If there is no valid AI response or an unexpected response.
	"""
	if "REQUEST_VERIFY" not in state["messages"] or "AI" not in state["messages"]["REQUEST_VERIFY"]:
		raise ValueError("[ERROR]: No AI message found in REQUEST_VERIFY.")
	ai_msgs = state["messages"]["REQUEST_VERIFY"]["AI"]
	if not ai_msgs:
		raise ValueError("[ERROR]: AI message list is empty in REQUEST_VERIFY.")
	ai_msg = ai_msgs[-1]
	if not hasattr(ai_msg, "content"):
		raise ValueError("[ERROR]: AI message has no content.")
	resp = ai_msg.content.strip().upper()
	next_step_map = {"YES": "PROMPT_AGENT", "NO": END}
	if resp in next_step_map:
		return next_step_map[resp]
	raise ValueError(f">>> [ERROR]: Unexpected response '{resp}'")



def prompt_agent(state: State) -> State:
	"""Prompt Agent parses user's requirements into JSON following a TypedDict schema.

	Args:
		state (State): The current state of the conversation.

	Returns:
		State: Updated state with the parsed JSON response.
	"""
	if "MANAGER_AGENT" not in state["messages"] or "HUMAN" not in state["messages"]["MANAGER_AGENT"]:
		raise ValueError(">>> [ERROR]: No HUMAN messages found in MANAGER_AGENT.")
	human_msg = get_latest_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN")
	parsed_json = conversation2json(msg_prompt=PROMPT_2_JSON_SYS_MSG_PROMPT, 
		llm_structure_output=LLM_STRUC_OUT_AUTOML,
		human_msg=human_msg,
		schema=Prompt2JSON
	)
	expected_keys, received_keys = set(Prompt2JSON.__annotations__), set(parsed_json)
	if missing_keys := expected_keys - received_keys: 
		raise ValueError(f"[ERROR]: JSON thiếu các trường bắt buộc: {missing_keys}")
	if extra_keys := received_keys - expected_keys: 
		raise ValueError(f"[ERROR]: JSON có các trường không hợp lệ: {extra_keys}")
	ai_msg_json = AIMessage(content=json.dumps(parsed_json, indent=2))
	add_unique_msg(state=state, node="PROMPT_AGENT", msgs_type="AI", msg=ai_msg_json)
	return state



def retrieval_augmented_planning_agent(state: State) -> State:
	"Retrieval-Augmented Planning Agent."
	human_msg_content = get_latest_msg(state=state, node="PROMPT_AGENT", msgs_type="AI").content
	plan_knowledge = "" 
	sys_msg = SystemMessage(content=RAP_SYS_MSG_PROMPT.format(
		BEGIN_OF_TEXT=BEGIN_OF_TEXT, 
		START_HEADER_ID=START_HEADER_ID, 
		END_HEADER_ID=END_HEADER_ID, 
		user_requirements=human_msg_content, 
		plan_knowledge=plan_knowledge,
		END_OF_TURN_ID=END_OF_TURN_ID
	))
	ai_msg = LLM_LTEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage(
			content=ai_msg.strip() 
			if isinstance(ai_msg, str) 
			else "At node_manager_agent, I'm unable to generate a response."
		)
	ai_msg = add_eoturn_eotext_to_ai_msg(
		ai_msg=ai_msg, 
		end_of_turn_id_token=END_OF_TURN_ID, 
		end_of_text_token=END_OF_TEXT
	)
	add_unique_msg(state=state, node="RAP", msgs_type="SYS", msg=sys_msg)
	add_unique_msg(state=state, node="RAP", msgs_type="HUMAN", msg=HumanMessage(human_msg_content))
	add_unique_msg(state=state, node="RAP", msgs_type="AI", msg=ai_msg)
	return state



def data_agent(state: State) -> State:
	"Data Agent."
	return state



def model_agent(state: State) -> State:
	"Model Agent."
	return state


# TODO: Tích hợp auto-cot
# TODO: Trước tiên cần tối ưu prompt.
# TODO: Build RAG pipeline


# TODO: Ý tưởng sử dụng Multi-Agent gọi đến Yolov8 API, Yolov8 
# API sẽ lấy mọi hình ảnh cỡ nhỏ nó phát hiện  được là lỗi và 
# đưa vào mô hình llama3.2-11b-vision để đọc ảnh, sau đó 
# Llama3.2-11b-vision gửi lại text đến Multi-Agent để 
# Multi-Agent xác định xem đấy có phải là lỗi không.

# Nếu muốn vậy thì hiện tại ta cần có data-agent và model-agent
# data-agent để generate dữ liệu training, model-agent để viết 
# kiến trúc model vision.

# Bây giờ cần build tools trước cho mô hình, bao gồm 
# tool vision, tool genData, tool llama3.2-11b-vision-instruct

# Nhưng tại sao lại ko dùng vision yolov8 để finetune. Mục tiêu 
# ở đây, ta sẽ tận dụng cả sức mạnh của LLM-Vision, để hiểu rõ
# hình ảnh có gì, sau đó gửi về cho LLM-instruct để xử lý text 
# từ LLM-vision.

# **1. Image-to-Image Generation (CycleGAN, Pix2Pix, StyleGAN)**
# **2. Data Augmentation (Biến đổi dữ liệu)**
workflow = StateGraph(State)

workflow.add_node("MANAGER_AGENT", manager_agent)
workflow.add_node("REQUEST_VERIFY", request_verify)
workflow.add_node("PROMPT_AGENT", prompt_agent)
workflow.add_node("RAP", retrieval_augmented_planning_agent)
workflow.add_node("DATA_AGENT", data_agent)
workflow.add_node("MODEL_AGENT", model_agent)

workflow.add_edge(START, "MANAGER_AGENT")
workflow.add_edge("MANAGER_AGENT", "REQUEST_VERIFY")
workflow.add_conditional_edges("REQUEST_VERIFY", req_ver_yes_or_no_control_flow, ["PROMPT_AGENT", END])
workflow.add_edge("PROMPT_AGENT", "RAP")
workflow.add_edge("RAP", "DATA_AGENT")
workflow.add_edge("RAP", "MODEL_AGENT")
workflow.add_edge("DATA_AGENT", "MODEL_AGENT")
workflow.add_edge("MODEL_AGENT", END)

app = workflow.compile(checkpointer=CHECKPOINTER, store=STORE, debug=DEBUG, name=NAME)



def main() -> None:
	"""Xử lý truy vấn của người dùng và hiển thị phản hồi từ AI.

	Workflow:
		1. Nhận truy vấn từ danh sách `QUERIES`.
		2. Kiểm tra xem người dùng có nhập "exit" để thoát không.
		3. Gửi truy vấn đến hệ thống AI thông qua `app.invoke()`.
		4. Kiểm tra tính hợp lệ của dữ liệu trả về từ `app.invoke()`.
		5. Trích xuất `messages` và hiển thị kết quả hội thoại.

	Raises:
		ValueError: Nếu `app.invoke()` không trả về dictionary hoặc không chứa key "messages".
	"""
	for i, user_query in enumerate(QUERIES, 1):
		print(f"\n👨_query_{i}:")
		print(user_query)
		print("\n🤖_response:")
		user_query = user_query.strip()
		if user_query.lower() == "exit":
			print("\n>>> [System Exit] Goodbye! Have a great day! 😊\n")
			break
		state_data = app.invoke(input={"human_query": [HumanMessage(user_query)], "messages": default_messages()}, config=CONFIG)
		if not isinstance(state_data, dict):
			raise ValueError("[ERROR]: app.invoke() không trả về dictionary.")
		if "messages" not in state_data:
			raise ValueError("[ERROR]: Key 'messages' không có trong kết quả.")
		messages = state_data["messages"]
		display_conversation_results(messages)



def display_conversation_results(messages: dict) -> None:
	"""Hiển thị kết quả hội thoại từ tất cả các agent trong hệ thống.

	Args:
		messages (dict): Dictionary chứa các tin nhắn được nhóm theo agent và loại tin nhắn.
			- **Node**: Tên agent (ví dụ: "MANAGER_AGENT", "PROMPT_AGENT").
			- **Msgs_type**: Dictionary chứa các loại tin nhắn ("HUMAN", "AI", "SYS"), mỗi loại là một danh sách tin nhắn.

	Example:
		messages = {
			"MANAGER_AGENT": {
				"SYS": [],
				"HUMAN": [HumanMessage(content="Hello!")],
				"AI": [AIMessage(content="Hi! How can I assist you?")]
			}
		}

	Returns:
		None: Hàm chỉ hiển thị kết quả trên terminal mà không trả về giá trị.
	"""
	print("\n===== [CONVERSATION RESULTS] =====\n")
	if not messages:
		print("[INFO]: Không có tin nhắn nào trong hội thoại.")
		return
	for node, msgs in messages.items():
		print(f"\n[{node}]")
		if isinstance(msgs, dict):
			for msg_category, msg_list in msgs.items():
				if msg_list:
					print(f"  {msg_category}:")
					for msg in msg_list:
						content = getattr(msg, "content", "[No content]")
						print(f"\t- {content}")
		else:
			raise ValueError(f"`msgs` phải là một dictionary chứa danh sách tin nhắn, `msgs` hiện tại là: {msgs}")
	print("\n===== [END OF CONVERSATION] =====\n")



if __name__ == "__main__":
	fire.Fire(main)
