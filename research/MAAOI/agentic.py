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
from const_vars import QUERIES, DEBUG, NAME, COLLECTION_NAME, EMBEDDING_MODEL_NAME, EMBEDDING_MODEL, PERSIS_DIRECTORY, VECTOR_DB, CONVERSATION_2_JSON_MSG_PROMPT, MGR_SYS_MSG_PROMPT, VER_RELEVANCY_MSG_PROMPT, VER_ADEQUACY_MSG_PROMPT, PROMPT_2_JSON_SYS_MSG_PROMPT, RAP_SYS_MSG_PROMPT, SPECIAL_TOKENS_LLAMA_MODELS, CONFIG, CHECKPOINTER, STORE, LLM_HTEMP, LLM_LTEMP, LLM_STRUC_OUT_AUTOML, LLM_STRUC_OUT_CONVERSATION
from nodes import manager_agent, request_verify, prompt_agent, retrieval_augmented_planning_agent, data_agent, model_agent
from auto_cot.api import cot



# TODO: Tích hợp auto-cot.

# TODO: Trước tiên cần tối ưu prompt.

# TODO: Build RAG pipeline.

# TODO:  `*state["messages"],` để đưa ngữ cảnh lịch sử trò chuyện vào mô hình.

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

# TODO: Lấy trạng thái, lịch sử bằng cách `app.get_state(CONFIG).values`

# TODO: Working on this, build integrate auto chain of thought to multi-agent
# x = cot(method="auto_cot", question="", debug=False)

# logging.basicConfig(level=logging.CRITICAL)
workflow = StateGraph(State)

workflow.add_node("MANAGER_AGENT", manager_agent)
workflow.add_node("REQUEST_VERIFY", request_verify)
workflow.add_node("PROMPT_AGENT", prompt_agent)
workflow.add_node("RAP", retrieval_augmented_planning_agent)
workflow.add_node("DATA_AGENT", data_agent)
workflow.add_node("MODEL_AGENT", model_agent)

workflow.add_edge(START, "MANAGER_AGENT")
workflow.add_edge("MANAGER_AGENT", "REQUEST_VERIFY")
workflow.add_conditional_edges("REQUEST_VERIFY", request_verify, ["PROMPT_AGENT", END])
workflow.add_edge("PROMPT_AGENT", END)


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
		state_data = app.invoke(
			input={
				"human_query": [HumanMessage(user_query)], 
				"messages": default_messages()}, 
			config=CONFIG)
		if not isinstance(state_data, dict): raise ValueError("[ERROR]: app.invoke() không trả về dictionary.")
		if "messages" not in state_data: raise ValueError("[ERROR]: Key 'messages' không có trong kết quả.")
		messages = state_data["messages"]
		print("\n===== [CONVERSATION RESULTS] =====\n")
		display_conversation_results(messages)
		print("\n===== [END OF CONVERSATION] =====\n")



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



if __name__ == "__main__":
	fire.Fire(main)
