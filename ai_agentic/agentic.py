from pprint import pprint
import fire 
from datetime import datetime
from transformers import AutoTokenizer
from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import (
	HumanMessage, 
	trim_messages
)
from langgraph.graph import (
	StateGraph, 
	START, END
)
from state import State, Input, Output
from prompts import (
	generate_prompt, 
	explain_prompt, 
	router_prompt, 
	medical_records_prompt, 
	insurance_faqs_prompt
)

def token_counter(messages):
    """Đếm số lượng token từ danh sách tin nhắn."""
    text = " ".join([msg.content for msg in messages])
    return len(tokenizer.encode(text)) 

def save_chat_history(user_input, assistant_response):
	"""Lưu lịch sử hội thoại vào ChromaDB."""
	timestamp =  datetime.now().strftime("%Y-%m-%d_%H:%M:%S-%f")
	text_to_store = f"""[{timestamp}] User: {user_input}\n[{timestamp}] Assistant: {assistant_response}"""
	vector_db.add_texts(
		texts=[text_to_store],
		metadatas=[{"source": "chat_history"}]
	)

def generate_sql(state: State) -> State:
	"""Update conversation history

	Args:
		state (State): _description_

	Returns:
		State: _description_
	"""
	user_message = HumanMessage(state["user_query"])
	messages = [generate_prompt, *state["messages"], user_message]
	respond = model_low_temp.invoke(messages)
	return {
		"sql_query": respond.content,
		"messages": [user_message, respond],
	}

def explain_sql(state: State) -> State:
	"""Contains user's query, SQL query from prev step and update update conversation history.
	
	Args:
		state (State): _description_
	
	Returns:
		State: _description_
	"""
	messages = [
		explain_prompt,
		*state["messages"]
	]
	respond = model_high_temp.invoke(messages)
	return {
		"sql_explanation": respond.content,
		"messages": respond,
	}

def router_node(state: State) -> State:
	"""_summary_.

	Args:
		state (State): _description_

	Returns:
		State: _description_
	"""
	user_message = HumanMessage(state["user_query"])
	messages = [router_prompt, *state["messages"], user_message]
	respond = model_low_temp.invoke(messages)
	return {
		"domain": respond.content,
		"messages": [user_message, respond],
	}

def pick_retriever(state: State) -> Literal["retrieve_medical_records", "retrieve_insurance_faqs"]:
	if state["domain"] == "records":
		return "retrieve_medical_records"
	else:
		return "retrieve_insurance_faqs"

def retrieve_medical_records(state: State) -> State:
	documents = medical_records_retriever.invoke(state["user_query"])
	return {"documents": documents}

def retrieve_insurance_faqs(state: State) -> State:
	documents = insurance_faqs_retriever.invoke(state["user_query"])
	return {"documents": documents}

def generate_answer(state: State) -> State:
	if state["domain"] == "records":
		prompt = medical_records_prompt
	else:
		prompt = insurance_faqs_prompt
	messages = [
		prompt,
		*state["messages"],
		HumanMessage(f"Documents: {state["documents"]}"),
	]
	respond = model_high_temp.invoke(messages)
	return {
		"answer": respond.content,
		"messages": respond,
	}

if True:
	tokenizer = AutoTokenizer.from_pretrained("/home/chwenjun225/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B")
	persist_directory = "/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
	collection_name = "medical_and_insurance"
	embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
	
	embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
	vector_db = Chroma(
		persist_directory=persist_directory, 
		embedding_function=embeddings, 
		collection_name=collection_name
	)

	medical_records_store = vector_db.from_documents([], embeddings )
	medical_records_retriever = medical_records_store.as_retriever()

	insurance_faqs_store = vector_db.from_documents([], embeddings )
	insurance_faqs_retriever = insurance_faqs_store.as_retriever()

	model_low_temp = ChatOpenAI(
		model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct", 
		openai_api_base="http://127.0.0.1:2026/v1", 
		openai_api_key="model_low_temp",
		temperature=0.1
	)
	model_high_temp = ChatOpenAI(
		model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct", 
		openai_api_base="http://127.0.0.1:2026/v1", 
		openai_api_key="model_higt_temp",
		temperature=0.7
	)
	trimmer = trim_messages(
		max_tokens=65,
		strategy="last",
		token_counter=token_counter,
		include_system=True,
		allow_partial=False,
		start_on="human",
	)

def main():
	"""Thực thi chương trình."""
	builder = StateGraph(State, input=Input, output=Output)
	builder.add_node("generate_sql", generate_sql)
	builder.add_node("explain_sql", explain_sql)

	builder.add_edge(START, "generate_sql")
	builder.add_edge("generate_sql", "explain_sql")
	builder.add_edge("explain_sql", END)

	graph = builder.compile()
	graph.invoke({
		"user_query": "What is the total sales for each product?"
	})
	input = {"messages": [HumanMessage('hi!')]}
	for chunk in graph.stream(input):
		pprint(chunk)

if __name__ == "__main__":
	fire.Fire(main)












	# model = get_llm(
	# 	port=2026, 
	# 	host="127.0.0.1", 
	# 	version="v1", 
	# 	temperature=0, 
	# 	openai_api_key="chwenjun225",
	# 	model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct"
	# )







# def get_llm(
# 		port=2026, 
# 		host="127.0.0.1", 
# 		version="v1", 
# 		temperature=0.2, 
# 		openai_api_key="chwenjun225",
# 		model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct"
# 	):
# 	"""Khởi tạo mô hình ngôn ngữ lớn DeepSeek-R1 từ LlamaCpp-Server."""
# 	openai_api_base = f"""http://{host}:{port}/{version}""" 
# 	return ChatOpenAI(
# 		model=model_name, 
# 		openai_api_base=openai_api_base, 
# 		openai_api_key=openai_api_key, 
# 		temperature=temperature
# 	)


# def chatbot(state: State):
# 	answer = model.invoke(state["messages"])
# 	return {"messages": [answer]}