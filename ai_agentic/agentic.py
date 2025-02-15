from uuid import uuid4 
from pprint import pprint
import fire 
from datetime import datetime
from transformers import AutoTokenizer
from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_core.messages import ToolCall, AIMessage, HumanMessage, ToolMessage, trim_messages # TODO: Cần thêm tính năng lọc và cắt tin nhắn

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from state import State, Input, Output
from prompts import generate_prompt, reflection_prompt
from tools_use import DuckDuckGoSearchRun, calculator

if True:
	tokenizer = AutoTokenizer.from_pretrained("/home/chwenjun225/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B")
	embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

	search = DuckDuckGoSearchRun()
	tools = [search, calculator]

	model = ChatOpenAI(
		model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct", 
		openai_api_base="http://127.0.0.1:2026/v1", 
		openai_api_key="chwenjun225",
		temperature=0.1
	)

	tools_retriever = InMemoryVectorStore.from_documents(
		[Document(tool.description, metadata={"name": tool.name}) for tool in tools],
		embeddings,
	).as_retriever()

	config = {"configurable": {"thread_id": "1"}}

def token_counter(messages):
	"""Đếm số lượng token từ danh sách tin nhắn."""
	text = " ".join([msg.content for msg in messages])
	return len(tokenizer.encode(text)) 

def chatbot(state: State) -> State:
	selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
	answer = model.bind_tools(selected_tools).invoke([generate_prompt] + state["messages"])
	return {"messages": [answer]}

def select_tools(state: State) -> State:
	query = state["messages"][-1].content
	tool_docs = tools_retriever.invoke(query)
	return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}

def reflect(state: State) -> State:
	class_map = {
		AIMessage: HumanMessage, 
		HumanMessage: AIMessage, 
		# ToolMessage: HumanMessage 
	}
	translated = [reflection_prompt, state["messages"][0]] + [
		class_map[msg.__class__](content=msg.content) 
		for msg in state["messages"][1:]
	]
	answer = model.invoke(translated)
	return {"messages": [HumanMessage(content=answer.content)]}

def should_continue(state: State):
	if len(state["messages"]) > 6:
		return END
	else:
		return "reflect"

def main():
	"""Thực thi chương trình."""
	builder = StateGraph(State)

	builder.add_node("select_tools", select_tools)
	builder.add_node("chatbot", chatbot)
	builder.add_node("tools", ToolNode(tools))
	builder.add_node("reflect", reflect)

	builder.add_edge(START, "select_tools")
	builder.add_edge("select_tools", "tools")
	builder.add_edge("tools", "chatbot")
	builder.add_conditional_edges("chatbot", tools_condition)
	builder.add_conditional_edges("chatbot", should_continue)
	builder.add_edge("reflect", "chatbot")

	graph = builder.compile(checkpointer=MemorySaver())

	user_input = {
		"messages": [HumanMessage("""What is the capital of Japan?""")]
	}
	for chunk in graph.stream(user_input, config):
		print(chunk)

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