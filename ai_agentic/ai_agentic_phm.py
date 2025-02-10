from langgraph.prebuilt import create_react_agent 
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.store.memory import InMemoryStore

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from mtools import (rag, planning, save_chat_history, search)

def get_llm(port, host, openai_api_key, model_name, temperature):
	"""
	Khởi tạo mô hình ngôn ngữ lớn DeepSeek-R1 từ LlamaCpp-Server.
	"""
	openai_api_base = "http://" + str(host) + ":" + str(port) 
	return ChatOpenAI(
		model=model_name, 
		openai_api_base=openai_api_base, 
		openai_api_key=openai_api_key, 
		temperature=temperature
	)

persist_directory = "/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
collection_name = "foxconn_ai_research"

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
system_prompt = """You are a friendly and helpful assistant. 
Your job is to answer human questions with care and detail. 
Keep your answers short and concise when possible."""

planning_prompt = ChatPromptTemplate.from_messages([("human", """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise. 
Question: {question} 
Context: {context} 
Answer:""")]) 

embedding_model = HuggingFaceEmbeddings(
	model_name=embedding_model_name
)
vector_db = Chroma(
	persist_directory=persist_directory, 
	embedding_function=embedding_model, 
	collection_name=collection_name
)

tools = [rag, planning, search]
model = get_llm(
	port=2026, 
	host="127.0.0.1", 
	temperature=0, 
	openai_api_key="chwenjun225",
	model_name="1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"
)

checkpointer=MemorySaver()
store = InMemoryStore()

agent_exe = create_react_agent(
	version="v1", 
	model=model, 
	tools=tools, 
	store=store, 
    prompt=system_prompt, 
	checkpointer=checkpointer
)

# TODO
final_state = agent_exe.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Vietnam?"}]}, 
    config={"configurable": {"thread_id": 42}}
)
print(final_state["messages"][-1].content)

# with open("chroma_logs.txt", "a") as f:
	# sys.stdout = f  # Chuyển tất cả print() vào file
	# llm = get_llm(
	# 	port=2026, host="127.0.0.1", openai_api_key="chwenjun225", 
	# 	model_name="1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct", 
	# 	temperature=0
	# )
	# search = TavilySearchResults(max_results=2)
	# tools = [search]
	# model_with_tools = llm.bind_tools(tools)

	# # user_input = input(">>> 👨 User: ") 
	# user_input = "What's the difference between revenue year of 2023 and 2024?"
	# # user_input = "Hello"
	# # 🔍 Truy vấn RAG từ ChromaDB
	# rag_output = rag(query=user_input, num_retrieved_docs=1)
	# # print(f">>> <rag_output>{rag_output}</rag_output>")

	# # 🧠 Lập kế hoạch phản hồi từ Planning Module
	# planning_module_output = planning(user_input, rag_output)
	# # print(f">>> <planning_module>{planning_module_output}</planning_module>")

	# # 🤖 Gửi vào LLM để nhận phản hồi
	# response_template = ChatPromptTemplate.from_messages([
	# 	("system", "You are a friendly and helpful assistant. Your job is to answer human questions with care and detail. Keep your answers short and concise when possible."),
	# 	("user", "{input}")
	# ])
	# formatted_response_template = response_template.invoke({"input": planning_module_output})
	# response = llm.invoke(formatted_response_template)
	# assistant_response = response.content
	# print(f">>> 🤖 Assistant:\n{assistant_response}")

	# # 📝 Lưu lịch sử hội thoại vào ChromaDB
	# save_chat_history_to_chromadb(user_input, assistant_response)
	# sys.stdout = sys.__stdout__  # Reset lại stdout về mặc định
