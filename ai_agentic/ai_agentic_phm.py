import sys
# from typing_extensions import List, TypedDict
from datetime import datetime

from openai import OpenAI # TODO: Sử dụng sau với multi-model language model
# from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

if "Khai báo RAG-Prompt":
	prompt = ChatPromptTemplate.from_messages([
		("human", "You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise. \
Question: {question} \
Context: {context} \
Answer:")
	]) 
if "Cấu hình cơ sở dữ liệu ChromaDB":
	PERSIST_DIRECTORY = "/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
	COLLECTION_NAME = "foxconn_ai_research"
	EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# if "Tạo bộ nhớ hội thoại":
# 	memory = ConversationBufferMemory(
# 		memory_key="chat_history", 
# 		return_messages=True
# 	)
if "Kết nối ChromaDB để lưu hội thoại và truy vấn RAG":
	embedding_model = HuggingFaceEmbeddings(
		model_name=EMBEDDING_MODEL_NAME
	)
	vector_db = Chroma(
		persist_directory=PERSIST_DIRECTORY,
		embedding_function=embedding_model,
		collection_name=COLLECTION_NAME
	)
# if "CACHE":
# 	CACHE = {}

# def cached_llm_response(query):
# 	"""
# 	Nếu câu hỏi giống 100% một truy vấn cũ, lấy kết quả từ cache thay vì gọi LLM.
# 	"""
# 	# TODO: Cache đến một giới hạn nào đó cần được xóa bớt.
# 	if query in CACHE: return CACHE[query]
# 	response = llm.invoke([{"role": "user", "content": query}])
# 	CACHE[query] = response.content
# 	return response.content

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

def save_chat_history_to_chroma_db(user_input, assistant_response):
	"""
	Lưu hội thoại vào ChromaDB.
	"""
	timestamp =  datetime.now().strftime("%Y-%m-%d_%H:%M:%S-%f")
	text_to_store = f"""[{timestamp}] 👨 User: {user_input}\n[{timestamp}] 🤖 Assistant: {assistant_response}"""
	vector_db.add_texts(
		texts=[text_to_store],
		metadatas=[{"source": "chat_history"}]
	)

def rag(query, num_retrieved_docs=3):
# TODO: 
	# 1. Thêm RAG-Prompt ("""Nếu không tìm thấy hãy nói rằng tôi không biết""")
	# 2. Cải Thiện Tốc Độ Truy Vấn RAG
	# Hiện tại: Hệ thống tìm kiếm trong ChromaDB bằng similarity_search(), có thể chậm khi dữ liệu lớn.
	# ✅ Giải pháp: Dùng FAISS Index hoặc Hybrid Search (tìm kiếm kết hợp từ khóa + vector).
	"""
	Truy vấn RAG từ ChromaDB.
	"""
# 	prompt = ChatPromptTemplate.from_messages([
# 		("human", """
# You are an assistant for question-answering tasks. 
# Use the following pieces of retrieved context to answer the question. 
# If you don't know the answer, just say that you don't know.
# Use three sentences maximum and keep the answer concise. 
# Question: {question} 
# Context: {context} 
# Answer:"""),
# 	])
	retriever_docs = vector_db.similarity_search(query, k=num_retrieved_docs)
	retrieved_texts = "\n".join([retriever_doc.page_content for retriever_doc in retriever_docs])
	return retrieved_texts

def planning_module(prompt_user, rag_output):
# TODO:
	# Hiện tại: Planning chỉ ghép prompt_user + rag_output.
	# ✅ Giải pháp:

	# Dùng prompt templates chuyên biệt để tối ưu phản hồi.
	# Sử dụng Chain of Thought (CoT) hoặc ReAct để tăng cường reasoning.
	# Nếu dùng LLM mạnh hơn, có thể áp dụng Multi-Step Planning.
	"""
	Xử lý thông tin từ RAG để tạo prompt phù hợp cho LLM.
	"""
	planning_prompt = prompt.invoke({
		"question": prompt_user,
		"context": rag_output
	})
# 	planning_prompt = f"""
# User Question: {prompt_user}
# Retrieved Knowledge from RAG:
# {rag_output}

# Generate the best response considering the retrieved knowledge. 
# Keep it concise yet informative.
# 	"""
	return planning_prompt

if __name__ == "__main__":
	with open("chroma_logs.txt", "a") as f:
		sys.stdout = f  # Chuyển tất cả print() vào file
		llm = get_llm(
			port=2026, 
			host="127.0.0.1", 
			openai_api_key="chwenjun225", 
			model_name="1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct", 
			temperature=0
		)
		# user_input = input(">>> 👨 User: ") 
		user_input = "What's the difference between revenue year of 2023 and 2024?"

		# 🔍 Truy vấn RAG từ ChromaDB
		rag_output = rag(query=user_input, num_retrieved_docs=1)
		print(f">>> <rag_output>{rag_output}</rag_output>")

		# 🧠 Lập kế hoạch phản hồi từ Planning Module
		planning_module_output = planning_module(user_input, rag_output)
		print(f">>> <planning_module>{planning_module_output}</planning_module>")

		# 🤖 Gửi vào LLM để nhận phản hồi
		response_template = ChatPromptTemplate.from_messages([
			("system", "You are a friendly and helpful assistant. \
	Your job is to answer human questions with care and detail. \
	Keep your answers short and concise when possible."),
			("user", "{input}")
		])
		formatted_response_template = response_template.invoke({"input": planning_module_output})
	# 	response = llm.invoke([
	# 		{"role": "system", "content": " You are a friendly and helpful assistant. \
	# Your job is to answer human questions with care and detail. \
	# Keep your answers short and concise when possible."}, 
	# 		{"role": "user", "content": planning_module_output}
	# 	])
		response = llm.invoke(formatted_response_template)
		assistant_response = response.content
		print(f">>> 🤖 Assistant:\n{assistant_response}")

		# 📝 Lưu lịch sử hội thoại vào ChromaDB
		save_chat_history_to_chroma_db(user_input, assistant_response)
		sys.stdout = sys.__stdout__  # Reset lại stdout về mặc định
