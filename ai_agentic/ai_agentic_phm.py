from openai import OpenAI

from langchain import hub 
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_openai import ChatOpenAI

from typing_extensions import List, TypedDict

# TODO: Xem và đọc thật kỹ tài liệu triển khai RAG 

class State(TypedDict):
	question: str
	context: List[Document]
	answer: str

def get_llm(port, host, openai_api_key, model_name, temperature):
	"""Mô hình ngôn ngữ lớn."""
	openai_api_base = "http://" + str(host) + ":" + str(port) 
	return ChatOpenAI(
		model=model_name, 
		openai_api_base=openai_api_base, 
		openai_api_key=openai_api_key, 
		temperature=temperature
	)

def get_retriever(num_relate_docs, embedding_model_name, persist_directory):
	# TODO: Sửa lại hàm này: 
	# 1. Làm sao để kết nối với ChromaDB. 
	# 2. Thực hiện similarity_search trong ChromaDB
	embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
	vector_db = Chroma(
		persist_directory=persist_directory, 
		embedding=embedding_model, 
	)
	retriever = vector_db.as_retriever(search_kwargs={"k": num_relate_docs})
	return retriever

def llm_chain(
		port, 
		host, 
		openai_api_key, 
		model_name, 
		temperature, 
		num_relate_docs, 
		embedding_model_name, 
		persist_directory, 
		output_key,
		memory_key,
		return_messages
	):
	"""Let's go..."""
	llm = get_llm(
		port=port, 
		host=host, 
		openai_api_key=openai_api_key, 
		model_name=model_name, 
		temperature=temperature
	)
	retriever = get_retriever(
		num_relate_docs=num_relate_docs,
		embedding_model_name=embedding_model_name, 
		persist_directory=persist_directory
	)
	memory = ConversationBufferMemory(
		output_key=output_key,
		memory_key=memory_key, 
		return_messages=True
	)
	chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=return_messages
    )
	return chain 

if __name__ == "__main__":
	chain = llm_chain(
		port=2026, 
		host="127.0.0.1", 
		openai_api_key="chwenjun225", 
		model_name="1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct", 
		temperature=0, 
		num_relate_docs=3, 
		embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
		persist_directory="/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db", 
		output_key="answer",
		memory_key="chat_history",
		return_messages=True
	)
	while True:
		user_input = input(">>> 👨‍💻 User: ")
		if user_input.lower() == "exit":
			break
	
		response = chain(
			{"question": user_input, "chat_history": memory.chat_memory.messages}
		)
		assistant_response = response["answer"]
	
		print(">>> 🤖 Assistant: ", assistant_response)
