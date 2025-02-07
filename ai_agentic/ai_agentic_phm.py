from openai import OpenAI

from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain.prompts import PromptTemplate

PATH_MODEL = "/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"

def get_retriever(
		collection_name="state_of_the_union", 
		persist_directory="./chroma_db"
	):
	"""Truy vấn dữ liệu từ ChromaDB."""
	embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
	vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
	retriever = vector_db.as_retriever(search_kwargs={"k": 3})
	return retriever

def get_rag_chain(collection_name="state_of_the_union"):
	"""RAG pipeline."""
	llm = get_llm()
	retriever = get_retriever(collection_name)

	prompt_template = PromptTemplate(
		input_variables=["context", "question"],
		template="""
		Bạn là một AI sử dụng mô hình DeepSeek-R1.
		Dưới đây là các thông tin từ database:
		{context}
		
		Câu hỏi của người dùng: {question}
		Trả lời dựa trên dữ liệu đã tìm được.
		"""
	)

	rag_chain = RetrievalQA(
		llm=llm,
		retriever=retriever,
		return_source_documents=True,
		prompt=prompt_template
	)
	return rag_chain

# 🔹 Chạy truy vấn RAG
def rag_query(question, collection_name="state_of_the_union"):
	rag_chain = get_rag_chain(collection_name)
	response = rag_chain({"query": question})

	print("\n🔎 **Truy vấn:**", question)
	print("📖 **Câu trả lời:**", response["result"])
	print("\n📂 **Nguồn dữ liệu sử dụng:**")
	for doc in response["source_documents"]:
		print(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")

	return response

def get_llm(
		path_model, 
		host, 
		port, 
		api_key
	):
	base_url = "http://" + str(host)+ ":" + str(port)
	client = OpenAI(base_url=base_url, api_key=api_key)

	completion = client.chat.completions.create(
		model=path_model,
		messages=[
			{"role": "system", "content": "You are an expert assistant."},
			{"role": "user", "content": "What is the different between America economy and China economy?"}
		]
	)
	print(">>> 🤖 AI Response:")
	print(completion.choices[0].message.content)

if __name__ == "__main__":
	get_llm(
		path_model=PATH_MODEL, 
		host="127.0.0.1", 
		port=2026, 
		api_key="Foxconn-AI Research", 
	)
