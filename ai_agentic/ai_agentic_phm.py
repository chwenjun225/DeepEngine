import asyncio
import chromadb

from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 

from openai import OpenAI

chroma_client = chromadb.PersistentClient(path="/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db") 
collection = chroma_client.get_or_create_collection(name="documents")

loader = TextLoader("../datasets/state_of_the_union.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

embedding_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(
	documents=split_docs, 
	embedding=embedding_func, 
	persist_directory="./chroma_db"
)

print(">>> Datas aved to ChromaDB")

PATH_MODEL = "/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"
client = OpenAI(
	base_url="http://localhost:2026/v1",
	api_key="chwenjun225",
)

query = input("User: ")
if query != "":
	retrieved_docs = vector_db.similarity_search(query, k=1)  
	context = "\n".join([doc.page_content for doc in retrieved_docs])  
	n = 0
	completion = client.chat.completions.create(
		model=PATH_MODEL,
		messages=[
			{"role": "system", "content": "You are an expert assistant."},
			{"role": "user", "content": f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"}
		]
	)
	print("\n✅ >>> AI Response:")
	print(completion.choices[0].message.content)
	n += 1
