import sys

import chromadb
import chromadb.errors

from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader 

def show_all_collections_data(host, port):
    """Hiển thị dữ liệu của tất cả collections trong ChromaDB."""
    client = chromadb.HttpClient(host=host, port=port)
    collections = client.list_collections()
    print(">>> 📂 Danh sách Collections trong ChromaDB:")
    for collection_name in collections:
        print(f">>> 🔍 Dữ liệu trong Collection: {collection_name}")
        collection = client.get_collection(name=collection_name)
        data = collection.get()
        for i in range(len(data['ids'])):
            print(f">>> 🆔 ID: {data['ids'][i]}")
            print(f">>> 📄 Document: {data['documents'][i]}")
            print(f">>> 📌 Metadata: {data['metadatas'][i]}")

def show_collections(host, port):
	"""Hiển thị danh sách Collections trong ChromaDB."""
	client = chromadb.HttpClient(host=host, port=port)
	collections = client.list_collections()
	print(">>> Danh sách các Collections:")
	for collection in collections:
		print(f">>> - {collection}")

def show_collection_data(
		host, 
		port, 
		collection_name
	):
	"""Hiển thị dữ liệu bên trong một collection."""
	client = chromadb.HttpClient(host=host, port=port)
	try:
		collection = client.get_collection(name=collection_name)
		data = collection.get()
		print(f">>> Dữ liệu bên trong Collection: {collection_name}")
		for i in range(len(data['ids'])):
			print(f">>> ID: {data['ids'][i]}")
			print(f">>> Document: {data['documents'][i]}")
			print(f">>> Metadata: {data['metadatas'][i]}")
	except chromadb.errors.InvalidCollectionException as e:
		print(e) 
		pass 

def upload_data_to_server(
		host, 
		port,
		chunk_size=500, # Số ký tự trong mỗi đoạn 
		chunk_overlap=50, # Độ chồng lấn trong mỗi đoạn 
		embedding_model="sentence-transformers/all-MiniLM-L6-v2", 
		name_of_collection="state_of_the_union", 
		file_path="/home/chwenjun225/Projects/Foxer/datasets/state_of_the_union.txt", 
		persist_directory="/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
	):
	"""Lưu dữ liệu & vector embeddings vào ChromaDB."""
	client = chromadb.HttpClient(host=host, port=port)
	collection = client.get_or_create_collection(name=name_of_collection)
	loader = TextLoader(file_path)
	documents = loader.load()
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size, 
		chunk_overlap=chunk_overlap, 
	)
	split_docs = text_splitter.split_documents(documents)
	embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
	vector_db = Chroma.from_documents(
		documents=split_docs, 
		embedding=embedding_model, 
		persist_directory=persist_directory
	)
	ids = [f"doc_{i}" for i in range(len(split_docs))]
	documents = [doc.page_content for doc in split_docs]
	metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(split_docs))]
	collection.add(
		ids=ids, 
		documents=documents, 
		metadatas=metadatas, 
	)
	print(f">>> Successfully loaded {len(split_docs)} data & vector to collection: {name_of_collection}")

if __name__ == "__main__":
	HOST, PORT = "127.0.0.1", 2027
	RUN = "show_all_collections_data"
	with open("chroma_logs.txt", "a") as f:
		sys.stdout = f  
		if "show_collections" == RUN:
			show_collections(host=HOST, port=PORT)
		if "show_all_collections_data" == RUN:
			show_all_collections_data(port=PORT, host=HOST)
		sys.stdout = sys.__stdout__
