import chromadb

from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader 

def show_collections(host="127.0.0.1", port=2027):
	"""Hiển thị danh sách collections hiện có trong server."""
	client = chromadb.HttpClient(host="127.0.0.1", port=2027)
	print("List of collections:")
	for collection in client.list_collections():
		print(f"- {collection}")

def upload_data_to_server(
		host="127.0.0.1", 
		port=2027,
		chunk_size=500, # Số ký tự trong mỗi đoạn 
		chunk_overlap=50, # Độ chồng lấn trong mỗi đoạn 
		embedding_model="sentence-transformers/all-MiniLM-L6-v2", 
		name_of_collection="state_of_the_union", 
		file_path="/home/chwenjun225/Projects/Foxer/datasets/state_of_the_union.txt", 
		persist_directory="/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
	):
	"""Tải dữ liệu lên chroma_db."""
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
	collection.add(
		ids=ids, 
		documents=documents, 
		metadatas=[{"source": file_path} for _ in split_docs], 
	)
	print(f">>> Đã tải {len(split_docs)} đoạn văn bản vào collection_name: {name_of_collection}")

if __name__ == "__main__":
	upload_data_to_server()
	show_collections()
