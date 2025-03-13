import chromadb
import chromadb.errors



from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader 



def show_all_collections_data(host: str, port: str) -> None:
	"""Hiển thị dữ liệu của tất cả collections trong ChromaDB."""
	client = chromadb.HttpClient(host=host, port=port)
	collections = client.list_collections()
	print(">>> Danh sách Collections trong ChromaDB:")
	for collection_name in collections:
		print(f">>> Dữ liệu trong Collection: {collection_name}")
		collection = client.get_collection(name=collection_name)
		data = collection.get()
		for i in range(len(data['ids'])):
			print(      f">>> ID: {data['ids'][i]}"      )
			print(f">>> Document: {data['documents'][i]}")
			print(f">>> Metadata: {data['metadatas'][i]}")



def show_collections(host: str, port: str) -> None:
	"""Hiển thị danh sách Collections trong ChromaDB."""
	client = chromadb.HttpClient(host=host, port=port)
	collections = client.list_collections()
	print(">>> Danh sách các Collections:")
	for collection in collections:
		print(f">>> - {collection}")



def show_collection_data(host: str, port: str, collection_name: str) -> None:
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
	except Exception as e:
		raise e



def upload_data_to_server(
		host: str, 
		port: str,
		chunk_size: int = 500, 
		chunk_overlap: int = 50, 
		embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
		name_of_collection: str = "FOXCONN-FULIAN-B09-AI-Research-陳文俊-V1047876", 
		file_path: str = "/home/chwenjun225/Projects/Foxer/datasets/state_of_the_union.txt", # TODO: Sửa lại sau 
		persist_directory: str = "/home/chwenjun225/projects/DeepEngine/src/DeepEngine/chromadb_storage"
	) -> None:
	"""Lưu dữ liệu & vector embeddings vào ChromaDB."""
	client = chromadb.HttpClient(host=host, port=port)
	collection = client.get_or_create_collection(name=name_of_collection)
	loader = TextLoader(file_path)
	documents = loader.load()
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size, 
		chunk_overlap=chunk_overlap
	)
	split_docs = text_splitter.split_documents(documents)
	embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
	vector_db = Chroma.from_documents(
		documents=split_docs, 
		embedding=embedding_model, 
		persist_directory=persist_directory, 
		collection_name=name_of_collection
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



def remove_collection(host: str, port: str, collection_name: str) -> None:
	"""Xóa collection được chỉ định."""
	client = chromadb.HttpClient(host=host, port=port)
	collections = client.list_collections()
	print(f">>> Danh sách collections trước khi xóa: {collections}")
	if collection_name in collections:
		try:
			client.delete_collection(name=collection_name)
		except Exception as e:
			print(f">>> Không thể xóa Collection {collection_name}. \nVì: {e}")



if __name__ == "__main__":
	HOST, PORT = "127.0.0.1", 2027
	RUN = "show_collections"
	if "show_collections" == RUN:
		show_collections(host=HOST, port=PORT)

	elif "show_all_collections_data" == RUN:
		show_all_collections_data(port=PORT, host=HOST)

	elif "remove_collection":
		remove_collection(port=PORT, host=HOST, collection_name="langchain")
