import chromadb

from langchain_community.document_loaders import TextLoader 

collections = None

def list_collections():
	"""Hiển thị collections hiện có trong server."""
	client = chromadb.HttpClient(host="127.0.0.1", port=2027)
	print("List of collections:")
	for collection in client.list_collections():
		print(f"- {collection}")

def upload_data_to_server():
	"""Tải dữ liệu lên máy chủ lữu trữ."""
	global collection
	client = chromadb.HttpClient(host='127.0.0.1', port=2027)
	collection = client.get_or_create_collection(name="fulian_b09_phm_data")
	collection.add(
        ids=["sensor_1", "sensor_2"],
        documents=["Temperature sensor data", "Vibration sensor data"], 
        metadatas=[{"status": "normal"}, {"status": "warning"}],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
	)
	print(">>> Đã load dữ liệu vào Chroma thành công.")

def query_data_from_chroma():
	"""Truy vấn dữ liệu dựa trên embeddings."""
	results = collection.query(
		query_embeddings=[[0.1, 0.2, 0.3]]
	)
	print(f">>> Kết quả truy vấn: {results}")

if __name__ == "__main__":
	loader = TextLoader("../datasets/state_of_the_union.txt")
	documents = loader.load()
	print(">>> DEBUG")