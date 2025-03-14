"""CLI to Interact with Chroma Vector Database."""



import os 
import json
import argparse



import chromadb
import chromadb.errors



from typing_extensions import List, Dict, Optional



from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredHTMLLoader, PyPDFLoader



START_SERVER = """
chroma run \
	--path "/home/chwenjun225/projects/DeepEngine/src/DeepEngine/chromadb_storage" \
	--host "127.0.0.1" \
	--port "2027" \
	--log-path "/home/chwenjun225/projects/DeepEngine/src/DeepEngine/chromadb_storage.log"
"""



ONLY_SHOW_COLLECTIONS_NAME = """
python database.py \
	--action only_show_collections_name \
	--persist_directory /home/chwenjun225/projects/DeepEngine/src/DeepEngine/chromadb_storage
"""


SHOW_COLLECTION_DATA = """
python database.py \
	--action show_collection_data \
	--collection_name foxconn_fulian_b09_ai_research_tranvantuan_v1047876 \
	--persist_directory /home/chwenjun225/projects/DeepEngine/src/DeepEngine/chromadb_storage \
	--output_file /home/chwenjun225/projects/DeepEngine/src/DeepEngine/output_file.json
"""



UPLOAD_DATA_TO_SERVER = """
python database.py \
	--action upload_data_to_server \
	--collection_name foxconn_fulian_b09_ai_research_tranvantuan_v1047876 \
	--file_path /home/chwenjun225/projects/DeepEngine/docs/pdf/Hands-On-ML-sklearn-keras-tf-2nd-Edition-Aurelien-Geron.pdf \
	--persist_directory /home/chwenjun225/projects/DeepEngine/src/DeepEngine/chromadb_storage
"""



REMOVE_COLLECTION = """
python database.py \
	--action remove_collection \
	--collection_name foxconn_fulian_b09_ai_research_tranvantuan_v1047876 \
	--persist_directory /home/chwenjun225/projects/DeepEngine/src/DeepEngine/chromadb_storage
"""



def get_chroma_client(
	chromadb_host: str = None, 
	chromadb_port: int = None, 
	persist_directory: str = None
):
	"""Khởi tạo ChromaDB client.
	- Sử dụng `HttpClient` nếu cung cấp `chromadb_host` + `chromadb_port` (kết nối máy chủ).
	- Sử dụng `PersistentClient` nếu có `persist_directory` (làm việc cục bộ).

	Args:
		chromadb_host (str, optional): Địa chỉ IP hoặc hostname của ChromaDB server.
		chromadb_port (int, optional): Cổng của ChromaDB server.
		persist_directory (str, optional): Đường dẫn đến thư mục lưu trữ ChromaDB cục bộ.

	Returns:
		chromadb.Client: Client ChromaDB phù hợp.

	Raises:
		ValueError: Nếu không có thông tin kết nối hợp lệ.
	"""
	if chromadb_host and chromadb_port:
		return chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
	if persist_directory:
		return chromadb.PersistentClient(path=persist_directory)
	
	raise ValueError("Cần cung cấp `chromadb_host` + `chromadb_port` hoặc `persist_directory`.")



def show_all_collections_data(
	chromadb_host: Optional[str] = "127.0.0.1",
	chromadb_port: Optional[int] = 2027,
	persist_directory: str = "/home/chwenjun225/projects/DeepEngine/src/DeepEngine/chromadb_storage", 
	output_file: str = "/home/chwenjun225/projects/DeepEngine/src/DeepEngine/all_collections.json"
) -> None:
	"""
	Hiển thị và xuất dữ liệu từ tất cả collections trong ChromaDB.
	- Hỗ trợ kết nối `HttpClient` (máy chủ) hoặc `PersistentClient` (cục bộ).
	- Hiển thị tổng quan về collections và tài liệu.
	- Hỗ trợ lưu dữ liệu vào file JSON.

	Args:
		chromadb_host (str, optional): Địa chỉ IP/hostname của ChromaDB server (mặc định: "127.0.0.1").
		chromadb_port (int, optional): Cổng của ChromaDB server (mặc định: 2027).
		persist_directory (str, optional): Đường dẫn ChromaDB cục bộ.
		output_file (str, optional): Xuất dữ liệu ra file JSON (mặc định: không xuất).

	Returns:
		None: Chỉ in dữ liệu ra terminal hoặc lưu vào JSON.
	"""
	if persist_directory and not os.path.exists(persist_directory):
		print(f">>> [Lỗi] Đường dẫn '{persist_directory}' không tồn tại.")
		return
	
	try: client = get_chroma_client(chromadb_host, chromadb_port, persist_directory)
	except Exception as e:
		print(f">>> [Lỗi] Không thể kết nối ChromaDB: {e}")
		return

	try: collections = client.list_collections()
	except Exception as e:
		print(f">>> [Lỗi] Không thể lấy danh sách collections: {e}")
		return
	
	if not collections:
		print(">>> Không có collections nào trong ChromaDB.")
		return

	print(f">>> Danh sách {len(collections)} collections trong ChromaDB:")
	all_data = {}
	for collection_name in collections:
		print(f"\n>>> Collection: {collection_name}")

		try:
			collection = client.get_collection(name=collection_name)
			data = collection.get()
			if not data["ids"]:
				print(f">>> Collection '{collection_name}' không có dữ liệu.")
				continue
		except Exception as e:
			print(f">>> [Lỗi] Không thể lấy dữ liệu từ '{collection_name}': {e}")
			continue
		collection_data = [{
				"id": data["ids"][i],
				"document": data["documents"][i],
				"metadata": data["metadatas"][i]
			} for i in range(len(data["ids"]))
		]

		all_data[collection_name] = collection_data
		print(f">>> Số lượng tài liệu: {len(collection_data)}")

		for doc in collection_data[:3]:
			print(f"\n>>> ID: {doc['id']}")
			print(f">>> Document: {doc['document'][:100]}...") 
			print(f">>> Metadata: {doc['metadata']}")
		if len(collection_data) > 3:
			print(f">>> ... (Đã rút gọn, tổng số tài liệu: {len(collection_data)})")

	if output_file:
		try:
			with open(output_file, "w", encoding="utf-8") as f:
				json.dump(all_data, f, indent=4, ensure_ascii=False)
			print(f"\n>>> Dữ liệu đã được lưu vào {output_file}")
		except Exception as e:
			print(f">>> [Lỗi] Không thể ghi vào file '{output_file}': {e}")
	print("\n>>> Hoàn thành việc hiển thị dữ liệu trong ChromaDB.")



def only_show_collections_name(
	chromadb_host: Optional[str] = "127.0.0.1",
	chromadb_port: Optional[int] = 2027,
	persist_directory: Optional[str] = None
) -> None:
	"""
	Hiển thị danh sách collections trong ChromaDB.

	- Hỗ trợ `HttpClient` (máy chủ) hoặc `PersistentClient` (cục bộ).
	- In danh sách collections hiện có.

	Args:
		chromadb_host (str, optional): Địa chỉ IP/hostname của ChromaDB server (mặc định: "127.0.0.1").
		chromadb_port (int, optional): Cổng của ChromaDB server (mặc định: 2027).
		persist_directory (str, optional): Đường dẫn đến thư mục lưu trữ ChromaDB cục bộ.

	Returns:
		None: Chỉ in dữ liệu ra terminal.
	"""
	try:
		client = get_chroma_client(chromadb_host, chromadb_port, persist_directory)
	except Exception as e:
		print(f">>> [Lỗi] Không thể kết nối ChromaDB: {e}")
		return

	try:
		collections = client.list_collections()
	except Exception as e:
		print(f">>> [Lỗi] Không thể lấy danh sách collections: {e}")
		return

	if not collections:
		print(">>> Không có collections nào trong ChromaDB.")
		return
	
	print(f">>> Danh sách {len(collections)} collections trong ChromaDB:")
	for collection in collections:
		print(f">>> - {collection}")
	print("\n>>> Hoàn thành việc hiển thị danh sách collections.")



def show_collection_data(
	chromadb_host: Optional[str] = "127.0.0.1",
	chromadb_port: Optional[int] = 2027,
	persist_directory: Optional[str] = None,
	collection_name: str = "foxconn_fulian_b09_ai_research_tranvantuan_v1047876",
	filter_key: Optional[str] = None,
	filter_value: Optional[str] = None,
	output_file: Optional[str] = None
) -> None:
	"""
	Hiển thị dữ liệu bên trong một collection của ChromaDB.

	- Hỗ trợ `HttpClient` (máy chủ) hoặc `PersistentClient` (cục bộ).
	- Lọc dữ liệu theo metadata (tùy chọn).
	- Xuất kết quả ra file JSON (tùy chọn).

	Args:
		chromadb_host (str, optional): Địa chỉ IP/hostname của ChromaDB server.
		chromadb_port (int, optional): Cổng của ChromaDB server.
		persist_directory (str, optional): Đường dẫn ChromaDB cục bộ.
		collection_name (str): Tên của collection cần lấy dữ liệu.
		filter_key (str, optional): Khóa metadata để lọc dữ liệu.
		filter_value (str, optional): Giá trị của metadata để lọc dữ liệu.
		output_file (str, optional): Xuất kết quả ra file JSON.

	Returns:
		None: Chỉ in dữ liệu ra terminal hoặc lưu vào JSON.
	"""
	if not collection_name:
		print(">>> [Lỗi] Bạn cần cung cấp tên collection (`collection_name`).")
		return

	try:
		client = get_chroma_client(chromadb_host, chromadb_port, persist_directory)
	except Exception as e:
		print(f">>> [Lỗi] Không thể kết nối ChromaDB: {e}")
		return

	try:
		collection = client.get_collection(name=collection_name)
		data = collection.get()
		if not data["ids"]:
			print(f">>> Collection '{collection_name}' không chứa dữ liệu.")
			return
	except Exception as e:
		print(f">>> [Lỗi] Không thể lấy dữ liệu từ collection '{collection_name}'. Vì: [{e}]")
		return

	filtered_data = [
		{
			"id": data["ids"][i],
			"document": data["documents"][i],
			"metadata": data["metadatas"][i]
		}
		for i in range(len(data["ids"]))
		if not filter_key or not filter_value or data["metadatas"][i].get(filter_key) == filter_value
	]

	print(f"\n>>> Dữ liệu trong Collection: {collection_name} (Tổng: {len(filtered_data)} tài liệu)")
	for doc in filtered_data[:3]:
		print(f"\n>>> ID: {doc['id']}")
		print(f">>> Document: {doc['document'][:100]}...") 
		print(f">>> Metadata: {doc['metadata']}")

	if len(filtered_data) > 3:
		print(f">>> ... (Đã rút gọn, tổng số tài liệu: {len(filtered_data)})")

	if output_file:
		try:
			with open(output_file, "w", encoding="utf-8") as f:
				json.dump(filtered_data, f, indent=4, ensure_ascii=False)
			print(f"\n>>> Dữ liệu đã được lưu vào {output_file}")
		except Exception as e:
			print(f">>> [Lỗi] Không thể ghi vào file '{output_file}': {e}")
	print("\n>>> Hoàn thành việc hiển thị dữ liệu trong collection.")



def clean_text(text: str) -> str:
	"""Làm sạch văn bản trước khi lưu vào ChromaDB."""
	text = text.replace("\n", " ").strip()
	text = " ".join(text.split())
	return text



def upload_data_to_server(
	DEBUG: Optional[bool] = False,
	chromadb_host: Optional[str] = "127.0.0.1",
	chromadb_port: Optional[int] = 2027,
	persist_directory: Optional[str] = None,
	name_of_collection: Optional[str] = None,
	file_path: Optional[str] = None,
	embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
	chunk_size: int = 500,
	chunk_overlap: int = 50
) -> None:
	"""
	Tải dữ liệu văn bản và vector embeddings lên ChromaDB.

	- Đọc dữ liệu từ file văn bản, chia nhỏ thành đoạn (`chunks`).
	- Tạo vector embeddings bằng mô hình `HuggingFaceEmbeddings`.
	- Lưu trữ vào collection của ChromaDB.

	Args:
		chromadb_host (str, optional): Địa chỉ IP/hostname của ChromaDB server.
		chromadb_port (int, optional): Cổng của ChromaDB server.
		persist_directory (str, optional): Đường dẫn thư mục lưu trữ ChromaDB cục bộ.
		name_of_collection (str): Tên collection để lưu dữ liệu.
		file_path (str): Đường dẫn file văn bản cần xử lý.
		embedding_model (str, optional): Mô hình nhúng văn bản (mặc định: `"sentence-transformers/all-MiniLM-L6-v2"`).
		chunk_size (int, optional): Kích thước mỗi đoạn văn bản (mặc định: `500`).
		chunk_overlap (int, optional): Số ký tự trùng lặp giữa các đoạn văn bản (mặc định: `50`).

	Returns:
		None: Chỉ in thông tin ra terminal.
	"""
	if not name_of_collection:
		print(">>> [Lỗi] Bạn cần cung cấp tên collection (`name_of_collection`).")
		return
	if not file_path or not os.path.exists(file_path):
		print(f">>> [Lỗi] File '{file_path}' không tồn tại hoặc không hợp lệ.")
		return


	chunk_data = process_document(file_path, embedding_model, chunk_size, chunk_overlap)
	if chunk_data is None or len(chunk_data) == 0:
		print(">>> [Lỗi] Không thể xử lý tài liệu hoặc tài liệu rỗng.")
		return


	try: 
		client = get_chroma_client(chromadb_host, chromadb_port, persist_directory)
		collection = client.get_or_create_collection(name=name_of_collection)
	except Exception as e:
		print(f">>> [Lỗi] Không thể kết nối hoặc tạo collection '{name_of_collection}'. Vì [{e}]")
		return


	print(f">>> Đang lưu {len(chunk_data)} chunks vào collection '{name_of_collection}' ...")


	valid_chunk_data = [{
		"id": doc["id"],
		"document": clean_text(str(doc["document"])) if isinstance(doc["document"], str) else "",
		"metadata": {"source": doc["metadata"]["source"], "chunk_id": int(doc["metadata"]["chunk_id"])}
	} for doc in chunk_data]


	if DEBUG:
		print(">>> Kiểm tra dữ liệu trước khi thêm vào ChromaDB:")
		print("IDs:", [doc["id"] for doc in valid_chunk_data])
		print("Documents:", [doc["document"] for doc in valid_chunk_data])
		print("Metadatas:", [doc["metadata"] for doc in valid_chunk_data])


	batch_size = 500
	for i in range(0, len(valid_chunk_data), batch_size):
		batch = valid_chunk_data[i:i + batch_size]
		try:
			collection.add(
				ids=[doc["id"] for doc in batch],
				documents=[doc["document"] for doc in batch],
				metadatas=[doc["metadata"] for doc in batch]
			)
			print(f">>> Đã lưu {len(batch)} chunk (từ {i} đến {i + len(batch)}) vào ChromaDB.")
		except Exception as e:
			print(f">>> [Lỗi] Không thể lưu batch {i}-{i + len(batch)} vào collection '{name_of_collection}': {e}")
	print(f">>> Successfully uploaded {len(chunk_data)} chunks into collection: {name_of_collection}")



def remove_collection(
	chromadb_host: Optional[str] = "127.0.0.1",
	chromadb_port: Optional[int] = 2027,
	persist_directory: Optional[str] = None,
	collection_name: Optional[str] = None
) -> None:
	"""Xóa một collection khỏi ChromaDB.

	- Hỗ trợ `HttpClient` (máy chủ) hoặc `PersistentClient` (cục bộ).
	- Kiểm tra trước khi xóa để tránh lỗi.

	Args:
		chromadb_host (str, optional): Địa chỉ IP/hostname của ChromaDB server.
		chromadb_port (int, optional): Cổng của ChromaDB server.
		persist_directory (str, optional): Đường dẫn lưu trữ ChromaDB cục bộ.
		collection_name (str): Tên collection cần xóa.

	Returns:
		None: Chỉ in kết quả ra terminal.
	"""
	if not collection_name:
		print(">>> [Lỗi] Bạn cần cung cấp tên collection (`collection_name`).")
		return

	try:
		client = get_chroma_client(chromadb_host, chromadb_port, persist_directory)
	except Exception as e:
		print(f">>> [Lỗi] Không thể kết nối ChromaDB: {e}")
		return

	try:
		collections = client.list_collections()
	except Exception as e:
		print(f">>> [Lỗi] Không thể lấy danh sách collections: {e}")
		return

	if collection_name not in collections:
		print(f">>> Collection '{collection_name}' không tồn tại.")
		return

	try:
		client.delete_collection(name=collection_name)
		print(f">>> Collection '{collection_name}' đã được xóa thành công.")
	except Exception as e:
		print(f">>> Không thể xóa Collection '{collection_name}'. Lỗi: {e}")



def process_document(
	file_path: str,
	embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
	chunk_size: int = 500,
	chunk_overlap: int = 50
) -> Optional[List[Dict]]:
	"""
	Xử lý tài liệu: đọc, chia nhỏ thành chunks và tạo embeddings.

	Args:
		file_path (str): Đường dẫn tài liệu.
		embedding_model (str, optional): Mô hình embeddings (mặc định: MiniLM-L6-v2).
		chunk_size (int, optional): Kích thước mỗi đoạn văn bản.
		chunk_overlap (int, optional): Số ký tự trùng lặp giữa các đoạn văn bản.

	Returns:
		List[Dict]: Danh sách các chunk với id, văn bản, và metadata.
	"""
	
	if not os.path.exists(file_path):
		print(f">>> [Lỗi] File '{file_path}' không tồn tại.")
		return None
	
	ext = os.path.splitext(file_path)[-1].lower()
	if ext == ".pdf":
		loader = PyPDFLoader(file_path)
	elif ext == ".txt":
		loader = TextLoader(file_path)
	elif ext == ".docx":
		loader = Docx2txtLoader(file_path)
	elif ext == ".html":
		loader = UnstructuredHTMLLoader(file_path)
	else:
		print(f">>> [Lỗi] Không hỗ trợ định dạng file: {ext}")
		return None

	print(f">>> Đang đọc tài liệu: {file_path} ...")
	documents = loader.load()
	
	print(f">>> Đang chia tài liệu thành chunks (size={chunk_size}, overlap={chunk_overlap}) ...")
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	split_docs = text_splitter.split_documents(documents)
	
	print(f">>> Đang tạo embeddings bằng mô hình: {embedding_model} ...")
	embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)

	chunk_data = [
		{
			"id": f"doc_{i}",
			"document": doc.page_content,
			"metadata": {"source": file_path, "chunk_id": i}
		}
		for i, doc in enumerate(split_docs)
	]
	return chunk_data



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="CLI to Interact with Chroma Vector Database.")

	parser.add_argument("--chromadb_host", type=str, help="ChromaDB Host (bắt buộc nếu không dùng persist_directory)")
	parser.add_argument("--chromadb_port", type=int, help="ChromaDB Port (bắt buộc nếu không dùng persist_directory)")
	parser.add_argument("--persist_directory", type=str, help="Đường dẫn đến ChromaDB cục bộ")
	

	parser.add_argument("--action", type=str, required=True, choices=[
		"show_all_collections_data",
		"only_show_collections_name",
		"show_collection_data",
		"upload_data_to_server",
		"remove_collection"
	], help="Chọn hành động để thực hiện")

	parser.add_argument("--collection_name", type=str, help="Tên Collection (bắt buộc cho hầu hết các thao tác).")
	parser.add_argument("--file_path", type=str, help="Đường dẫn file để upload.")
	parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Mô hình embedding (mặc định: MiniLM-L6-v2).")
	parser.add_argument("--chunk_size", type=int, default=500, help="Kích thước mỗi đoạn văn bản.")
	parser.add_argument("--chunk_overlap", type=int, default=50, help="Số ký tự trùng lặp giữa các đoạn văn bản.")
	parser.add_argument("--output_file", type=str, help="Đường dẫn đến ChromaDB cục bộ", default="/home/chwenjun225/projects/DeepEngine/src/DeepEngine/output_file.json")
	parser.add_argument("--debug", type=bool, default=False, help="Debug hiển thị dữ liệu được xử lý.")

	args = parser.parse_args()

	if not args.persist_directory and (not args.chromadb_host or not args.chromadb_port):
		print(">>> [Lỗi] Bạn cần cung cấp `persist_directory` hoặc `chromadb_host` + `chromadb_port`.")
		exit(1)

	if args.action == "show_all_collections_data":
		show_all_collections_data(
			chromadb_host=args.chromadb_host, 
			chromadb_port=args.chromadb_port,
			persist_directory=args.persist_directory, 
			output_file=args.output_file
		)

	elif args.action == "only_show_collections_name":
		only_show_collections_name(
			chromadb_host=args.chromadb_host, 
			chromadb_port=args.chromadb_port,
			persist_directory=args.persist_directory
		)

	elif args.action == "show_collection_data":
		if not args.collection_name:
			print(">>> [Lỗi] Vui lòng cung cấp `--collection_name`.")
			exit(1)
		show_collection_data(
			chromadb_host=args.chromadb_host, 
			chromadb_port=args.chromadb_port, 
			persist_directory=args.persist_directory,
			collection_name=args.collection_name
		)

	elif args.action == "upload_data_to_server":
		if not args.file_path or not args.collection_name or not args.persist_directory:
			print(">>> [Lỗi] Cần cung cấp `--file_path`, `--collection_name`, và `--persist_directory` để upload dữ liệu.")
			exit(1)
		upload_data_to_server(
			chromadb_host=args.chromadb_host, 
			chromadb_port=args.chromadb_port, 
			persist_directory=args.persist_directory,
			name_of_collection=args.collection_name,
			file_path=args.file_path,
			embedding_model=args.embedding_model,
			chunk_size=args.chunk_size,
			chunk_overlap=args.chunk_overlap,
			DEBUG=args.debug
		)

	elif args.action == "remove_collection":
		if not args.collection_name:
			print(">>> [Lỗi] Vui lòng cung cấp `--collection_name` để xóa Collection.")
			exit(1)
		remove_collection(
			chromadb_host=args.chromadb_host, 
			chromadb_port=args.chromadb_port, 
			persist_directory=args.persist_directory,
			collection_name=args.collection_name
		)
