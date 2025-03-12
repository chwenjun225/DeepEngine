import argparse
import os
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from pathlib import Path
from tqdm import tqdm



def get_chroma_client(persistent_path: str):
	"""Khởi tạo ChromaDB Client với chế độ Persistent."""
	return chromadb.PersistentClient(path=persistent_path)



def read_text_file(file_path: str) -> str:
	"""Đọc file văn bản TXT/CSV."""
	try:
		with open(file_path, "r", encoding="utf-8") as file:
			return file.read()
	except Exception as e:
		print(f">>> Lỗi khi đọc file {file_path}: {e}")
		return None



def read_pdf_file(file_path: str) -> str:
	"""Đọc file PDF."""
	try:
		pdf_reader = PdfReader(file_path)
		return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
	except Exception as e:
		print(f">>> Lỗi khi đọc PDF {file_path}: {e}")
		return None



def read_parquet_file(file_path: str) -> str:
	"""Đọc file Parquet."""
	try:
		df = pd.read_parquet(file_path)
		return json.dumps(df.to_dict(orient="records"), ensure_ascii=False)
	except Exception as e:
		print(f">>> Lỗi khi đọc Parquet {file_path}: {e}")
		return None



def process_file(file_path):
	"""Xử lý file để trích xuất văn bản."""
	ext = file_path.suffix.lower()
	try:
		if ext == ".txt":
			return read_text_file(file_path)
		elif ext == ".pdf":
			return read_pdf_file(file_path)
		elif ext == ".csv":
			return read_text_file(file_path) 
		elif ext == ".parquet":
			return read_parquet_file(file_path)
		else:
			print(f"⚠️ Không hỗ trợ định dạng: {ext}")
			return None
	except Exception as e:
		print(f"❌ Lỗi khi xử lý file {file_path.name}: {e}")
		return None



def add_documents(client: chromadb, collection_name: str, docs_path: str):
	"""Thêm tài liệu vào database."""
	collection = client.get_or_create_collection(name=collection_name)
	docs_path = Path(docs_path)
	if docs_path.is_file():
		files = [docs_path]
	elif docs_path.is_dir():
		files = list(docs_path.glob("*"))
	else:
		print(f"❌ Đường dẫn không hợp lệ: {docs_path}")
		return

	print(f"📂 Đang thêm {len(files)} file vào ChromaDB...")

	for file_path in tqdm(files):
		text = process_file(file_path)
		if not text:
			print(f"⚠️ Bỏ qua {file_path.name} vì dữ liệu rỗng hoặc lỗi khi đọc file.")
			continue
		if not isinstance(text, str):
			print(f"⚠️ Bỏ qua {file_path.name} vì dữ liệu không phải kiểu `str`.")
			continue
		doc_id = str(file_path.stem) 
		collection.add(ids=[doc_id], documents=[text])
		print(f"✅ Đã thêm: {file_path.name}")



def get_document(client, collection_name, doc_id):
	"""Truy vấn tài liệu theo ID."""
	collection = client.get_or_create_collection(name=collection_name)
	result = collection.get(ids=[doc_id])
	if result["documents"]:
		print(f"🔍 Tài liệu {doc_id}: {result['documents'][0]}")
	else:
		print(f"⚠️ Không tìm thấy tài liệu với ID: {doc_id}")



def list_documents(client, collection_name):
	"""Liệt kê tất cả tài liệu."""
	collection = client.get_or_create_collection(name=collection_name)
	result = collection.get()
	if result["ids"]:
		print("📄 Danh sách tài liệu:")
		for i, doc in enumerate(result["documents"]):
			print(f"{i + 1}. {result['ids'][i]}: {doc[:200]}...") 
	else:
		print("⚠️ Collection hiện đang trống.")



def delete_document(client: chromadb, collection_name, doc_id):
	"""Xóa tài liệu theo ID."""
	collection = client.get_or_create_collection(name=collection_name)
	collection.delete(ids=[doc_id])
	print(f"🗑️ Đã xóa tài liệu có ID: {doc_id}")



def main():
	"""Xử lý tham số đầu vào từ terminal."""
	parser = argparse.ArgumentParser(description="ChromaDB CLI Interface")

	parser.add_argument("--persistent-path", type=str, default="./chromadb_storage", help="Đường dẫn lưu trữ persistent")

	parser.add_argument("--add-doc", type=str, help="Thêm tài liệu từ file hoặc thư mục")
	parser.add_argument("--get-doc", action="store_true", help="Lấy tài liệu theo ID")
	parser.add_argument("--list-docs", action="store_true", help="Liệt kê tất cả tài liệu")
	parser.add_argument("--delete-doc", action="store_true", help="Xóa tài liệu theo ID")
	parser.add_argument("--id", type=str, help="ID của tài liệu để lấy hoặc xóa")

	args = parser.parse_args()

	client = get_chroma_client(args.persistent_path)
	collection_name = "my_collection"

	if args.add_doc:
		add_documents(client, collection_name, args.add_doc)
	elif args.get_doc and args.id:
		get_document(client, collection_name, args.id)
	elif args.list_docs:
		list_documents(client, collection_name)
	elif args.delete_doc and args.id:
		delete_document(client, collection_name, args.id)
	else:
		parser.print_help()

if __name__ == "__main__":
	main()
