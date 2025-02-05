import chromadb 
from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from openai import OpenAI

# 1. Kết nối tới ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db") 
collection = chroma_client.get_or_create_collection(name="documents")

# 2. Tải tài liệu từ file 
loader = TextLoader("../datasets/state_of_the_union.txt")
documents = loader.load()

# 3. Chia nhỏ tài liệu thành các đoạn nhỏ 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# 4. Chuyển đổi đoạn văn thành vector embeddings & lưu vào ChromaDB 
embedding_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(
    documents=documents, 
    embedding=embedding_func, 
    persist_directory="./chroma_db"
)

print("✅ >>> Dữ liệu đã được lưu vào ChromaDB!")