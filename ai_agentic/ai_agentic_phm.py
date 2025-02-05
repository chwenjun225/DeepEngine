from pprint import pprint

import chromadb 
from langchain.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from openai import OpenAI

# 1. Kết nối tới ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db") 
collection = chroma_client.get_or_create_collection(name="documents")

# 2. Tải tài liệu từ file 
loader = TextLoader("../state_of_the_union.txt")
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

# 5. Kết nối đến deepseek-r1-host-server 
PATH_MODEL = "/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"
client = OpenAI(
	base_url="http://localhost:2025/v1",
	api_key="chwenjun225",
)

# 6. Truy vấn từ ChromaDB
query = "What did the President say about the economy?"
# Tìm 3 đoạn văn bản liên quan nhất
retrieved_docs = vector_db.similarity_search(query, k=3)  
# Kết hợp đoạn văn bản
context = "\n".join([doc.page_content for doc in retrieved_docs])  

### 7️⃣ Gửi truy vấn đến mô hình AI cục bộ
completion = client.chat.completions.create(
    model=PATH_MODEL,
    messages=[
        {"role": "system", "content": "You are an expert assistant."},
        {"role": "user", "content": f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"}
    ]
)

print("\n✅ >>> AI Response:")
pprint(completion.choices[0].message)












