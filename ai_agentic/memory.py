from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Tạo bộ nhớ hội thoại
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Kết nối với ChromaDB để lưu trữ thông tin bảo trì
embedding_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_func)
client = ""
# Tạo chuỗi hỏi-đáp có bộ nhớ
qa_with_memory = ConversationalRetrievalChain.from_llm(
    llm=client,
    retriever=vector_db.as_retriever(),
    memory=memory
)

# Đưa vào lịch sử hội thoại & truy vấn mới
chat_history = []
query = "Thiết bị này đã được bảo trì lần cuối vào khi nào?"
response = qa_with_memory({"question": query, "chat_history": chat_history})

print("\n✅ >>> AI Memory Response:")
print(response)
