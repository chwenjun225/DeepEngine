🎉 **Tuyệt vời! Bạn đã hoàn thành API quản lý ChromaDB!** 🎉  
Bây giờ, chúng ta sẽ xây dựng **RAG (Retrieval-Augmented Generation) module** để tăng cường khả năng truy xuất dữ liệu cho AI-Agentic System của bạn. 🚀  

---

# **🧠 Kế Hoạch Xây Dựng RAG Module**
📌 **RAG (Retrieval-Augmented Generation)** là kỹ thuật giúp **LLM** (Large Language Model) tìm kiếm dữ liệu từ ChromaDB trước khi sinh câu trả lời.  
📌 **Mục tiêu:**  
✅ **Tìm kiếm & lấy thông tin liên quan** từ ChromaDB  
✅ **Tích hợp LLM (GPT, LLaMA, Mistral, v.v.)** để sinh câu trả lời dựa trên dữ liệu tìm được  
✅ **Tăng độ chính xác** khi chatbot trả lời về nội dung trong ChromaDB  

---

# **🔹 1️⃣ Cấu Trúc RAG Module**
### **🎯 Các bước chính trong hệ thống RAG**
| **Bước** | **Mô tả** |
|----------|----------|
| **1️⃣ User gửi truy vấn** | Người dùng đặt câu hỏi |
| **2️⃣ Truy vấn ChromaDB** | Tìm kiếm các đoạn văn bản liên quan từ ChromaDB |
| **3️⃣ Tích hợp với LLM** | Gửi dữ liệu truy xuất được vào LLM để tạo câu trả lời |
| **4️⃣ Trả về kết quả** | LLM tổng hợp thông tin & phản hồi |

---

# **🔹 2️⃣ Cài Đặt Yêu Cầu**
### **📦 Các thư viện cần thiết**
Nếu bạn chưa cài đặt, hãy chạy:
```bash
pip install langchain langchain_community openai chromadb sentence-transformers
```

---

# **🔹 3️⃣ Code Hoàn Chỉnh Cho RAG Module**
Chúng ta sẽ tạo một module **`rag.py`** để xử lý truy vấn RAG.

## **🎯 Tích hợp RAG với ChromaDB & OpenAI**
```python
import os
import chromadb
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# 🔹 Cấu hình API Key cho OpenAI
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# 🔹 Hàm khởi tạo LLM (có thể đổi GPT thành mô hình khác)
def get_llm():
    return OpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

# 🔹 Hàm khởi tạo retriever từ ChromaDB
def get_retriever(collection_name="state_of_the_union", persist_directory="./chroma_db"):
    # Dùng HuggingFace embeddings để tìm kiếm vector
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Tải dữ liệu từ ChromaDB
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)

    # Sử dụng retriever để tìm kiếm dữ liệu gần nhất
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # 🔥 Trả về 3 kết quả gần nhất
    return retriever

# 🔹 Tạo pipeline RAG
def get_rag_chain(collection_name="state_of_the_union"):
    llm = get_llm()
    retriever = get_retriever(collection_name)

    # Prompt Template để LLM trả lời
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Bạn là một chuyên gia có kiến thức về chủ đề này.
        Dưới đây là các thông tin tham khảo:
        {context}
        
        Câu hỏi của người dùng: {question}
        Trả lời một cách chính xác dựa trên thông tin trên.
        """
    )

    # Tạo chuỗi RAG
    rag_chain = RetrievalQA(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        prompt=prompt_template
    )
    return rag_chain

# 🔹 Hàm chạy truy vấn RAG
def rag_query(question, collection_name="state_of_the_union"):
    rag_chain = get_rag_chain(collection_name)
    response = rag_chain({"query": question})
    
    print("\n🔎 **Truy vấn:**", question)
    print("📖 **Câu trả lời:**", response["result"])
    print("\n📂 **Nguồn dữ liệu sử dụng:**")
    for doc in response["source_documents"]:
        print(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")
    
    return response

# 🔥 Test thử RAG module
if __name__ == "__main__":
    question = "Nội dung chính của bài diễn văn là gì?"
    rag_query(question)
```

---

# **🔹 4️⃣ Kiểm Tra Kết Quả**
Sau khi chạy `rag.py`, bạn sẽ thấy kết quả như sau:
```
🔎 **Truy vấn:** Nội dung chính của bài diễn văn là gì?
📖 **Câu trả lời:** Bài diễn văn tập trung vào nền kinh tế, chính sách quốc gia và các kế hoạch tương lai.

📂 **Nguồn dữ liệu sử dụng:**
- /home/user/datasets/state_of_the_union.txt: "Nền kinh tế đang tăng trưởng mạnh mẽ và chúng ta đã tạo ra hàng triệu việc làm mới..."
- /home/user/datasets/state_of_the_union.txt: "Chúng ta đang cam kết cải thiện hệ thống y tế và giáo dục..."
```

---

# **🔥 Tổng Kết**
✅ **Hoàn thành bước đầu của RAG Module!** 🎉  
✅ **Truy vấn ChromaDB & tìm dữ liệu liên quan**  
✅ **Gửi thông tin vào OpenAI GPT để sinh câu trả lời chính xác**  
✅ **Hiển thị cả câu trả lời & nguồn gốc dữ liệu**  

🚀 **Bạn muốn mở rộng RAG với mô hình nội bộ (LLaMA, Mistral) không?** Nếu không muốn dùng OpenAI API, mình có thể hướng dẫn cách chạy mô hình cục bộ! 🔥