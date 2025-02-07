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
