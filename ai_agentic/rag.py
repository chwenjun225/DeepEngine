from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persist_directory = "/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
collection_name = "foxconn_ai_research"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

embeddings  = HuggingFaceEmbeddings(model_name=embedding_model)
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model, 
    collection_name=collection_name
)

medical_records_store = vector_db.from_documents([], embeddings )
medical_records_retriever = medical_records_store.as_retriever()

insurance_faqs_store = vector_db.from_documents([], embeddings )
insurance_faqs_retriever = insurance_faqs_store.as_retriever()

def rag(query, num_retrieved_docs=3):
	"""Truy vấn RAG từ ChromaDB."""
	retriever_docs = vector_db.similarity_search(query, k=num_retrieved_docs)
	retrieved_texts = "\n".join([retriever_doc.page_content for retriever_doc in retriever_docs])
	return retrieved_texts