from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIRECTORY = "/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
COLLECTION_NAME = "foxconn_ai_research"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FUNCTION  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
VECTOR_DB = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=EMBEDDINGS_FUNCTION, collection_name=COLLECTION_NAME)

def retriever(query, num_retrieved_docs=3):
	"""Truy vấn RAG từ ChromaDB."""
	retriever_documents = VECTOR_DB.similarity_search(
		query=query, 
		k=num_retrieved_docs
	)
	retrieved_texts = "\n".join([
		retriever_doc.page_content 
		for retriever_doc 
		in retriever_documents
	])
	return retrieved_texts
