import json 
from langchain_core.tools import tool

@tool
def recommend_maintenance_strategy(input_str=""):
	"""Đề xuất chiến lược bảo trì."""
	result = {
			"strategy": ["Preventive Maintenance"], 
			"justification": ["Failure probability is 0.03"]
		}
	return json.dump(result, indent=4)

@tool
def diagnose_fault_of_machine(input_str=""):
	"""Chẩn đoán lỗi máy móc."""
	result = {
			"fault": ["Overheating", "Coolant System Failure"], 
			"recommendation": ["Reduce workload and check cooling system"]
		}
	return json.dump(result, indent=2)

@tool
def remaining_useful_life_prediction(input_str=""):
	"""Dự đoán tuổi thọ còn lại của một thành phần thiết bị."""
	result = {
		"predicted_rul": [150],  
		"confidence": [0.85], 
		"recommendations": ["Reduce operating load to extend lifespan"]
	}
	return json.dumps(result, indent=2)

@tool
def rag(query, vector_db, num_retrieved_docs=3):
	"""Truy vấn RAG từ ChromaDB."""
	retriever_docs = vector_db.similarity_search(query, k=num_retrieved_docs)
	retrieved_texts = "\n".join([retriever_doc.page_content for retriever_doc in retriever_docs])
	return retrieved_texts

@tool
def planning(prompt_user, prompt_planning, rag_output):
	"""Xử lý thông tin từ RAG để tạo prompt phù hợp cho LLM."""
	planning_prompt = prompt_planning.invoke({"question": prompt_user, "context": rag_output})
	return planning_prompt

@tool
def add(a: int, b: int) -> int:
	"""Add two integers.

	Args:
		a: First integer
		b: Second integer
	"""
	return a + b

@tool
def multiply(a: int, b: int) -> int:
	"""Multiply two integers.

	Args:
		a: First integer
		b: Second integer
	"""
	return a * b

