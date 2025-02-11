import json 
from datetime import datetime

def recommend_maintenance_strategy(input_str=""):
	"""Đề xuất chiến lược bảo trì."""
	result = {
			"strategy": ["Preventive Maintenance"], 
			"justification": ["Failure probability is 0.03"]
		}
	return json.dump(result, indent=4)

def diagnose_fault_of_machine(input_str=""):
	"""Chẩn đoán lỗi máy móc."""
	result = {
			"fault": ["Overheating", "Coolant System Failure"], 
			"recommendation": ["Reduce workload and check cooling system"]
		}
	return json.dump(result, indent=2)

def remaining_useful_life_prediction(input_str=""):
	"""Dự đoán tuổi thọ còn lại của một thành phần thiết bị."""
	result = {
		"predicted_rul": [150],  
		"confidence": [0.85], 
		"recommendations": ["Reduce operating load to extend lifespan"]
	}
	return json.dumps(result, indent=2)

def search(query: str):
    """Call to surf the web."""
    if "vn" in query.lower() or "Vietnam" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

def rag(query, vector_db, num_retrieved_docs=3):
	"""Truy vấn RAG từ ChromaDB."""
	retriever_docs = vector_db.similarity_search(query, k=num_retrieved_docs)
	retrieved_texts = "\n".join([retriever_doc.page_content for retriever_doc in retriever_docs])
	return retrieved_texts

def planning(prompt_user, prompt_planning, rag_output):
	"""Xử lý thông tin từ RAG để tạo prompt phù hợp cho LLM."""
	planning_prompt = prompt_planning.invoke({"question": prompt_user, "context": rag_output})
	return planning_prompt

def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

def check_weather(location: str, at_time: datetime | None = None) -> str:
	"""Return the weather forecast for the specified location."""
	return f"It's always sunny in Hanoi"
