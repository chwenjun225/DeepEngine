if False:
	import chromadb
	from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

	# Sử dụng embedding mạnh hơn
	embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

	# Tạo client-database
	chroma_client = chromadb.PersistentClient(path="./chroma_db")
	collection = chroma_client.get_or_create_collection(name="phm_records", embedding_function=embedding_function)

	# Tăng cường kết quả truy vấn 
	results = collection.query(query_texts=["High vibration detected"], n_results=3)

	# Thêm dữ liệu bảo trì vào ChromaDB
	collection.add(
		documents=["Machine 22 had vibration issues due to misalignment. Solution: Realignment and lubrication."],
		metadatas=[{"issue": "vibration", "solution": "realignment"}],
		ids=["1"]
	)

	# Tìm kiếm dữ liệu tương tự
	results = collection.query(query_texts=["High vibration detected"], n_results=1)
	print("Kết quả tìm kiếm:", results)
if False:
	from openai import OpenAI
	path_model = "/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"

	client = OpenAI(
		base_url="http://localhost:2025/v1",
		api_key="chwenjun225",
	)

	completion = client.chat.completions.create(
		model=path_model,
		messages=[
			{"role": "user", "content": "Hello!"}, 
			{"role": "user", "content": "Explain what is AI-Agent?"}
		]
	)

	print(completion.choices[0].message)

from typing import Optional, List 

from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun

class CustomServerLLM(BaseLLM):
	def _call(
		self, 
		prompt: str, 
		stop: Optional[List[str]] = None,
		run_manager: Optional[CallbackManagerForLLMRun] = None
	) -> str:
		# Implement your API request logic here
		# Make request to your server
		# Return the response
		pass

	@property
	def _llm_type(self) -> str:
		return "custom_server_llm"
	
# TODO: (very urgent)
# 1. Build VectorDB and DB.
# 2. Learn how to modify the chat-langchain: https://github.com/langchain-ai/chat-langchain/blob/master/MODIFY.md