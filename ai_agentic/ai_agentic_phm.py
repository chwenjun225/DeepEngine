from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt import create_react_agent 
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.tools import tool 

from chwenjun225_tools import rag, planning, save_chat_history_to_chromadb, get_llm
SYSTEM_PROMPT = "You are a friendly and helpful assistant. Your job is to answer human questions with care and detail. Keep your answers short and concise when possible."
PERSIST_DIRECTORY = "/home/chwenjun225/Projects/Foxer/ai_agentic/chroma_db"
COLLECTION_NAME = "foxconn_ai_research"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

planning_prompt = ChatPromptTemplate.from_messages([(
	"human", 
	"""Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
	Use three sentences maximum and keep the answer concise. 
	Question: {question} 
	Context: {context} 
	Answer:""")]) 

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model, collection_name=COLLECTION_NAME)

# Create the agent
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# with open("chroma_logs.txt", "a") as f:
	# sys.stdout = f  # Chuyển tất cả print() vào file
	# llm = get_llm(
	# 	port=2026, host="127.0.0.1", openai_api_key="chwenjun225", 
	# 	model_name="1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct", 
	# 	temperature=0
	# )
	# search = TavilySearchResults(max_results=2)
	# tools = [search]
	# model_with_tools = llm.bind_tools(tools)

	# # user_input = input(">>> 👨 User: ") 
	# user_input = "What's the difference between revenue year of 2023 and 2024?"
	# # user_input = "Hello"
	# # 🔍 Truy vấn RAG từ ChromaDB
	# rag_output = rag(query=user_input, num_retrieved_docs=1)
	# # print(f">>> <rag_output>{rag_output}</rag_output>")

	# # 🧠 Lập kế hoạch phản hồi từ Planning Module
	# planning_module_output = planning(user_input, rag_output)
	# # print(f">>> <planning_module>{planning_module_output}</planning_module>")

	# # 🤖 Gửi vào LLM để nhận phản hồi
	# response_template = ChatPromptTemplate.from_messages([
	# 	("system", "You are a friendly and helpful assistant. Your job is to answer human questions with care and detail. Keep your answers short and concise when possible."),
	# 	("user", "{input}")
	# ])
	# formatted_response_template = response_template.invoke({"input": planning_module_output})
	# response = llm.invoke(formatted_response_template)
	# assistant_response = response.content
	# print(f">>> 🤖 Assistant:\n{assistant_response}")

	# # 📝 Lưu lịch sử hội thoại vào ChromaDB
	# save_chat_history_to_chromadb(user_input, assistant_response)
	# sys.stdout = sys.__stdout__  # Reset lại stdout về mặc định
