import os 
import uuid
import threading 
from queue import Queue 



import tiktoken
from ultralytics import YOLO 



from langchain_ollama import ChatOllama
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore



import prompts



MAX_TOKENS = 32_000
PRODUCT_STATUS = "..."
FRAME_QUEUE = Queue()
MESSAGES_HISTORY_UI = []
STATUS_LOCK = threading.Lock()



ENCODING = tiktoken.get_encoding("cl100k_base") 
YOLO_OBJECT_DETECTION = YOLO("/home/chwenjun225/projects/DeepEngine/research/MAAOI/VisionAgent/runs/detect/train/weights/best.pt")
REASONING_INSTRUCT_LLM = ChatOllama(model="DeepSeek-R1-Distill-Qwen-7B-Q4_K_M:latest", num_predict=128_000)
VISION_INSTRUCT_LLM = ChatOllama(model="Llama-3.2-11B-Vision-Instruct.Q4_K_M:latest", num_predict=128_000)



MANAGER_AGENT_PROMPT_MSG 					= 		prompts.MANAGER_AGENT_PROMPT
ROUTER_AGENT_PROMPT_MSG 					= 		prompts.ROUTER_AGENT_PROMPT
SYSTEM_AGENT_PROMPT_MSG 					= 		prompts.SYSTEM_AGENT_PROMPT
ORCHESTRATE_AGENT_PROMPT_MSG				=	 	prompts.ORCHESTRATE_AGENT_PROMPT
REASONING_AGENT_PROMPT_MSG 					= 		prompts.REASONING_AGENT_PROMPT
RESEARCH_AGENT_PROMPT_MSG 					= 		prompts.RESEARCH_AGENT_PROMPT
PLANNING_AGENT_PROMPT_MSG 					= 		prompts.PLANNING_AGENT_PROMPT
EXECUTION_AGENT_PROMPT_MSG 					= 		prompts.EXECUTION_AGENT_PROMPT
COMMUNICATION_AGENT_PROMPT_MSG 				= 		prompts.COMMUNICATION_AGENT_PROMPT
EVALUATION_AGENT_PROMPT_MSG					= 		prompts.EVALUATION_AGENT_PROMPT
DEBUGGING_AGENT_PROMPT_MSG					= 		prompts.DEBUGGING_AGENT_PROMPT

INSTRUCT_VISISON_AGENT_PROMPT_MSG			=		prompts.INSTRUCT_VISION_AGENT_PROMPT
INSTRUCT_VISION_EXPLAIN_AGENT_PROMPT_MSG 	= 		prompts.INSTRUCT_VISION_EXPLAIN_AGENT_PROMPT



CONNECTION = "postgresql+psycopg://langchain:langchain@localhost:2028/langchain" 
COLLECTION_NAME = "maaoi"
EMBEDDING_FUNC = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



CHAT_HISTORY_COLLECTION_NAME = "maaoi"



CONFIG = {"configurable": {"thread_id": str(uuid.uuid4()), "recursion_limit": 100}}
CHECKPOINTER = MemorySaver()
STORE = InMemoryStore()



QUERIES = [
	"""Hello, good morning""", 
	
	"""You are an engineer working on a production line, responsible for inspecting the visual quality of PCB boards. Your task is to identify any surface defects on the board. If you detect any defects, respond with 'NG'. If there are no defects, respond with 'OK'.""",

	"""Develop a machine learning model, potentially using ResNet or EfficientNet, to inspect industrial products for surface defects (scratches, dents, discoloration). 
The dataset is provided as 'industrial_defects_images'. The model should achieve at least 0.97 accuracy""", 

	"""Base on your knowledge, build a deep learning model, to detect defects in PCB (Printed Circuit Board) images. 
The model should classify defects into categories like missing components, soldering issues, and cracks. We have uploaded the dataset as 'pcb_defects_dataset'. The model must achieve at least 0.95 accuracy.""", 

	"""I need a highly accurate machine learning model developed to classify images within the Butterfly Image Classification dataset into their correct species categories. 
The dataset has been uploaded with its label information in the labels.csv file. 
Please use a convolutional neural network (CNN) architecture for this task, leveraging transfer learning from a pre-trained ResNet-50 model to improve accuracy. 
Optimize the model using cross-validation on the training split to fine-tune hyperparameters, and aim for an accuracy of at least 0.95 (95%) on the test split. 
Provide the final trained model, a detailed report of the training process, hyperparameter settings, accuracy metrics, and a confusion matrix to evaluate performance across different categories.""", 

	"""Please provide a classification model that categorizes images into one of four clothing categories. 
The image path, along with its label information, can be found in the files train labels.csv and test labels.csv. 
The model should achieve at least 0.95 (95%) accuracy on the test set and be implemented using PyTorch. Additionally, please include data augmentation techniques and a confusion matrix in the evaluation.""", 

	"""Hello, What is heavier a kilo of feathers or a kilo of steel?""", 
	
	"""exit"""
]



DEBUG = False



LLAMA_TOKENS = {
	"BEGIN_OF_TEXT"			:	"<|begin_of_text|>"		,
	"END_OF_TEXT"			:	"<|end_of_text|>"		,
	"START_HEADER_ID"		:	"<|start_header_id|>"	,
	"END_HEADER_ID"			:	"<|end_header_id|>"		,
	"END_OF_MESSAGE_ID"		:	"<|eom_id|>"			,
	"END_OF_TURN_ID"		:	"<|eot_id|>"			,
}
