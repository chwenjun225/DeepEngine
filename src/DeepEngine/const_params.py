import uuid



from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore



from prompts import Prompts
from state import Conversation, Prompt2JSON



QUERIES = [
	"""I need a highly accurate machine learning model developed to classify images within the Butterfly Image Classification dataset into their correct species categories. 
	The dataset has been uploaded with its label information in the labels.csv file. 
	Please use a convolutional neural network (CNN) architecture for this task, leveraging transfer learning from a pre-trained ResNet-50 model to improve accuracy. 
	Optimize the model using cross-validation on the training split to fine-tune hyperparameters, and aim for an accuracy of at least 0.95 (95%) on the test split. 
	Provide the final trained model, a detailed report of the training process, hyperparameter settings, accuracy metrics, and a confusion matrix to evaluate performance across different categories.""",

	"""Please provide a classification model that categorizes images into one of four clothing categories. 
	The image path, along with its label information, can be found in the files train labels.csv and test labels.csv. 
	The model should achieve at least 0.95 (95%) accuracy on the test set and be implemented using PyTorch. 
	Additionally, please include data augmentation techniques and a confusion matrix in the evaluation."""	

	"""Hello, What is heavier a kilo of feathers or a kilo of steel?""", 

	"""exit"""
]



DEBUG = False
NAME = "foxconn_fulian_b09_ai_research_tranvantuan_v1047876"



COLLECTION_NAME = "foxconn_fulian_b09_ai_research_tranvantuan_v1047876"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
PERSIS_DIRECTORY = "/home/chwenjun225_laptop/projects/DeepEngine/src/DeepEngine/chromadb_storage"
VECTOR_DB = Chroma(persist_directory=PERSIS_DIRECTORY, embedding_function=EMBEDDING_MODEL_NAME, collection_name=COLLECTION_NAME)



CONVERSATION_2_JSON_MSG_PROMPT 	= 	Prompts.CONVERSATION_2_JSON_PROMPT 
MGR_SYS_MSG_PROMPT 				= 	Prompts.AGENT_MANAGER_PROMPT
VER_RELEVANCY_MSG_PROMPT 		= 	Prompts.REQUEST_VERIFY_RELEVANCY
VER_ADEQUACY_MSG_PROMPT 		= 	Prompts.REQUEST_VERIFY_ADEQUACY
PROMPT_2_JSON_SYS_MSG_PROMPT 	= 	Prompts.PROMPT_AGENT_PROMPT
RAP_SYS_MSG_PROMPT 				= 	Prompts.RETRIEVAL_AUGMENTED_PLANNING_PROMPT



BEGIN_OF_TEXT		=	"<|begin_of_text|>"
END_OF_TEXT			= 	"<|end_of_text|>"
START_HEADER_ID		= 	"<|start_header_id|>"
END_HEADER_ID		= 	"<|end_header_id|>"
END_OF_MESSAGE_ID	= 	"<|eom_id|>"
END_OF_TURN_ID		= 	"<|eot_id|>"



CONFIG = {"configurable": {"thread_id": str(uuid.uuid4()), "recursion_limit": 100}}
CHECKPOINTER = MemorySaver()
STORE = InMemoryStore()



LLM_HTEMP	=	ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0.8, num_predict=128_000)
LLM_LTEMP 	= 	ChatOllama(model="llama3.2:1b-instruct-fp16", temperature=0, num_predict=128_000)
LLM_STRUC_OUT_CONVERSATION 	=	LLM_HTEMP.with_structured_output(schema=Conversation, method="json_schema")
LLM_STRUC_OUT_AUTOML 		= 	LLM_HTEMP.with_structured_output(schema=Prompt2JSON, method="json_schema")
