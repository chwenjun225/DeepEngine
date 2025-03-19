import json 



from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage



from langgraph.graph import END



from state import State, Prompt2JSON
from const_vars import (
	SPECIAL_TOKENS_LLAMA_MODELS, 
	MGR_SYS_MSG_PROMPT, 
	LLM_LTEMP, 
	PROMPT_2_JSON_SYS_MSG_PROMPT, 
	LLM_STRUC_OUT_AUTOML, 
	RAP_SYS_MSG_PROMPT
)
from utils import add_eoturn_eotext_to_ai_msg, add_unique_msg, get_latest_msg, conversation2json



def manager_agent(state: State) -> State:
	"""Manager Agent.

	Example:
		>>> Human query: I need a very accurate model to classify images in the 
				Butterfly Image Classification dataset into their respective 
				categories. The dataset has been uploaded with its label 
				information in the labels.csv file.
		>>> AI response: Here is a sample code that uses the Keras library to develop and train a convolutional neural network (CNN) model for ...
	"""
	sys_msg = SystemMessage(content=MGR_SYS_MSG_PROMPT.format(
			BEGIN_OF_TEXT=SPECIAL_TOKENS_LLAMA_MODELS["BEGIN_OF_TEXT"], 
			START_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["START_HEADER_ID"], 
			END_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_HEADER_ID"], 
			END_OF_TURN_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"]
		))
	human_msg = HumanMessage(content=state["human_query"][-1].content)
	ai_msg = LLM_LTEMP.invoke([sys_msg, human_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage( 
			content=ai_msg.strip() 
			if isinstance(ai_msg, str) 
			else "At node_manager_agent, I'm unable to generate a response."
		)
	ai_msg = add_eoturn_eotext_to_ai_msg(ai_msg=ai_msg, end_of_turn_id_token=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"], end_of_text_token=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TEXT"])
	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="SYS", msg=sys_msg)
	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN", msg=human_msg)
	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="AI", msg=AIMessage(ai_msg.content))
	return state



def request_verify(state: State) -> State:
	"""Determines the next step based on the AI response from REQUEST_VERIFY.

	Args:
		state (State): The current conversation state.

	Returns:
		str: The next agent ("PROMPT_AGENT" or END).

	Raises:
		ValueError: If there is no valid AI response or an unexpected response.
	"""
	if "REQUEST_VERIFY" not in state["messages"] or "AI" not in state["messages"]["REQUEST_VERIFY"]:
		raise ValueError("[ERROR]: No AI message found in REQUEST_VERIFY.")
	ai_msgs = state["messages"]["REQUEST_VERIFY"]["AI"]
	if not ai_msgs:
		raise ValueError("[ERROR]: AI message list is empty in REQUEST_VERIFY.")
	ai_msg = ai_msgs[-1]
	if not hasattr(ai_msg, "content"):
		raise ValueError("[ERROR]: AI message has no content.")
	resp = ai_msg.content.strip().upper()
	next_step_map = {"YES": "PROMPT_AGENT", "NO": END}
	if resp in next_step_map:
		return next_step_map[resp]
	raise ValueError(f">>> [ERROR]: Unexpected response '{resp}'")



def prompt_agent(state: State) -> State:
	"""Prompt Agent parses user's requirements into JSON following a TypedDict schema.

	Args:
		state (State): The current state of the conversation.

	Returns:
		State: Updated state with the parsed JSON response.
	"""
	if "MANAGER_AGENT" not in state["messages"] or "HUMAN" not in state["messages"]["MANAGER_AGENT"]:
		raise ValueError(">>> [ERROR]: No HUMAN messages found in MANAGER_AGENT.")
	human_msg = get_latest_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN")
	parsed_json = conversation2json(msg_prompt=PROMPT_2_JSON_SYS_MSG_PROMPT, 
		llm_structure_output=LLM_STRUC_OUT_AUTOML,
		human_msg=human_msg,
		schema=Prompt2JSON
	)
	expected_keys, received_keys = set(Prompt2JSON.__annotations__), set(parsed_json)
	if missing_keys := expected_keys - received_keys: 
		raise ValueError(f"[ERROR]: JSON thiếu các trường bắt buộc: {missing_keys}")
	if extra_keys := received_keys - expected_keys: 
		raise ValueError(f"[ERROR]: JSON có các trường không hợp lệ: {extra_keys}")
	ai_msg_json = AIMessage(content=json.dumps(parsed_json, indent=2))
	add_unique_msg(state=state, node="PROMPT_AGENT", msgs_type="AI", msg=ai_msg_json)
	return state



def retrieval_augmented_planning_agent(state: State) -> State:
	"Retrieval-Augmented Planning Agent."
	human_msg_content = get_latest_msg(state=state, node="PROMPT_AGENT", msgs_type="AI").content
	plan_knowledge = """This is RAG steps""" 
	sys_msg = SystemMessage(content=RAP_SYS_MSG_PROMPT.format(
		BEGIN_OF_TEXT=SPECIAL_TOKENS_LLAMA_MODELS["BEGIN_OF_TEXT"], 
		START_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["START_HEADER_ID"], 
		END_HEADER_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_HEADER_ID"], 
		user_requirements=human_msg_content, 
		plan_knowledge=plan_knowledge,
		END_OF_TURN_ID=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"]
	))
	ai_msg = LLM_LTEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage(
			content=ai_msg.strip() 
			if isinstance(ai_msg, str) 
			else "At node_manager_agent, I'm unable to generate a response."
		)
	ai_msg = add_eoturn_eotext_to_ai_msg(
		ai_msg=ai_msg, 
		end_of_turn_id_token=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TURN_ID"], 
		end_of_text_token=SPECIAL_TOKENS_LLAMA_MODELS["END_OF_TEXT"]
	)
	add_unique_msg(state=state, node="RAP", msgs_type="SYS", msg=sys_msg)
	add_unique_msg(state=state, node="RAP", msgs_type="HUMAN", msg=HumanMessage(human_msg_content))
	add_unique_msg(state=state, node="RAP", msgs_type="AI", msg=ai_msg)
	return state



def data_agent(state: State) -> State:
	"Data Agent."
	return state



def model_agent(state: State) -> State:
	"Model Agent."
	return state
