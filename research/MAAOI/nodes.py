import json 
from typing_extensions import List



from langchain_core.messages import (
	HumanMessage, 
	AIMessage, 
	SystemMessage, 
	BaseMessage
)



from langgraph.graph import END



from state import State
from const_vars import (
	LLAMA_TOKENS, 
	MGR_SYS_MSG_PROMPT, 
	LLM_LTEMP, 
	RAP_SYS_MSG_PROMPT, 
	RELEVANCY_MSG_PROMPT, 
)
from utils import add_unique_msg, get_latest_msg



def ORCHESTRATE_AGENTS(state: State) -> State: 
	"""Điều phối, kích hoạt và sắp xếp luồng chạy của các agent."""
	return state 



def SYSTEM_AGENT(state: State) -> State:
	"""Quản lý toàn bộ workflow, đảm bảo tính logic và nhất quán của hệ thống."""
	return state



def REASONING_AGENT(state: State) -> State:
	"""Suy luận và phân tích yêu cầu người dùng để xác định bản chất vấn đề."""
	return state 



def RESEARCH_AGENT(state: State) -> State:
	"""Tìm kiếm thông tin, tài liệu, công cụ hỗ trợ, lập luận sâu để giải quyết bài toán."""
	return state 



def PLANNING_AGENT(state: State) -> State: 
	"""Xây dựng kế hoạch hành động, bao gồm mô hình, dữ liệu, công cụ cần dùng."""
	return state 



def EXECUTION_AGENT(state: State) -> State: 
	"""Thực thi kế hoạch: huấn luyện mô hình, xử lý dữ liệu, chạy pipeline."""
	return state 



def EVALUATION_AGENT(state: State) -> State: 
	"""Đánh giá kết quả đầu ra, hiệu suất mô hình hoặc độ phù hợp của giải pháp."""
	return state 



def DEBUGGING_AGENT(state: State) -> State: 
	"""Kiểm tra, phát hiện và khắc phục lỗi trong quá trình thực thi."""
	return state



def relevancy(state: State) -> List[BaseMessage]:
	"""Check request verification-relevancy of human_query."""
	human_msg = get_latest_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN")
	sys_msg = SystemMessage(RELEVANCY_MSG_PROMPT.format(instruction=human_msg.content))
	ai_msg = LLM_LTEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage(content=ai_msg.strip() \
			if isinstance(ai_msg, str) \
			else "At node_request_verify-REQUEST_VERIFY_RELEVANCY, I'm unable to generate a response.")
	return [sys_msg, human_msg, ai_msg]



def request_verify_adequacy_or_relevancy(state: State) -> State:
	"""Request verification output of Agent Manager."""
	ai_msg_relevancy = relevancy(state=state)[2]
	yes_or_no_answer = "YES" if "YES" in [ai_msg_relevancy.content.upper()] else "NO"
	ai_msg = AIMessage(content=yes_or_no_answer)
	add_unique_msg(state=state, node="REQUEST_VERIFY", msgs_type="AI", msg=ai_msg)
	return state



def manager_agent(state: State) -> State:
	"""Manager Agent.

	Example:
		>>> Human query: I need a very accurate model to classify images in the 
				Butterfly Image Classification dataset into their respective 
				categories. The dataset has been uploaded with its label 
				information in the labels.csv file.
		>>> AI response: Here is a sample code that uses the Keras library to develop and train a convolutional neural network (CNN) model for ...
	"""
	sys_msg = SystemMessage(content=MGR_SYS_MSG_PROMPT)
	human_msg = HumanMessage(content=state["human_query"][-1].content)
	ai_msg = LLM_LTEMP.invoke([sys_msg, human_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage( 
			content=ai_msg.strip() 
			if isinstance(ai_msg, str) 
			else "At node_manager_agent, I'm unable to generate a response."
		)
	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="SYS", msg=sys_msg)
	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN", msg=human_msg)
	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="AI", msg=AIMessage(ai_msg.content))
	return state



def request_verify_control_flow(state: State) -> State:
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
	"""Call AI-Vision.

	Args:
		state (State): The current state of the conversation.

	Returns:
		State: Updated state with the parsed JSON response.
	"""
	human_msg = get_latest_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN")
	parsed_json = {"tool_execution": "AI_VISION"}
	ai_msg = AIMessage(content=json.dumps(parsed_json, indent=2))
	add_unique_msg(state=state, node="PROMPT_AGENT", msgs_type="AI", msg=ai_msg)
	return state



def retrieval_augmented_planning_agent(state: State) -> State:
	"Retrieval-Augmented Planning Agent."
	human_msg_content = get_latest_msg(state=state, node="PROMPT_AGENT", msgs_type="AI").content
	plan_knowledge = """This is RAG steps""" 
	sys_msg = SystemMessage(content=RAP_SYS_MSG_PROMPT.format(
		BEGIN_OF_TEXT=LLAMA_TOKENS["BEGIN_OF_TEXT"], 
		START_HEADER_ID=LLAMA_TOKENS["START_HEADER_ID"], 
		END_HEADER_ID=LLAMA_TOKENS["END_HEADER_ID"], 
		user_requirements=human_msg_content, 
		plan_knowledge=plan_knowledge,
		END_OF_TURN_ID=LLAMA_TOKENS["END_OF_TURN_ID"]
	))
	ai_msg = LLM_LTEMP.invoke([sys_msg])
	if not isinstance(ai_msg, AIMessage): 
		ai_msg = AIMessage(
			content=ai_msg.strip() 
			if isinstance(ai_msg, str) 
			else "At node_manager_agent, I'm unable to generate a response."
		)
	add_unique_msg(state=state, node="RAP", msgs_type="SYS", msg=sys_msg)
	add_unique_msg(state=state, node="RAP", msgs_type="HUMAN", msg=HumanMessage(human_msg_content))
	add_unique_msg(state=state, node="RAP", msgs_type="AI", msg=ai_msg)
	return state
