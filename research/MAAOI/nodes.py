from typing_extensions import cast, Literal


from langchain_core.messages import ( 
	convert_to_openai_messages,
	BaseMessage, 
	SystemMessage, 
	HumanMessage,
	AIMessage, 
)



from langgraph.types import Command



from state import State
from const_vars import (
	MANAGER_AGENT_PROMPT_MSG		,
	ROUTER_AGENT_PROMPT_MSG			,
	SYSTEM_AGENT_PROMPT_MSG			,
	ORCHESTRATE_AGENT_PROMPT_MSG	,
	REASONING_AGENT_PROMPT_MSG		, 
	RESEARCH_AGENT_PROMPT_MSG		,
	PLANNING_AGENT_PROMPT_MSG		,
	EXECUTION_AGENT_PROMPT_MSG		,
	COMMUNICATION_AGENT_PROMPT_MSG	,
	EVALUATION_AGENT_PROMPT_MSG		,
	DEBUGGING_AGENT_PROMPT_MSG		, 

	REASONING_INSTRUCT_LLM			,
)
from utils import (
	trim_context					,
	has_agent_got_sys_prompt		,
	has_agent_got_name_attr			,
	replace_message_content			,
	prepare_context					,
)



def MANAGER_AGENT(state: State) -> State:
	"""Tiếp nhận truy vấn người dùng và phản hồi theo ngữ cảnh."""
	ctx, sys_msg = prepare_context(state, "MANAGER_AGENT", MANAGER_AGENT_PROMPT_MSG)
	ai_msg = has_agent_got_name_attr(REASONING_INSTRUCT_LLM.invoke(ctx), "MANAGER_AGENT")
	return {"messages": [msg for msg in (sys_msg, ai_msg) if msg]}



def ROUTER_AGENT(state: State) -> Command[Literal["__end__", "SYSTEM_AGENT"]]:
	"""Phân loại truy vấn có thuộc domain AI/ML không."""
	ctx, sys_msg = prepare_context(state, "ROUTER_AGENT", ROUTER_AGENT_PROMPT_MSG)
	ai_msg = has_agent_got_name_attr(REASONING_INSTRUCT_LLM.invoke(ctx), "ROUTER_AGENT")

	goto = "__end__" if ai_msg.content != "SYSTEM_AGENT" else "SYSTEM_AGENT"
	final_msg = ai_msg if goto != "SYSTEM_AGENT" else replace_message_content(ai_msg, goto)
	return Command(update={"messages": [msg for msg in (sys_msg, final_msg) if msg]}, goto=goto)



def SYSTEM_AGENT(state: State) -> State:
	"""Chuẩn hóa đầu vào và đảm bảo logic yêu cầu người dùng."""
	msgs = convert_to_openai_messages(
		messages=cast(list[dict], state["messages"])
	)
	if not has_agent_got_sys_prompt(
		messages=msgs, 
		agent_name="SYSTEM_AGENT"
	):
		sys_msg = {
			"role": "system", 
			"name": "SYSTEM_AGENT", 
			"content": SYSTEM_AGENT_PROMPT_MSG
		}
		msgs = msgs + [sys_msg]
	ctx = trim_context(messages=msgs)
	resp = REASONING_INSTRUCT_LLM.invoke(input=ctx)
	resp["name"] = "SYSTEM_AGENT"
	return {"messages": [resp]}



def ORCHESTRATE_AGENT(state: State) -> State: 
	"""Xác định các agent cần thiết cho luồng hiện tại."""
	msgs = convert_to_openai_messages(
		messages=cast(list[dict], state["messages"])
	)
	if not has_agent_got_sys_prompt(
		messages=msgs, 
		agent_name="ORCHESTRATE_AGENT"
	):
		sys_msg = {
			"role": "system", 
			"name": "ORCHESTRATE_AGENT", 
			"content": ORCHESTRATE_AGENT_PROMPT_MSG 
		}
		messages = messages + [sys_msg]
	ctx = trim_context(messages=msgs)
	resp = REASONING_INSTRUCT_LLM.invoke(input=ctx)
	resp["name"] = "ORCHESTRATE_AGENT"
	return {"messages": [resp]}



def REASONING_AGENT(state: State) -> State:
	"""Phân tích yêu cầu để hiểu bản chất vấn đề và mục tiêu."""
	msgs = convert_to_openai_messages(
		messages=cast(list[dict], state["messages"])
	)	
	if not has_agent_got_sys_prompt(
		messages=msgs, 
		agent_name="REASONING_AGENT"
	):
		sys_msg = {
			"role": "system", 
			"name": "REASONING_AGENT", 
			"content": REASONING_AGENT_PROMPT_MSG 
		}
		msgs = msgs + [sys_msg]
	ctx = trim_context(msgs)
	resp = REASONING_INSTRUCT_LLM.invoke(input=ctx)
	resp["name"] = "REASONING_AGENT"
	return {"messages": [resp]}



def RESEARCH_AGENT(state: State) -> State:
	"""Tìm kiếm kiến thức, tài liệu, hoặc công cụ phục vụ bài toán."""
	msgs = convert_to_openai_messages(
		messages=cast(list[dict], state["messages"])
	)
	if not has_agent_got_sys_prompt(
		messages=msgs, 
		agent_name="RESEARCH_AGENT"
	):
		sys_msg = {
			"role": "system", 
			"name": "RESEARCH_AGENT", 
			"content": RESEARCH_AGENT_PROMPT_MSG 
		}
		msgs = msgs + [sys_msg]

	ctx = trim_context(sys_msg)
	resp = REASONING_INSTRUCT_LLM.invoke(input=ctx)
	resp["name"] = "RESEARCH_AGENT"
	return {"messages": [resp]}



def PLANNING_AGENT(state: State) -> State: 
	"""Xây dựng kế hoạch chi tiết cho các bước tiếp theo."""
	msgs = convert_to_openai_messages(
		messages=cast(list[dict], state["messages"])
	)
	if not has_agent_got_sys_prompt(
		messages=msgs, 
		agent_name="PLANNING_AGENT"
	):
		sys_msg = {
			"role": "system", 
			"name": "PLANNING_AGENT", 
			"content": PLANNING_AGENT_PROMPT_MSG 
		}
		msgs = msgs + [sys_msg]
	ctx = trim_context(msgs)
	resp = REASONING_INSTRUCT_LLM.invoke(input=ctx)
	resp["name"] = "PLANNING_AGENT"
	return {"messages": [resp]}



def EXECUTION_AGENT(state: State) -> State: 
	"""Thực thi kế hoạch đã lên bằng mô hình, pipeline hoặc thao tác logic."""
	msgs = convert_to_openai_messages(
		messages=cast(list[dict], state["messages"])
	)
	if not has_agent_got_sys_prompt(
		messages=msgs, 
		agent_name="EXECUTION_AGENT"
	):
		sys_msg = {
			"role": "system", 
			"name": "EXECUTION_AGENT", 
			"content": EXECUTION_AGENT_PROMPT_MSG 
		}
		msgs = msgs + [sys_msg]
	ctx = trim_context(msgs)
	resp = REASONING_INSTRUCT_LLM.invoke(ctx)
	resp["name"] = "EXECUTION_AGENT"
	return {"messages": [resp]} # Giống với ROUTER_AGENT 



def DEBUGGING_AGENT(state: State) -> State: 
	"""Chẩn đoán và khắc phục lỗi phát sinh trong quá trình thực thi."""
	msgs = convert_to_openai_messages(
		messages=cast(list[dict], state["messages"])
	)
	if not has_agent_got_sys_prompt(msgs, "DEBUGGING_AGENT"):
		sys_msg = {
			"role": "system", 
			"name": "DEBUGGING_AGENT", 
			"content": DEBUGGING_AGENT_PROMPT_MSG 
		}
		msgs = msgs + [sys_msg]
	ctx = trim_context(msgs)
	resp = REASONING_INSTRUCT_LLM.invoke(ctx)
	resp["name"] = "DEBUGGING_AGENT"
	return {"messages": [resp]}



def EVALUATION_AGENT(state: State) -> State: 
	"""Đánh giá chất lượng kết quả sau bước thực thi."""
	msgs = convert_to_openai_messages(
		messages=cast(list[dict], state["messages"])
	)
	if not has_agent_got_sys_prompt(msgs, "EVALUATION_AGENT"):
		sys_msg = {
			"role": "system", 
			"name": "EVALUATION_AGENT", 
			"content": EVALUATION_AGENT_PROMPT_MSG 
		}
		msgs = msgs + [sys_msg]
	ctx = trim_context(msgs)
	resp = REASONING_INSTRUCT_LLM.invoke(ctx)
	resp["name"] = "EVALUATION_AGENT"
	return {"messages": [resp]}



def COMMUNICATION_AGENT(state: State) -> State: 
	"""Tổng hợp kết quả và phản hồi cuối cùng tới người dùng."""
	msgs = convert_to_openai_messages(
		messages=cast(list[dict], state["messages"])
	)
	if not has_agent_got_sys_prompt(
		messages=msgs, 
		agent_name="COMMUNICATION_AGENT"
	):
		sys_msg = {
			"role": "system", 
			"name": "COMMUNICATION_AGENT", 
			"content": COMMUNICATION_AGENT_PROMPT_MSG 
		}
		msgs = msgs + [sys_msg]
	ctx = trim_context(msgs)
	resp = REASONING_INSTRUCT_LLM.invoke(ctx)
	resp["name"] = "COMMUNICATION_AGENT"
	return {"messages": [resp]}



# def relevancy(state: State) -> List[BaseMessage]:
# 	"""Check request verification-relevancy of human_query."""
# 	human_msg = get_latest_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN")
# 	sys_msg = SystemMessage(RELEVANCY_MSG_PROMPT.format(instruction=human_msg.content))
# 	ai_msg = LLM_LTEMP.invoke([sys_msg])
# 	if not isinstance(ai_msg, AIMessage): 
# 		ai_msg = AIMessage(content=ai_msg.strip() \
# 			if isinstance(ai_msg, str) \
# 			else "At node_request_verify-REQUEST_VERIFY_RELEVANCY, I'm unable to generate a response.")
# 	return [sys_msg, human_msg, ai_msg]



# def request_verify_adequacy_or_relevancy(state: State) -> State:
# 	"""Request verification output of Agent Manager."""
# 	ai_msg_relevancy = relevancy(state=state)[2]
# 	yes_or_no_answer = "YES" if "YES" in [ai_msg_relevancy.content.upper()] else "NO"
# 	ai_msg = AIMessage(content=yes_or_no_answer)
# 	add_unique_msg(state=state, node="REQUEST_VERIFY", msgs_type="AI", msg=ai_msg)
# 	return state



# def manager_agent(state: State) -> State:
# 	"""Manager Agent.

# 	Example:
# 		>>> Human query: I need a very accurate model to classify images in the 
# 				Butterfly Image Classification dataset into their respective 
# 				categories. The dataset has been uploaded with its label 
# 				information in the labels.csv file.
# 		>>> AI response: Here is a sample code that uses the Keras library to develop and train a convolutional neural network (CNN) model for ...
# 	"""
# 	sys_msg = SystemMessage(content=MGR_SYS_MSG_PROMPT)
# 	human_msg = HumanMessage(content=state["human_query"][-1].content)
# 	ai_msg = LLM_LTEMP.invoke([sys_msg, human_msg])
# 	if not isinstance(ai_msg, AIMessage): 
# 		ai_msg = AIMessage( 
# 			content=ai_msg.strip() 
# 			if isinstance(ai_msg, str) 
# 			else "At node_manager_agent, I'm unable to generate a response."
# 		)
# 	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="SYS", msg=sys_msg)
# 	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN", msg=human_msg)
# 	add_unique_msg(state=state, node="MANAGER_AGENT", msgs_type="AI", msg=AIMessage(ai_msg.content))
# 	return state



# def request_verify_control_flow(state: State) -> State:
# 	"""Determines the next step based on the AI response from REQUEST_VERIFY.

# 	Args:
# 		state (State): The current conversation state.

# 	Returns:
# 		str: The next agent ("PROMPT_AGENT" or END).

# 	Raises:
# 		ValueError: If there is no valid AI response or an unexpected response.
# 	"""

# 	if "REQUEST_VERIFY" not in state["messages"] or "AI" not in state["messages"]["REQUEST_VERIFY"]:
# 		raise ValueError("[ERROR]: No AI message found in REQUEST_VERIFY.")
# 	ai_msgs = state["messages"]["REQUEST_VERIFY"]["AI"]
# 	if not ai_msgs:
# 		raise ValueError("[ERROR]: AI message list is empty in REQUEST_VERIFY.")
# 	ai_msg = ai_msgs[-1]
# 	if not hasattr(ai_msg, "content"):
# 		raise ValueError("[ERROR]: AI message has no content.")
# 	resp = ai_msg.content.strip().upper()
# 	next_step_map = {"YES": "PROMPT_AGENT", "NO": END}
# 	if resp in next_step_map:
# 		return next_step_map[resp]
# 	raise ValueError(f">>> [ERROR]: Unexpected response '{resp}'")



# def prompt_agent(state: State) -> State:
# 	"""Call AI-Vision.

# 	Args:
# 		state (State): The current state of the conversation.

# 	Returns:
# 		State: Updated state with the parsed JSON response.
# 	"""
# 	human_msg = get_latest_msg(state=state, node="MANAGER_AGENT", msgs_type="HUMAN")
# 	parsed_json = {"tool_execution": "AI_VISION"}
# 	ai_msg = AIMessage(content=json.dumps(parsed_json, indent=2))
# 	add_unique_msg(state=state, node="PROMPT_AGENT", msgs_type="AI", msg=ai_msg)
# 	return state



# def retrieval_augmented_planning_agent(state: State) -> State:
# 	"Retrieval-Augmented Planning Agent."
# 	human_msg_content = get_latest_msg(state=state, node="PROMPT_AGENT", msgs_type="AI").content
# 	plan_knowledge = """This is RAG steps""" 
# 	sys_msg = SystemMessage(content=RAP_SYS_MSG_PROMPT.format(
# 		BEGIN_OF_TEXT=LLAMA_TOKENS["BEGIN_OF_TEXT"], 
# 		START_HEADER_ID=LLAMA_TOKENS["START_HEADER_ID"], 
# 		END_HEADER_ID=LLAMA_TOKENS["END_HEADER_ID"], 
# 		user_requirements=human_msg_content, 
# 		plan_knowledge=plan_knowledge,
# 		END_OF_TURN_ID=LLAMA_TOKENS["END_OF_TURN_ID"]
# 	))
# 	ai_msg = LLM_LTEMP.invoke([sys_msg])
# 	if not isinstance(ai_msg, AIMessage): 
# 		ai_msg = AIMessage(
# 			content=ai_msg.strip() 
# 			if isinstance(ai_msg, str) 
# 			else "At node_manager_agent, I'm unable to generate a response."
# 		)
# 	add_unique_msg(state=state, node="RAP", msgs_type="SYS", msg=sys_msg)
# 	add_unique_msg(state=state, node="RAP", msgs_type="HUMAN", msg=HumanMessage(human_msg_content))
# 	add_unique_msg(state=state, node="RAP", msgs_type="AI", msg=ai_msg)
# 	return state
