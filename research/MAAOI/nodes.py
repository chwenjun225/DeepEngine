from state import State
from const import (
	LLM							,
)
from utils import (
	trim_context				,
	has_agent_got_sys_prompt	,
	has_agent_got_name_attr		,
	replace_message_content		,
	prepare_context				,
)



def TEMPORAL_PATTERN_AGENT(state: State) -> State:
	"""Phân loại lỗi."""
	return state 



def DEFECT_REASONING_AGENT(state: State) -> State:
	"""Đặt câu hỏi, tư duy nhanh về lỗi."""
	return state 




def CRITICAL_ASSESSMENT_AGENT(state: State) -> State:
	"""Đánh giá mức độ lỗi."""
	return state 




def REPORT_GENERATOR_AGENT(state: State) -> State:
	"""Báo cáo kết quả cho người dùng."""
	return state 



def VISUAL_AGENT(state: State) -> State:
	"""Đưa ra tọa độ lỗi cuối cùng sau quá trình suy luận."""
	return state 


# TODO: ----THAM KHẢO 2 MẪU CODE CHO CÁC AGENTS----



# def MANAGER_AGENT(state: State) -> State:
# 	"""Tiếp nhận truy vấn người dùng và phản hồi theo ngữ cảnh."""
# 	ctx, sys_msg = prepare_context(state, "MANAGER_AGENT", MANAGER_AGENT_PROMPT_MSG)
# 	ai_msg = has_agent_got_name_attr(REASONING_LLM.invoke(ctx), "MANAGER_AGENT")
# 	return {"messages": [msg for msg in (sys_msg, ai_msg) if msg]}



# def ROUTER_AGENT(state: State) -> Command[Literal["__end__", "SYSTEM_AGENT"]]:
# 	"""Phân loại truy vấn có thuộc domain AI/ML không."""
# 	ctx, sys_msg = prepare_context(state, "ROUTER_AGENT", ROUTER_AGENT_PROMPT_MSG)
# 	ai_msg = has_agent_got_name_attr(REASONING_LLM.invoke(ctx), "ROUTER_AGENT")

# 	goto = "__end__" if ai_msg.content != "SYSTEM_AGENT" else "SYSTEM_AGENT"
# 	final_msg = ai_msg if goto != "SYSTEM_AGENT" else replace_message_content(ai_msg, goto)
# 	return Command(update={"messages": [msg for msg in (sys_msg, final_msg) if msg]}, goto=goto)
