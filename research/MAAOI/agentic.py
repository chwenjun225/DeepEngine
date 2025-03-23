from langgraph.graph import(
    StateGraph, 
    START, 
    END
)



from state import State 
from const_vars import (
    PATH_MAP, 
    DEBUG, 
    CHECKPOINTER, 
    STORE
)
from nodes import (
	SYSTEM_AGENT, 
    ORCHESTRATE_AGENTS, 
	REASONING_AGENT, 
	RESEARCH_AGENT, 
	PLANNING_AGENT, 
	EXECUTION_AGENT, 
    COMMUNICATION_AGENT, 
	EVALUATION_AGENT, 
	DEBUGGING_AGENT
)



WORKFLOW = StateGraph(State)

WORKFLOW.add_node(	node="SYSTEM_AGENT"				,		action=SYSTEM_AGENT			)
WORKFLOW.add_node(	node="ORCHESTRATE_AGENTS"		,		action=ORCHESTRATE_AGENTS	)
WORKFLOW.add_node(	node="REASONING_AGENT"			,		action=REASONING_AGENT		)
WORKFLOW.add_node(	node="RESEARCH_AGENT"			,		action=RESEARCH_AGENT		)
WORKFLOW.add_node(	node="PLANNING_AGENT"			,		action=PLANNING_AGENT		)
WORKFLOW.add_node(	node="EXECUTION_AGENT"			,		action=EXECUTION_AGENT		)
WORKFLOW.add_node(	node="COMMUNICATION_AGENT"		,		action=COMMUNICATION_AGENT	)
WORKFLOW.add_node(	node="EVALUATION_AGENT"			,		action=EVALUATION_AGENT		)
WORKFLOW.add_node(	node="DEBUGGING_AGENT"			,		action=DEBUGGING_AGENT		)

WORKFLOW.add_edge(start_key=START, end_key="SYSTEM_AGENT")
WORKFLOW.add_conditional_edges(
	source="SYSTEM_AGENT", 
	path_map=PATH_MAP, 
	then="cleanup"
)



AGENTIC = WORKFLOW.compile(
	store=STORE, 
    debug=DEBUG, 
	checkpointer=CHECKPOINTER,
	name="foxconn_fulian_b09_ai_research_tranvantuan_v1047876"
)


# TODO: Ta sẽ truyền ngữ cảnh nghĩa là toán lịch sử cuộc trò chuyện vào cho mô hình ngôn ngữ lớn, để làm được như vậy, ta cũng cần tính toán 
# số lượng max_token mà mô hình cho phép xử lý, nếu vượt quá số đó thì ta cần gọi đến AGENT_SUMZATION để tổng hợp lại toàn bộ ngữ cảnh cuộc trò 
# chuyện sau đó lại nhắc lại system_prompt cho agent. 


# Cần phải có một thuật toán để tính được độ dài ngữ cảnh của lịch sử trò chuyện, nếu đạt đến quá đủ thì cần phải nhắc lại prompt cho agent.


# Vấn đề khi xây dựng hệ thống ai-agent: làm sao để xây dựng intent-classfication 


# Tiếp tục xây dựng agent reason
# đầu tiên cần có một agent lập luận, xem truy vấn của user thuộc chủ đề n
# Con người khi giao tiếp và làm việc với nhau thì cần gì, nhận vấn đề ->
# xác nhận vấn đề -> vấn đề giải quyết thế nào

# Cần thiết kế lại mô hình theo hướng ReAct (Suy nghĩ hướng tới hành động)

# Tích hợp auto-cot.

# Tối ưu prompt.

# Build RAG pipeline.

#  `*state["messages"],` tự động đưa ngữ cảnh lịch sử trò chuyện vào mô hình bằng cách pass từng thành phần list vào.

# Ý tưởng sử dụng Multi-Agent gọi đến Yolov8 API, Yolov8 
# API sẽ lấy mọi hình ảnh cỡ nhỏ nó phát hiện  được là lỗi và 
# đưa vào mô hình llama3.2-11b-vision để đọc ảnh, sau đó 
# Llama3.2-11b-vision gửi lại text đến Multi-Agent để 
# Multi-Agent xác định xem đấy có phải là lỗi không.

# Nếu muốn vậy thì hiện tại ta cần có data-agent và model-agent
# data-agent để generate dữ liệu training, model-agent để viết 
# kiến trúc model vision.

# Bây giờ cần build tools trước cho mô hình, bao gồm 
# tool vision, tool genData, tool llama3.2-11b-vision-instruct

# Nhưng tại sao lại ko dùng vision yolov8 để finetune. Mục tiêu 
# ở đây, ta sẽ tận dụng cả sức mạnh của LLM-Vision, để hiểu rõ
# hình ảnh có gì, sau đó gửi về cho LLM-instruct để xử lý text 
# từ LLM-vision.

# **1. Image-to-Image Generation (CycleGAN, Pix2Pix, StyleGAN)**
# **2. Data Augmentation (Biến đổi dữ liệu)**

#  Lấy trạng thái, lịch sử bằng cách `AGENTIC.get_state(CONFIG).values`

#  Working on this, build integrate auto chain of thought to multi-agent
# x = cot(method="auto_cot", question="", debug=False)


# Truyền vào tham số phải là dạng dict theo role 