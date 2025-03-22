from langgraph.graph import( StateGraph, START, END)



from state import State 
from const_vars import (DEBUG, CHECKPOINTER, STORE)
from nodes import (
	manager_agent, 
	request_verify_adequacy_or_relevancy,
	request_verify_control_flow, 
	prompt_agent, 
	retrieval_augmented_planning_agent, 
)



WORKFLOW = StateGraph(State)

WORKFLOW.add_node("MANAGER_AGENT", manager_agent)
WORKFLOW.add_node("REQUEST_VERIFY", request_verify_adequacy_or_relevancy)
WORKFLOW.add_node("PROMPT_AGENT", prompt_agent)
WORKFLOW.add_node("RAP", retrieval_augmented_planning_agent)

WORKFLOW.add_edge(START, "MANAGER_AGENT")
WORKFLOW.add_edge("MANAGER_AGENT", "REQUEST_VERIFY")
WORKFLOW.add_conditional_edges("REQUEST_VERIFY", request_verify_control_flow, ["PROMPT_AGENT", END])
WORKFLOW.add_edge("PROMPT_AGENT", END)

AGENTIC = WORKFLOW.compile(
    store=STORE, debug=DEBUG, 
    checkpointer=CHECKPOINTER,
    name="foxconn_fulian_b09_ai_research_tranvantuan_v1047876"
)



# TODO: Tích hợp auto-cot.

# TODO: Tối ưu prompt.

# TODO: Build RAG pipeline.

# TODO:  `*state["messages"],` tự động đưa ngữ cảnh lịch sử trò chuyện vào mô hình bằng cách pass từng thành phần list vào.

# TODO: Ý tưởng sử dụng Multi-Agent gọi đến Yolov8 API, Yolov8 
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

# TODO: Lấy trạng thái, lịch sử bằng cách `AGENTIC.get_state(CONFIG).values`

# TODO: Working on this, build integrate auto chain of thought to multi-agent
# x = cot(method="auto_cot", question="", debug=False)


# Truyền vào tham số phải là dạng dict theo role 



