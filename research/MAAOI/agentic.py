from langgraph.graph import(
	StateGraph, 
	START, 
	END,
)



from state import State 
from const import (
	DEBUG,
	CHECKPOINTER,
	STORE,
)
from nodes import (
	TEMPORAL_PATTERN_AGENT					,
	DEFECT_REASONING_AGENT					,
	QUALITY_CONTROL_JUDGEMENT_AGENT	,
	VISUAL_AGENT										,
)



AGENTS = [	
	("TEMPORAL_PATTERN_AGENT", TEMPORAL_PATTERN_AGENT, "Phân loại lỗi.", "logic, non-LLM"),
	("DEFECT_REASONING_AGENT", DEFECT_REASONING_AGENT, "Đặt câu hỏi về lỗi.", "reasoning, LLM"),
	("QUALITY_CONTROL_JUDGEMENT_AGENT", QUALITY_CONTROL_JUDGEMENT_AGENT	, "Đưa ra quyết định là OK/NG.", "logic, LLM"),
	("VISUAL_AGENT", VISUAL_AGENT, "Đưa ra tọa độ lỗi cuối cùng.",	"logic, non-LLM"	),
]	



WORKFLOW = StateGraph(State)

for name, the_func, desc, group in AGENTS:
	WORKFLOW.add_node(
		node=name,
		action=the_func,
		metadata={
			"description": desc,
			"group": group,
			"tags": [group, "agent"],
	})

WORKFLOW.add_edge(START, "TEMPORAL_PATTERN_AGENT")
WORKFLOW.add_edge("TEMPORAL_PATTERN_AGENT", "DEFECT_REASONING_AGENT") 
WORKFLOW.add_edge("DEFECT_REASONING_AGENT", "QUALITY_CONTROL_JUDGEMENT_AGENT")
WORKFLOW.add_edge("QUALITY_CONTROL_JUDGEMENT_AGENT",	"VISUAL_AGENT")
WORKFLOW.add_edge("VISUAL_AGENT",	END)

AGENTIC = WORKFLOW.compile(
	store=STORE,
	debug=DEBUG,
	checkpointer=CHECKPOINTER,
	name="tranvantuan",
)
