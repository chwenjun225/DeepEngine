from langgraph.graph import(
	StateGraph					, 
	START						, 
	END							,
)



from state import State 
from const import (
	DEBUG						,
	CHECKPOINTER				,
	STORE						,
)
from nodes import (
	TEMPORAL_PATTERN_AGENT		,
	DEFECT_REASONING_AGENT		,
	CRITICAL_ASSESSMENT_AGENT	,
	REPORT_GENERATOR_AGENT		,
	VISUAL_AGENT				,
)



AGENTS = [
	(	"TEMPORAL_PATTERN_AGENT"	, TEMPORAL_PATTERN_AGENT	, "Phân loại lỗi."					, "logic, non-LLM"	),
	(	"DEFECT_REASONING_AGENT"	, DEFECT_REASONING_AGENT	, "Đặt câu hỏi về lỗi."				, "reasoning, LLM"	),
	(	"CRITICAL_ASSESSMENT_AGENT"	, CRITICAL_ASSESSMENT_AGENT	, "Đánh giá độ nghiêm trọng lỗi."	, "logic, LLM"		),
	(	"REPORT_GENERATOR_AGENT"	, REPORT_GENERATOR_AGENT	, "Báo cáo kết quả cho người dùng"	, "logic, non-LLM"	),
	(	"VISUAL_AGENT"				, VISUAL_AGENT				, "Đưa ra tọa độ lỗi cuối cùng."	, "logic, non-LLM"	),
]	



WORKFLOW = StateGraph(State)

for name, func, desc, group in AGENTS:
	WORKFLOW.add_node(
		node=name						,
		action=func						,
		metadata={
			"description": desc			, 
			"group": group, 
			"tags": [group, "agent"]
	})

WORKFLOW.add_edge(	START						, 	"TEMPORAL_PATTERN_AGENT"		)
WORKFLOW.add_edge(	"TEMPORAL_PATTERN_AGENT"	,	"DEFECT_REASONING_AGENT"		) 
WORKFLOW.add_edge(	"DEFECT_REASONING_AGENT"	, 	"CRITICAL_ASSESSMENT_AGENT"		)
WORKFLOW.add_edge(	"CRITICAL_ASSESSMENT_AGENT"	,	"REPORT_GENERATOR_AGENT"		)
WORKFLOW.add_edge(	"CRITICAL_ASSESSMENT_AGENT"	,	"VISUAL_AGENT"					)

AGENTIC = WORKFLOW.compile(
	store=STORE, 
	debug=DEBUG, 
	checkpointer=CHECKPOINTER,
	name="tranvantuan"
)
