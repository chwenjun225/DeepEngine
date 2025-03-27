from langgraph.graph import(
	StateGraph	, 
	START		, 
	END			,
)



from state import State 
from const_vars import (
	DEBUG			, 
	CHECKPOINTER	, 
	STORE			,
)
from nodes import (
	MANAGER_AGENT		,
	ROUTER_AGENT		,
	SYSTEM_AGENT		,
	ORCHESTRATE_AGENT	,
	REASONING_AGENT		,
	RESEARCH_AGENT		,
	PLANNING_AGENT		,
	EXECUTION_AGENT		,
	COMMUNICATION_AGENT	,
	EVALUATION_AGENT	,
	DEBUGGING_AGENT		,
)



AGENTS = [
	(	"MANAGER_AGENT"			, MANAGER_AGENT			, "Giao tiếp với người dùng"	, "core"	),
	(	"ROUTER_AGENT"			, ROUTER_AGENT			, "Định tuyến theo chủ đề"		, "logic"	),
	(	"SYSTEM_AGENT"			, SYSTEM_AGENT			, "Đảm bảo logic hệ thống"		, "control"	),
	(	"ORCHESTRATE_AGENT"		, ORCHESTRATE_AGENT		, "Điều phối agent"				, "control"	),
	(	"REASONING_AGENT"		, REASONING_AGENT		, "Suy luận yêu cầu"			, "core"	),
	(	"RESEARCH_AGENT"		, RESEARCH_AGENT		, "Tìm kiếm & hỗ trợ"			, "core"	),
	(	"PLANNING_AGENT"		, PLANNING_AGENT		, "Lập kế hoạch"				, "core"	),
	(	"EXECUTION_AGENT"		, EXECUTION_AGENT		, "Thực thi tác vụ"				, "exec"	),
	(	"DEBUGGING_AGENT"		, DEBUGGING_AGENT		, "Kiểm lỗi"					, "verify"	),
	(	"EVALUATION_AGENT"		, EVALUATION_AGENT		, "Đánh giá kết quả"			, "verify"	),
	(	"COMMUNICATION_AGENT"	, COMMUNICATION_AGENT	, "Tổng hợp phản hồi"			, "output"	),
]



WORKFLOW = StateGraph(State)

for name, func, desc, group in AGENTS:
	WORKFLOW.add_node(
		node=name,
		action=func,
		metadata={"description": desc, "group": group, "tags": [group, "agent"]}
	)

WORKFLOW.add_edge(	START					, 	"MANAGER_AGENT"			)
WORKFLOW.add_edge(	"MANAGER_AGENT"			,	"ROUTER_AGENT"			)
### Router -> (END | SYSTEM_AGENT) 
WORKFLOW.add_edge(	"SYSTEM_AGENT"			, 	"ORCHESTRATE_AGENT"		)
WORKFLOW.add_edge(	"ORCHESTRATE_AGENT"		,	"REASONING_AGENT"		)
WORKFLOW.add_edge(	"REASONING_AGENT"		, 	"RESEARCH_AGENT"		)
WORKFLOW.add_edge(	"RESEARCH_AGENT"		,	"PLANNING_AGENT"		)
WORKFLOW.add_edge(	"PLANNING_AGENT"		,	"EXECUTION_AGENT"		)
WORKFLOW.add_edge(	"DEBUGGING_AGENT"		, 	"EVALUATION_AGENT"		)
WORKFLOW.add_edge(	"EVALUATION_AGENT"		, 	"COMMUNICATION_AGENT"	)
WORKFLOW.add_edge(	"COMMUNICATION_AGENT"	, 	END						)

AGENTIC = WORKFLOW.compile(
	store=STORE, 
	debug=DEBUG, 
	checkpointer=CHECKPOINTER,
	name="maaoi"
)
