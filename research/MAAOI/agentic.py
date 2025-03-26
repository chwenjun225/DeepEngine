from langgraph.graph import(
	StateGraph					, 
	START						, 
	END							,
)



from state import State 
from const_vars import (
	DEBUG						, 
	CHECKPOINTER				, 
	STORE						,
)
from nodes import (
	MANAGER_AGENT				,
	ROUTER_AGENT				,
	SYSTEM_AGENT				,
	ORCHESTRATE_AGENT			,
	REASONING_AGENT				,
	RESEARCH_AGENT				,
	PLANNING_AGENT				,
	EXECUTION_AGENT				,
	COMMUNICATION_AGENT			,
	EVALUATION_AGENT			,
	DEBUGGING_AGENT				,
)
from utils import (
	passthrough
)



WORKFLOW = StateGraph(State)

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

for name, func, desc, group in AGENTS:
	WORKFLOW.add_node(
		node=name,
		action=func,
		metadata={
			"description": desc,
			"group": group,
			"tags": [group, "agent"]
		}
	)

WORKFLOW.add_edge(	start_key=START					, 	end_key="MANAGER_AGENT"			)
WORKFLOW.add_edge(	start_key="MANAGER_AGENT"		, 	end_key="ROUTER_AGENT"			)

WORKFLOW.add_conditional_edges(	
	source="ROUTER_AGENT"							, 
	path={
		"END": passthrough()						, 
		"SYSTEM_AGENT": passthrough()				, 
	}												, 
)
WORKFLOW.add_edge(	start_key="SYSTEM_AGENT"		, 	end_key="ROUTER_AGENT"			)
WORKFLOW.add_edge(	start_key="SYSTEM_AGENT"		, 	end_key="ORCHESTRATE_AGENT"		)
WORKFLOW.add_edge(	start_key="ORCHESTRATE_AGENT"	,	end_key="REASONING_AGENT"		)
WORKFLOW.add_edge(	start_key="REASONING_AGENT"		, 	end_key="RESEARCH_AGENT"		)
WORKFLOW.add_edge(	start_key="RESEARCH_AGENT"		,	end_key="PLANNING_AGENT"		)
WORKFLOW.add_edge(	start_key="PLANNING_AGENT"		,	end_key="EXECUTION_AGENT"		)

WORKFLOW.add_conditional_edges(
	source="EXECUTION_AGENT"						,
	path={
		"DEBUGGING_AGENT": passthrough()			, 
		"EVALUATION_AGENT": passthrough()			, 
	}
)
WORKFLOW.add_edge(	start_key="DEBUGGING_AGENT"		, 	end_key="EXECUTION_AGENT"		)
WORKFLOW.add_edge(	start_key="EVALUATION_AGENT"	, 	end_key="EXECUTION_AGENT"		)
WORKFLOW.add_edge(	start_key="DEBUGGING_AGENT"		, 	end_key="EVALUATION_AGENT"		)
WORKFLOW.add_edge(	start_key="EVALUATION_AGENT"	, 	end_key="COMMUNICATION_AGENT"	)
WORKFLOW.add_edge(	start_key="COMMUNICATION_AGENT"	, 	end_key=END						)

AGENTIC = WORKFLOW.compile(
	store=STORE, 
	debug=DEBUG, 
	checkpointer=CHECKPOINTER,
	name="maaoi"
)
