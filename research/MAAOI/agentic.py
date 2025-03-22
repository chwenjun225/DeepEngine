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
