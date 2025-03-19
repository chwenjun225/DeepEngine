from typing_extensions import (
    TypedDict, 
    Dict, 
    List, 
    Annotated
)



from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import (
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    AnyMessage
)



from langgraph.graph import (
    StateGraph, 
    START, 
    END
)
from langgraph.graph.message import add_messages



class State(TypedDict):
    """Lưu giữ trạng thái của các Agent."""
    messages: Annotated[List[Dict|str|AnyMessage], add_messages]



def SYSTEM_AGENT(state: State) -> State:
    """Quản lý toàn cục."""
    return state 



def REASONING_AGENT(state: State) -> State:
    """Suy luận và phân tích."""
    return state 



def RESEARCH_AGENT(state: State) -> State:
    """Thu thập thông tin."""
    return state 



def PLANNING_AGENT(state: State) -> State:
    """Lập kế hoạch và ra quyết định."""
    return state 



def EXECUTION_AGENT(state: State) -> State:
    """Thực thi hành động."""
    return state     



def COMMUNICATION_AGENT(state: State) -> State:
    """Tóm tắt và giao tiếp với người dùng."""
    return state 



def EVALUATION_AGENT(state: State) -> State:
    """Đánh giá và kiểm tra chất lượng."""
    return state 



def DEBUG_AGENT(state: State):
    """Kiểm tra và sửa lỗi."""
    return state 

# Hệ thống này là react agent nó có khả năng reasoning mạnh và thực hiện hành động.
worflow = StateGraph(State)

