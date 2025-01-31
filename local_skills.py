from langchain_core.runnables import ConfigurableField 
from langchain_core.tools import tool 
from langchain_openai import ChatOpenAI 

# Defines tools using concise function definitions 
@tool 
def multiply(x: float, y: float) -> float:
    """Multiply x times y."""
    return x * y 

@tool 
def exponentiate(x: float, y: float) -> float:
    """Raise x to the y."""
    return x ** y 

@tool
def add(x: float, y: float) -> float:
    """Add x and y."""
    return x + y 

# Register with tools class from LangChain 
tools = [multiply, exponentiate, add]

# Initialize the LLm with GPT-4o and bind the tools 
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

query = "What is 393 * 12.25? Also, what is 11 + 49?"
messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {
        "add": add, 
        "multiply": multiply, 
        "exponentiate": exponentiate
    }[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)

    print(f"{tool_msg.name} {tool_call['args']} {tool_msg.content}")
    messages.append(tool_msg)