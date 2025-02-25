from langchain_community.llms.fake import FakeListLLM
from langchain_core.messages import (AIMessage, HumanMessage, ToolMessage, SystemMessage)
from langchain_core.prompts import PromptTemplate



tool_desc = PromptTemplate.from_template("""{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}""")
react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}""")



MODEL = FakeListLLM(responses=["""Chúc bạn có một ngày mới tốt lành, đây là FakeLMM!"""])



res = MODEL.invoke([
	SystemMessage(content="""You're a helpful assistant"""),
	HumanMessage(content="""What is the purpose of model regularization?""")    
])

print(res)
