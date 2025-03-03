from langchain_core.prompts import PromptTemplate



react_tool_desc = PromptTemplate.from_template("""{name_for_model}: Call this tool to interact with the {name_for_human} API. 
What is the {name_for_human} API useful for? 
{description_for_model}.
Parameters: {parameters}""")



react_prompt = PromptTemplate.from_template("""You are an AI assistant that follows the ReAct reasoning framework. 
You have access to the following APIs:

{react_tool_desc}

Use the following strict format:

### Input Format:

question: [The input question]
thought: [Think logically about the next step]
action: [Select from available tools: {tools_name}]
action_input: [Provide the required input]
observation: [Record the output from the action]
... (Repeat the thought/action/observation loop as needed)
final_thought: I now know the final answer
final_answer: [Provide the final answer]

Begin!

Question: {query}""")