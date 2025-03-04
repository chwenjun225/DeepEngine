from langchain_core.prompts import PromptTemplate



begin_of_text = "<|begin_of_text|>"
end_of_text = "<|end_of_text|>"
start_header_id = "<|start_header_id|>"
end_header_id = "<|end_header_id|>"
end_of_message_id = "<|eom_id|>"
end_of_turn_id = "<|eot_id|>"



tool_desc_prompt = PromptTemplate.from_template(
	"""{name_for_model}: Call this tool to interact with the {name_for_human} API. 
What is the {name_for_human} API useful for? 
{description_for_model}.
Type: {type}.
Properties: {properties}.
Required: {required}."""
)



react_prompt = PromptTemplate.from_template(
"""{begin_of_text}
{start_header_id}system{end_header_id}
You are an AI assistant that follows the ReAct reasoning framework. 
You have access to the following APIs:

{tools_desc}

Use the following strict format:

### Input Format:

user_query: The original query provided by the user.
thought: Logical reasoning before executing an action.
action: The action to be taken, chosen from available tools: {tools_name}.
action_input: The required input for the action.
observation: The outcome of executing the action. 
...(Repeat the thought/action/observation loop as needed)
final_thought: I now know the final answer.
final_answer: Provide the final answer.

Begin!

Question:{end_of_turn_id}

{start_header_id}user{end_header_id}
{user_query}{end_of_turn_id}

{start_header_id}assistant{end_header_id}"""
)
