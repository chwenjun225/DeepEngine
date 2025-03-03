import random 
import json



from typing_extensions import Annotated



from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults



from langgraph.types import Command, interrupt


@tool
def human_assistance(
		user_prompt: str, 
		thought: str, 
		action: str, 
		action_input: str, 
		observation: str, 
		justification: str, 
		tool_call_id: Annotated[str, InjectedToolCallId]
	) -> str:
	"""Request assistance from a human."""
	human_response = interrupt({
		"question": "Is this correct?",
		"user_prompt": user_prompt,
		"thought": thought,
		"action": action,
		"action_input": action_input,
		"observation": observation,
		"justification": justification
	})
	if human_response.get("correct", "").lower().startswith("y"):
		verified_user_prompt = user_prompt
		verified_thought = thought
		verified_action = action
		verified_action_input = action_input
		verified_observation = observation
		verified_justification = justification
		response = "Correct"
	else:
		verified_user_prompt = human_response.get("user_prompt", user_prompt)
		verified_thought = human_response.get("thougt", thought)
		verified_action = human_response.get("action", action)
		verified_action_input = human_response.get("action_input", action_input)
		verified_observation = human_response.get("observation", observation)
		verified_justification = human_response.get("justification", justification)
		response = f"Made a correction: {human_response}"
	state_update = {
		"user_prompt": verified_user_prompt, 
		"thought": verified_thought, 
		"action": verified_action, 
		"action_input": verified_action_input, 
		"observation": verified_observation, 
		"justification": verified_justification, 
		"messages": [ToolMessage(response, tool_call_id=tool_call_id)], 
	}
	return Command(update=state_update)



@tool
def random_number_maker(user_prompt: str) -> int:
	"""Generates a random number between 0 and 100."""
	return random.randint(0, 100)



@tool
def text_to_image(user_prompt: str) -> dict:
	"""Generates an image based on a text description."""
	return json.dumps({"image_url": f"https://image.pollinations.ai/prompt/{user_prompt}"}, ensure_ascii=False)



@tool
def tavily_search(user_prompt: str):
	"""Perform a web search using the Tavily API and return the top results."""
	return TavilySearchResults(max_results=2)
