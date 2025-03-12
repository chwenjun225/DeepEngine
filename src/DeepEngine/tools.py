import math



from typing_extensions import Annotated, Optional



from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults



from langgraph.types import Command, interrupt



@tool(parse_docstring=True)
def ocr():
	pass 



@tool(parse_docstring=True)
def gen_img_data():
	pass 



@tool(parse_docstring=True)
def create_model_block():
	pass 



@tool(parse_docstring=True)
def add(a: float, b: float) -> float:
	"""Returns the sum of two numbers.

	Args:
		a (float): The first number.
		b (float): The second number.

	Returns:
		float: The sum of `a` and `b`.
	"""
	return a + b



@tool(parse_docstring=True)
def subtract(a: float, b: float) -> float:
	"""Returns the difference of two numbers.

	Args:
		a (float): The first number.
		b (float): The second number.

	Returns:
		float: The result of `a - b`.
	"""
	return a - b



@tool(parse_docstring=True)
def multiply(a: float, b: float) -> float:
	"""Returns the product of two numbers.

	Args:
		a (float): The first number.
		b (float): The second number.

	Returns:
		float: The result of `a * b`.
	"""
	return a * b



@tool(parse_docstring=True)
def divide(a: float, b: float) -> float:
	"""Returns the quotient of two numbers.

	Args:
		a (float): The dividend.
		b (float): The divisor.

	Returns:
		float: The result of `a / b`.

	Raises:
		ValueError: If `b` is zero.
	"""
	if b == 0:
		raise ValueError("Division by zero is not allowed.")
	return a / b



@tool(parse_docstring=True)
def power(a: float, b: float) -> float:
	"""Returns the result of raising `a` to the power of `b`.

	Args:
		a (float): The base number.
		b (float): The exponent.

	Returns:
		float: The result of `a ** b`.
	"""
	return a ** b



@tool(parse_docstring=True)
def square_root(a: float) -> float:
	"""Returns the square root of a number.

	Args:
		a (float): The number to find the square root of.

	Returns:
		float: The square root of `a`.

	Raises:
		ValueError: If `a` is negative.
	"""
	if a < 0:
		raise ValueError("Cannot compute the square root of a negative number.")
	return math.sqrt(a)



@tool(parse_docstring=True)
def human_assistance(
		user_prompt: str, 
		thought: str, 
		action: str, 
		action_input: str, 
		observation: str, 
		tool_call_id: Annotated[str, InjectedToolCallId]
	) -> str:
	"""Request assistance from a human.

	Args:
		user_prompt (str): The original question provided by the user.
		thought (str): The reasoning before executing an action.
		action (str): The selected action.
		action_input (str): The input required for the action.
		observation (str): The outcome of executing the action.
		tool_call_id (str): The unique identifier for the tool call.

	Returns:
		str: A command object containing the updated state after human verification.
	"""
	human_response = interrupt({
		"question": "Is this correct?",
		"user_prompt": user_prompt,
		"thought": thought,
		"action": action,
		"action_input": action_input,
		"observation": observation,
	})
	if human_response.get("correct", "").lower().startswith("y"):
		verified_user_prompt = user_prompt
		verified_thought = thought
		verified_action = action
		verified_action_input = action_input
		verified_observation = observation
		response = "Correct"
	else:
		verified_user_prompt = human_response.get("user_prompt", user_prompt)
		verified_thought = human_response.get("thougt", thought)
		verified_action = human_response.get("action", action)
		verified_action_input = human_response.get("action_input", action_input)
		verified_observation = human_response.get("observation", observation)
		response = f"Made a correction: {human_response}"
	state_update = {
		"user_prompt": verified_user_prompt, 
		"thought": verified_thought, 
		"action": verified_action, 
		"action_input": verified_action_input, 
		"observation": verified_observation, 
		"messages": [ToolMessage(response, tool_call_id=tool_call_id)], 
	}
	return Command(update=state_update)



@tool(parse_docstring=True)
def random_number_maker(user_prompt: Optional[str]) -> str:
	"""Generates a random number between 0 and 100.

	Args:
		user_prompt (Optional[str], optional): The input prompt from the user (not used in this function).

	Returns:
		str: A random number between 0 and 100.
	"""
	import random 
	return str(random.randint(0, 100))



@tool(parse_docstring=True)
def text_to_image(user_prompt: str) -> dict:
	"""Generates an image based on a text description.

	Args:
		user_prompt (str): The text prompt describing the desired image.

	Returns:
		dict: A dictionary containing the image URL.
	"""
	import json
	return json.dumps({"image_url": f"https://image.pollinations.ai/prompt/{user_prompt}"}, ensure_ascii=False)



@tool(parse_docstring=True)
def tavily_search(user_prompt: str):
	"""Perform a web search using the Tavily API and return the top results.

	Args:
		user_prompt (str): The search query entered by the user.

	Returns:
		TavilySearchResults: The top search results retrieved from Tavily API.
	"""
	return TavilySearchResults(max_results=2)
