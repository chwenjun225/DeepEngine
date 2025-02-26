import ast 
import json 
from typing_extensions import Union

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from state import State

@tool
def calculator(query: str) -> str:
	"""A simple calculator tool. Input should be a mathematical expression."""
	return ast.literal_eval(query)

@tool
def get_weather(state: State):
	"""Call to get the current weather."""
	if state.location.lower() in ["tokyo", "san francisco", "hanoi"]:
		return "It's 25 degrees, the weather is beautiful with cloud."
	else:
		return f"Sorry, I don't have weather information for {state.location}."

@tool
def get_coolest_cities():
	"""Get a list of coolest cities"""
	return "nyc, sf"

@tool
def recommend_maintenance_strategy(input_str: str):
	"""Đề xuất chiến lược bảo trì."""
	if input_str is not None: 
		result = {
				"strategy": ["Preventive Maintenance"], 
				"justification": ["Failure probability is 0.03"]
			}
	return json.dump(result, indent=4)

@tool
def diagnose_fault_of_machine(input_str: str):
	"""Chẩn đoán lỗi máy móc."""
	if input_str is not None: 
		result = {
				"fault": ["Overheating", "Coolant System Failure"], 
				"recommendation": ["Reduce workload and check cooling system"]
			}
	return json.dump(result, indent=2)

@tool
def remaining_useful_life_prediction(input_str: str):
	"""Dự đoán tuổi thọ còn lại của một thành phần thiết bị."""
	if input_str is not None: 
		result = {
			"predicted_rul": [150],  
			"confidence": [0.85], 
			"recommendations": ["Reduce operating load to extend lifespan"]
		}
	return json.dumps(result, indent=2)

@tool
def add(*numbers: Union[int, float]) -> Union[int, float]:
	"""Add multiple numbers (integers or floats).
	Args:
		numbers: A list of numbers to add.
	Returns:
		The sum of all given numbers, as an integer if all inputs are integers, otherwise a float.
	"""
	if not numbers:
		raise ValueError("At least one number must be provided for addition.")
	result = sum(numbers)
	return int(result) if all(isinstance(num, int) for num in numbers) else result

@tool
def multiply(*numbers: Union[int, float]) -> Union[int, float]:
	"""Multiply multiple numbers (integers or floats).
	Args:
		numbers: A list of numbers to multiply.
	Returns:
		The product of all given numbers, as an integer if all inputs are integers, otherwise a float.
	"""
	if not numbers:
		raise ValueError("At least one number must be provided for multiplication.")
	result = 1
	for num in numbers:
		result *= num
	return int(result) if all(isinstance(num, int) for num in numbers) else result
