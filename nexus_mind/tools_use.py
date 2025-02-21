import json 

from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper

def tool_wrapper_for_qwen(tool):
	def tool_(query):
		query = json.loads(query)["query"]
		return tool.run(query)
	return tool_


TOOLS = [
	{
		'name_for_human':
			'google search',
		'name_for_model':
			'Search',
		'description_for_model':
			'useful for when you need to answer questions about current events.',
		'parameters': [{
			"name": "query",
			"type": "string",
			"description": "search query of google",
			'required': True
		}], 
		'tool_api': tool_wrapper_for_qwen(search)
	},
	{
		'name_for_human':
			'Wolfram Alpha',
		'name_for_model':
			'Math',
		'description_for_model':
			'Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life.',
		'parameters': [{
			"name": "query",
			"type": "string",
			"description": "the problem to solved by Wolfram Alpha",
			'required': True
		}], 
		'tool_api': tool_wrapper_for_qwen(WolframAlpha)
	},  
	{
		'name_for_human':
			'arxiv',
		'name_for_model':
			'Arxiv',
		'description_for_model':
			'A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles on arxiv.org.',
		'parameters': [{
			"name": "query",
			"type": "string",
			"description": "the document id of arxiv to search",
			'required': True
		}], 
		'tool_api': tool_wrapper_for_qwen(arxiv)
	},
	{
		'name_for_human':
			'python',
		'name_for_model':
			'python',
		'description_for_model':
			"A Python shell. Use this to execute python commands. When using this tool, sometimes output is abbreviated - Make sure it does not look abbreviated before using it in your answer. "
			"Don't add comments to your python code.",
		'parameters': [{
			"name": "query",
			"type": "string",
			"description": "a valid python command.",
			'required': True
		}],
		'tool_api': tool_wrapper_for_qwen(python)
	}

]