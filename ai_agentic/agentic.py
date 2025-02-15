from uuid import uuid4 
from pprint import pprint
import fire 
from datetime import datetime
from transformers import AutoTokenizer
from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_core.messages import ToolCall, AIMessage, HumanMessage, ToolMessage, trim_messages # TODO: Cần thêm tính năng lọc và cắt tin nhắn

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from state import SupervisorDecision, State, Input, Output
from prompts import generate_prompt, reflection_prompt, system_prompt_part_1, system_prompt_part_2
from tools_use import DuckDuckGoSearchRun, calculator

if True:
	tokenizer = AutoTokenizer.from_pretrained("/home/chwenjun225/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B")
	embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

	search = DuckDuckGoSearchRun()
	tools = [search, calculator]

	model = ChatOpenAI(model_name="/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct", openai_api_base="http://127.0.0.1:2026/v1", openai_api_key="chwenjun225", temperature=0.1)

	tools_retriever = InMemoryVectorStore.from_documents(
		[Document(tool.description, metadata={"name": tool.name}) for tool in tools],
		embeddings,
	).as_retriever()

	config = {"configurable": {"thread_id": "1"}}

def token_counter(messages):
	"""Đếm số lượng token từ danh sách tin nhắn."""
	text = " ".join([msg.content for msg in messages])
	return len(tokenizer.encode(text)) 

def select_tools(state: State) -> State:
	query = state["messages"][-1].content
	tool_docs = tools_retriever.invoke(query)
	return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}

def reflect(state: State) -> State:
	class_map = {
		AIMessage: HumanMessage, 
		HumanMessage: AIMessage, 
		ToolMessage: HumanMessage 
	}
	translated = [reflection_prompt, state["messages"][0]] + [
		class_map[msg.__class__](content=msg.content) 
		for msg in state["messages"][1:]
	]
	answer = model.invoke(translated)
	return {"messages": [HumanMessage(content=answer.content)]}

def should_continue(state: State):
	if len(state["messages"]) > 6:
		return END
	else:
		return "reflect"

def chatbot(state: State) -> State:
	selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
	answer = model.bind_tools(selected_tools).invoke([generate_prompt] + state["messages"])
	return {"messages": [answer]}

def main():
	"""Thực thi chương trình."""
	builder = StateGraph(State)

	builder.add_node("select_tools", select_tools)
	builder.add_node("chatbot", chatbot)
	builder.add_node("tools", ToolNode(tools))
	builder.add_node("reflect", reflect)

	builder.add_edge(START, "select_tools")
	builder.add_edge("select_tools", "chatbot")
	builder.add_conditional_edges("chatbot", tools_condition)
	builder.add_edge("tools", "chatbot")
	builder.add_conditional_edges("chatbot", should_continue)
	builder.add_edge("reflect", "chatbot")
	
	graph = builder.compile(checkpointer=MemorySaver())

	user_input = {
		"messages": [HumanMessage("""What is Large Language Model?""")]
	}
	for chunk in graph.stream(user_input, config):
		print(chunk)

if __name__ == "__main__":
	fire.Fire(main)


















































































"""
(Foxer_env) (base) chwenjun225@chwenjun225:~/Projects/Foxer/ai_agentic$ python agentic.py 
{'select_tools': {'selected_tools': ['duckduckgo_search', 'calculator']}}
{'chatbot': 
	{'messages': 
		[AIMessage(
			content='', 
			additional_kwargs={
				'tool_calls': [{'id': 'chatcmpl-tool-2de73a0da7e2431abb750c9d0e16a92a', 'function': {'arguments': '{"properties": {"query": "Large Language Model"}}', 'name': duckduckgo_search'}, 'type': 'function'}], 
				'refusal': None
			}, 
			response_metadata={
				'token_usage': {'completion_tokens': 25, 'prompt_tokens': 341, 'total_tokens': 366, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 
				'model_name': '/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct', 
				'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, 
				id='run-f0d947c2-f55d-4a55-a89c-36bdba732974-0', 
				tool_calls=[{'name': 'duckduckgo_search', 'args': {'properties': {'query': 'Large Language Model'}}, 'id': 'chatcmpl-tool-2de73a0da7e2431abb750c9d0e16a92a', 'type': 'tool_call'}], 
				usage_metadata={'input_tokens': 341, 'output_tokens': 25, 'total_tokens': 366, 'input_token_details': {}, 'output_token_details': {}})]}}
{'tools': {'messages': [ToolMessage(
	content="Error: 1 validation error for DDGInput\nquery\n  Field required [type=missing, input_value={'properties': {'query': 'Large Language Model'}}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.10/v/missing\n Please fix your mistakes.", name='duckduckgo_search', tool_call_id='chatcmpl-tool-2de73a0da7e2431abb750c9d0e16a92a', status='error')]}}
{'reflect': {'messages': [HumanMessage(content='A Large Language Model (LLM) is a type of artificial intelligence (AI) model that is trained on a massive corpus of text data, typically consisting of a large volume of books, articles, and other written content. The goal of an LLM is to learn the patterns and relationships within the text data, and to use this knowledge to generate human-like text.\n\nLarge Language Models are trained using a process called deep learning, which involves layering multiple neural networks to process and analyze the text data. The training data is typically a massive corpus of text, which is then fed into the neural networks to learn the patterns and relationships within the data.\n\nSome key characteristics of Large Language Models include:\n\n1. **Massive training data**: Large Language Models are trained on enormous amounts of text data, which allows them to learn the patterns and relationships within the data.\n2. **Deep neural networks**: Large Language Models are built using deep neural networks, which are composed of multiple layers of interconnected nodes (neurons) that process and analyze the text data.\n3. **Self-supervised learning**: Large Language Models are trained using self-supervised learning, which involves providing the model with labeled data (e.g. text pairs) and allowing it to learn the patterns and relationships within the data.\n4. **Generative capabilities**: Large Language Models can generate human-like text, including sentences, paragraphs, and even entire articles.\n\nThe benefits of Large Language Models include:\n\n1. **Improved language understanding**: Large Language Models can learn to understand the nuances of language, including idioms, colloquialisms, and figurative language.\n2. **Increased text generation capabilities**: Large Language Models can generate high-quality text, including articles, stories, and even entire books.\n3. **Enhanced language translation**: Large Language Models can be used for language translation, allowing for more accurate and efficient translation of text.\n\nHowever, Large Language Models also have some limitations and challenges, including:\n\n1. **Data quality and availability**: Large Language Models require high-quality training data, which can be difficult to obtain and maintain.\n2. **Bias and fairness**: Large Language Models can perpetuate biases and prejudices present in the training data, which can lead to unfair and discriminatory outcomes.\n3. **Explainability and transparency**: Large Language Models can be difficult to interpret and understand, making it challenging to explain their decisions and actions.\n\nIn the context of the essay submission you provided earlier, I would recommend the following:\n\n* **Length**: The essay is a bit long for a Large Language Model, which typically requires training data that is several hundred million tokens long. Consider breaking the essay into multiple sections or chapters to make it more manageable.\n* **Depth**: While the essay provides some interesting insights and analysis, it could benefit from more in-depth exploration of the topics. Consider adding more supporting evidence and examples to make the essay more convincing.\n* **Style**: The essay has a somewhat formal tone, which may not be suitable for a Large Language Model. Consider using a more conversational tone to make the essay more engaging and accessible.\n* **Organization**: The essay jumps around between different topics and ideas. Consider organizing the essay into clear sections or chapters to make it easier to follow.\n\nOverall, the essay has some interesting ideas and insights, but could benefit from more depth, organization, and clarity. With some revisions and refinements, the essay could be even stronger.', additional_kwargs={}, response_metadata={})]}}
{'chatbot': {'messages': [AIMessage(content='<|python_tag|>{"type": "function", "function": {"name": "duckduckgo_search", "parameters": {"properties": "{\'query\': \'Large Language Model\'}"}}', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 36, 'prompt_tokens': 1134, 'total_tokens': 1170, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': '/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-080cad33-0179-44d0-89c2-3c0c6720606f-0', usage_metadata={'input_tokens': 1134, 'output_tokens': 36, 'total_tokens': 1170, 'input_token_details': {}, 'output_token_details': {}})]}}
{'reflect': {'messages': [HumanMessage(content='It seems like you\'re trying to use the DuckDuckGo search engine, but the error message is indicating that the `duckduckgo_search` function is not properly defined.\n\nThe `duckduckgo_search` function is a part of the `pydantic` library, which is used for building robust, scalable, and maintainable data models. However, it\'s not a built-in function in the `pydantic` library.\n\nTo fix the error, you need to import the `duckduckgo_search` function from the `pydantic` library. Here\'s how you can do it:\n\n```python\nfrom pydantic import BaseModel\n\nclass LargeLanguageModel(BaseModel):\n    query: str\n\n    def search(self, query):\n        # Implement your search logic here\n        # For example, you can use the DuckDuckGo search engine\n        response = duckduckgo_search(query)\n        return response\n\n# Example usage:\nmodel = LargeLanguageModel(query="Large Language Model")\nresult = model.search("Large Language Model")\nprint(result)\n```\n\nIn this example, we define a `LargeLanguageModel` class that inherits from `pydantic.BaseModel`. The `search` method is used to perform the search query and return the result.\n\nPlease note that you\'ll need to install the `pydantic` library if you haven\'t already done so. You can do this by running the following command in your terminal:\n\n```bash\npip install pydantic\n```\n\nAlso, make sure that the `duckduckgo_search` function is implemented correctly and returns the expected response.', additional_kwargs={}, response_metadata={}, id='8c93be59-ead7-4225-9ee0-cc520dc5e810')]}}
{'chatbot': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'chatcmpl-tool-018c71ba9ca04f78a1d2bbb6c5eed631', 'function': {'arguments': '{"properties": "{\'query\': \'Large Language Model\'}"}', 'name': 'duckduckgo_search'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 33, 'prompt_tokens': 1506, 'total_tokens': 1539, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': '/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-708b6eff-a4c3-469f-9538-35fc1d09d5a4-0', tool_calls=[{'name': 'duckduckgo_search', 'args': {'properties': "{'query': 'Large Language Model'}"}, 'id': 'chatcmpl-tool-018c71ba9ca04f78a1d2bbb6c5eed631', 'type': 'tool_call'}], usage_metadata={'input_tokens': 1506, 'output_tokens': 33, 'total_tokens': 1539, 'input_token_details': {}, 'output_token_details': {}})]}}
{'tools': {'messages': [ToolMessage(content='Error: 1 validation error for DDGInput\nquery\n  Field required [type=missing, input_value={\'properties\': "{\'query\':...Large Language Model\'}"}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.10/v/missing\n Please fix your mistakes.', name='duckduckgo_search', id='4127a4d1-4c4f-4ef4-a006-7fcb21c07050', tool_call_id='chatcmpl-tool-018c71ba9ca04f78a1d2bbb6c5eed631', status='error')]}}
{'chatbot': {'messages': [AIMessage(content='<|python_tag|>from pydantic import BaseModel\n\nclass LargeLanguageModel(BaseModel):\n    query: str\n\n    def search(self, query):\n        # Implement your search logic here\n        # For example, you can use the DuckDuckGo search engine\n        response = {\n            "url": "https://www.duckduckgo.com/",\n            "result": {\n                "title": "Large Language Model",\n                "snippet": "Large language models are a type of artificial intelligence that can understand and generate human-like text.",\n                "link": "https://duckduckgo.com/?q=Large+Language+Model"\n            }\n        }\n        return response\n\n# Example usage:\nmodel = LargeLanguageModel(query="Large Language Model")\nresult = model.search("Large Language Model")\nprint(result)', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 162, 'prompt_tokens': 1610, 'total_tokens': 1772, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': '/home/chwenjun225/Projects/Foxer/models/Llama-3.2-1B-Instruct', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-43394e31-b474-4e8b-9624-2f2171beeef7-0', usage_metadata={'input_tokens': 1610, 'output_tokens': 162, 'total_tokens': 1772, 'input_token_details': {}, 'output_token_details': {}})]}}

"""
