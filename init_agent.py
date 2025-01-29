from langchain.tools import Tool 
from langchain_community.llms.ollama import Ollama 
from langchain.agents import AgentExecutor 
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.tools import tool 

from tools_for_agent import (
	recommend_maintenance_strategy, 
	release_resources, 
	remaining_useful_life_prediction,
	diagnose_fault_of_machine
)

def agent():
	global llm, agent_executor 
	if not llm:
		llm = Ollama(
			model="llama3.2:1b", 
			base_url="http://localhost:11434", 
			num_gpu=0, 
			temperature=0, 
			system="", 
			template="", 
			format="", 
		)
		# Define the tools 
		tools = [
				Tool(
					name="remaining_useful_life_prediction", 
					func=remaining_useful_life_prediction, 
					description="Predict the Remaining Useful Life (RUL) of a component based on the provided sensor data."),
				Tool(
					name="diagnose_fault_of_machine", 
					func=diagnose_fault_of_machine, 
					description="Identify the fault of a machine based on the provided sensor data."),
				Tool(
					name="recommend_maintenance_strategy", 
					func=recommend_maintenance_strategy, 
					description="Suggest the best maintenance strategy to minimize downtime and costs."),
			]

		# Extract tool names and descriptions 
		tool_names = ", ".join([tool.name for tool in tools])
		tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

		# ✅ Updated PromptTemplate
		prompt_template = PromptTemplate(
		input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
		template="""
			You are an AI assistant specializing in Prognostics and Health Management (PHM) for industrial systems. 
			Your responsibilities include diagnosing faults, predicting Remaining Useful Life (RUL), and recommending maintenance strategies.

			**TOOLS:**  
			{tools}

			**Available Tool Names (use exactly as written):**  
			{tool_names}

			**FORMAT:**  
			Thought: [Your reasoning]  
			Action: [Tool Name]  
			Action Input: [Input to the Tool as JSON]  
			Observation: [Result]  
			Final Answer: [Answer to the User]  

			**Examples:**  
			- To predict the Remaining Useful Life (RUL) of a component:  
				Thought: I need to predict the RUL of this component based on the provided sensor data.  
				Action: predict_rul_tool  
				Action Input: {{ 
					"sensor_data": {{ 
						"temperature": [75, 78, 79], 
						"vibration": [0.12, 0.15, 0.18], 
						"pressure": [101, 99, 98] 
					}}, 
					"operating_conditions": {{ 
						"load": 85, 
						"speed": 1500 
					}} 
				}}

			- To diagnose a fault in a rotating machine:  
				Thought: I need to identify the fault based on vibration and temperature data.  
				Action: diagnose_fault_tool  
				Action Input: {{ 
					"sensor_data": {{ 
						"vibration": [0.20, 0.35, 0.50], 
						"temperature": [90, 92, 94] 
					}} 
				}}

			- To recommend a maintenance strategy:  
				Thought: I need to suggest the best maintenance strategy to minimize downtime and costs.  
				Action: recommend_maintenance_tool  
				Action Input: {{ 
					"historical_data": {{ 
						"failures": 5, 
						"downtime_cost": 3000, 
						"maintenance_cost": 500 
					}}, 
					"failure_probability": 0.03 
				}}

			**Begin!**

			Question: {input}  
			{agent_scratchpad}"""
		)	
		logging.info(f"🛠️ Registered tools: {[tool.name for tool in tools]}")
		# ✅ Pass 'tool_names' and 'tools' to the agent
		agent = create_react_agent(
			llm=llm, 
			tools=tools, 
			prompt=prompt_template.partial(tool_names=tool_names, tools=tool_descriptions)
		)
		# ✅ AgentExecutor with error handling
		agent_executor = AgentExecutor(
			agent=agent, 
			tools=tools, 
			handle_parsing_errors=True, 
			verbose=True, 
			max_iterations=50 
		)
		logging.info("🚀 AgentExecutor initialized with tools.")
