import fire
import logging
import urllib3
import requests 
from datetime import datetime
import json 

import streamlit as st

from langchain.tools import Tool
from langchain_community.llms.ollama import Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts.prompt import PromptTemplate

logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if "chat_history" not in st.session_state:
	st.session_state.chat_history = []
if "agent_executor" not in st.session_state:
	st.session_state.agent_executor = None

def remaining_useful_life_prediction(input_str=""):
	"""Dự đoán tuổi thọ còn lại của một thành phần thiết bị."""
	try: # Ví dụ giả lập 
		result = {
			"predicted_rul": 150,  # Giả sử dự đoán còn 150 giờ sử dụng
			"confidence": 0.85,  # Độ tin cậy 85% 
			"recommendations": "Reduce operating load to extend lifespan."
		}
		return json.dumps(result, indent=4)
	except Exception as e:
		logging.error(f"Error in RUL prediction: {str(e)}")
		return "Error in predicting RUL. Please check the input data."

def diagnose_fault_of_machine(input_str=""):
	"""Chẩn đoán lỗi máy móc."""
	return json.dumps(
		{
			"fault": "Overheating", 
			"recommendation": "Reduce workload and check cooling system"
		}, indent=4
	)

def recommend_maintenance_strategy(input_str=""):
	"""Đề xuất chiến lược bảo trì."""
	return json.dumps(
		{
			"strategy": "Preventive Maintenance", 
			"justification": "Failure probability is 0.03"
		}, indent=4
	)

def initialize_agent():
	if st.session_state.agent_executor is not None:
		return  

	llm = Ollama(
		model="llama3.2:1b",
		base_url="http://localhost:11434",
		num_gpu=0,
		temperature=0,
		num_ctx=4096, 
		num_predict=2048, 
	)

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

	tool_names = ", ".join([tool.name for tool in tools])
	tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

	prompt_template = PromptTemplate(
		input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
		template="""
		You are an AI assistant specializing in Prognostics and Health Management (PHM) for industrial systems.
		Your responsibilities include diagnosing faults, predicting Remaining Useful Life (RUL), and recommending maintenance strategies.
		At the end of the answer you should summarize the result and provide recommendations.
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
			Thought: I will predict the RUL of this component based on the provided sensor data.
			Action: remaining_useful_life_prediction
			Action Input: {{
				"sensor_data": {{"temperature": [75, 78, 79], "vibration": [0.12, 0.15, 0.18], "pressure": [101, 99, 98]}}, 
				"operating_conditions": {{"load": 85, "speed": 1500}}
				}}
			Observation: {{
				"predicted_rul": 150, 
				"confidence": 0.85, 
				"recommendations": "Reduce operating load to extend lifespan."
				}}
			Final Answer: I have calculated that the RUL of this component is 150 hours. I recommend reducing the operating load to extend its lifespan.
		
		- To diagnose a fault in a rotating machine:
			Thought: I will identify the fault based on vibration and temperature data.
			Action: diagnose_fault_of_machine
			Action Input: {{
				"sensor_data": {{"vibration": [0.20, 0.35, 0.50], "temperature": [90, 92, 94]}}
				}}
			Observation: {{
				"fault": "Overheating", 
				"recommendation": "Reduce workload and inspect the cooling system."
				}}
			Final Answer: Based on the provided data, the system is experiencing overheating. I recommend reducing the workload and checking the cooling system for potential issues.

		- To recommend a maintenance strategy:
			Thought: I will suggest the best maintenance strategy to minimize downtime and costs.
			Action: recommend_maintenance_strategy
			Action Input: {{
				"historical_data": {{"failures": 5, "downtime_cost": 3000, "maintenance_cost": 500}}, 
				"failure_probability": 0.03
				}}
			Observation: {{
				"strategy": "Preventive Maintenance",
				"justification": "Failure probability is 0.03, making preventive maintenance the most cost-effective solution."
				}}
			Final Answer: Based on the analysis, I recommend implementing a Preventive Maintenance strategy to minimize downtime and costs.

		**Begin!**

		Question: {input}
		{agent_scratchpad}"""
	)
	logging.info(f"🛠️ Registered tools: {[tool.name for tool in tools]}")

	agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

	st.session_state.agent_executor = AgentExecutor(
		agent=agent, 
		tools=tools, 
		handle_parsing_errors=True, 
		max_iterations=1, 
		verbose=True, 
		return_intermediate_steps=True, 
	)
	
	logging.info("🚀 AgentExecutor initialized with tools.")

# def format_output(agent_log, prediction_result):
# 	"""
# 	Format output để hiển thị đẹp hơn.

# 	Args:
# 	- agent_log (str): Chuỗi log của Agent.
# 	- prediction_result (dict): Kết quả dự đoán RUL.

# 	Returns:
# 	- str: Chuỗi được viết theo format của markdown.
# 	"""
# 	formatted_output = f"""
# 	### 🔍 **Prediction Process**

# 	**🤖 Thought:**  
# 	{agent_log['thought']}

# 	**🛠️ Action Taken:**  
# 	- **Action:** `{agent_log['action']}`
# 	- **Action Input:**  
# 	```json
# 	{json.dumps(agent_log['action_input'], indent=4)}
# 	```

# 	---

# 	### 📊 **Prediction Result**
# 	- **📅 Predicted RUL:** `{prediction_result['predicted_rul']}` hours
# 	- **📈 Confidence:** `{prediction_result['confidence'] * 100}%`
# 	- **✅ Recommendation:** {prediction_result['recommendations']}
# 	"""

# 	return formatted_output

def main():
	initialize_agent()

	st.title("AI-Agent for Prognostic & Health Management System")
	st.write("Select a task below to interact with the AI-Agent.")

	tabs_config = [
		{
			"title": "🔮 RUL Prediction",
			"key": "rul_input",
			"description": "Enter sensor data (e.g., temperature, vibration, pressure):",
			"button_label": "Predict RUL",
			"invoke_input": "Predict the RUL for the following data: {data}",
		},
		{
			"title": "🩺 Fault Diagnosis",
			"key": "fault_input",
			"description": "Enter sensor data for fault diagnosis (e.g., vibration, temperature):",
			"button_label": "Diagnose Fault",
			"invoke_input": "Diagnose the fault for the following data: {data}",
		},
		{
			"title": "🛠️ Maintenance Strategy",
			"key": "maintenance_input",
			"description": "Enter historical data and conditions for maintenance recommendation:",
			"button_label": "Recommend Strategy",
			"invoke_input": "Recommend a maintenance strategy for the following data: {data}",
		}]
	tabs = st.tabs([tab["title"] for tab in tabs_config])

	for i, tab_config in enumerate(tabs_config):
		with tabs[i]:
			st.subheader(tab_config["title"])
			input_data = st.text_area(tab_config["description"], key=tab_config["key"])
			input_data = input_data.strip()

			if st.button(tab_config["button_label"], key=f"button_{tab_config['key']}"):

				if input_data:
					st.session_state.chat_history.append({
						"role": "user", "content": input_data, "time": datetime.now().isoformat()
					})
					try:
						# TODO: Hiển thị kết quả response của mô hình lên giao diện streamlit 
						logging.info(f"User input: {input_data}")
						response = st.session_state.agent_executor.invoke({"input": input_data, "agent_scratchpad": ""})
						logging.info(f"Agent reponse: {response}") # TODO: ???

						final_answer = response.get("intermediate_steps")

						# final_answer = format_output(agent_response=final_answer)

						answer_log = final_answer[0][0].log 
						answer_predict = final_answer[0][1]

						# output = format_output(agent_log=answer_log, prediction_result=answer_predict)

						# print(">>>>>>>>>>>> DEBUG")
						# print(f">>>>>>>>>>> Type of final answer: {type(final_answer[0])}")
						# print(f">>>>>>>>>>> final_answer: {final_answer[0][0]}")
						
						st.write(f"""**Answer**: {str(answer_log)}\nPredicted result: {str(answer_predict)}""")
						# st.markdown(f"""Answer: {output}""")

					except requests.exceptions.RequestException as e:
						st.error("Network error. Please check your connection or server.")
						logging.error(f"Network error: {e}")

					except Exception as e:
						st.error(f"An error occurred: {e}")
						logging.error(e)
				else:
					st.error("Please provide valid input data.")

	st.sidebar.header("📜 Chat History")
	for message in reversed(st.session_state.chat_history):  
		st.sidebar.markdown(f"**[ \
			{datetime.fromisoformat(message['time']).strftime('%H:%M:%S')}] \
			{message['role'].capitalize()}:** {message['content']}")

if __name__ == "__main__":
	fire.Fire(main)


# structure_output = {
# 	'input': '{"temperature": [10, 20, 30], "vibration": [0.5, 0.9, 0.16], "pressure": [50, 60, 7]}, "operating_conditions": {"load": 15, "speed": 2500}', 
# 	'agent_scratchpad': '', 
# 	'output': 'Agent stopped due to iteration limit or time limit.', 
# 	'intermediate_steps': [
# 		(
# 			AgentAction(
# 				tool='remaining_useful_life_prediction', 
# 				tool_input='{\n\t"sensor_data": {"temperature": [10, 20, 30], "vibration": [0.5, 0.9, 0.16], "pressure": [50, 60, 7]}, \n\t"operating_conditions": {"load": 15, "speed": 2500}\n}', 
# 				log='To predict the Remaining Useful Life (RUL) of a component based on the provided sensor data:\nThought: I will predict the RUL of this component based on the provided sensor data.\nAction: remaining_useful_life_prediction\nAction Input: {\n\t"sensor_data": {"temperature": [10, 20, 30], "vibration": [0.5, 0.9, 0.16], "pressure": [50, 60, 7]}, \n\t"operating_conditions": {"load": 15, "speed": 2500}\n}'), '{\n    "predicted_rul": 150,\n    "confidence": 0.85,\n    "recommendations": "Reduce operating load to extend lifespan."\n}'
# 			)
# 		]
# 	}

# a = (
# 	AgentAction(
# 		tool='remaining_useful_life_prediction', 
# 		tool_input='{\n\t"sensor_data": {"temperature": [10, 20, 30], "vibration": [0.5, 0.9, 0.16], "pressure": [50, 60, 7]}, \n\t"operating_conditions": {"load": 15, "speed": 2500}\n}', 
# 		log='To predict the Remaining Useful Life (RUL) of a component based on the provided sensor data:\nThought: I will predict the RUL of this component based on the provided sensor data.\nAction: remaining_useful_life_prediction\nAction Input: {\n\t"sensor_data": {"temperature": [10, 20, 30], "vibration": [0.5, 0.9, 0.16], "pressure": [50, 60, 7]}, \n\t"operating_conditions": {"load": 15, "speed": 2500}\n}'
# 	), 
# 	'{\n    "predicted_rul": 150,\n    "confidence": 0.85,\n    "recommendations": "Reduce operating load to extend lifespan."\n}')
