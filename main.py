import os 
import fire 
import logging 
import urllib3

import streamlit as st 

from init_agent import agent

# Configure logging 
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global varibales for lazy initialization 
llm = None
agent_executor = None 

def main():
	st.title("AI-Agent for Prognostic & Health Management System")
	st.write("Select a task below to interact with the AI-Agent.")

	# Danh sách cấu hình cho các tab
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
		},
	]
	# Tạo các tab động
	tabs = st.tabs([tab["title"] for tab in tabs_config])
	
	# Lặp qua từng tab
	for i, tab_config in enumerate(tabs_config):
		with tabs[i]:
			st.subheader(tab_config["title"])
			input_data = st.text_area(tab_config["description"], key=tab_config["key"])

			if st.button(tab_config["button_label"], key=f"button_{tab_config['key']}"):
				if input_data.strip():
					# Lưu tin nhắn người dùng với thời gian
					st.session_state.chat_history.append({
						"role": "user",
						"content": user_input,
						"time": datetime.now().isoformat()  # Lưu thời gian
					})
					try:
						# Gọi agent để xử lý
						result = agent_executor.invoke({
							"input": tab_config["invoke_input"].format(data=input_data),
							"agent_scratchpad": ""
						})
						st.success("Result:")
						st.write(result.get('output', 'No answer provided.'))
					except requests.exceptions.RequestException as e:
						st.error("Network error. Please check your connection or server.")
						logging.error(f"Network error: {str(e)}")
					except Exception as e:
						st.error(f"An error occurred: {str(e)}")
				else:
					st.error("Please provide valid input data.")

	# Hiển thị lịch sử hội thoại
	if "chat_history" not in st.session_state: 
		st.session_state.chat_history = []

	# Hiển thị tiêu đề Sidebar
	st.sidebar.header("📜 Chat History", divider=True)

	# Tạo dictionary để nhóm tin nhắn theo ngày
	chat_by_date = {}

	# Duyệt qua các tin nhắn và nhóm theo ngày
	for message in st.session_state.chat_history:

		# Chuyển đổi thời gian thành chuỗi ngày
		message_time = datetime.fromisoformat(message["time"])
		date_key = message_time.strftime("%Y-%m-%d")  # YYYY-MM-DD

		# Thêm tin nhắn vào nhóm của ngày tương ứng
		if date_key not in chat_by_date:
			chat_by_date[date_key] = []
		chat_by_date[date_key].append(message)

	# Hiển thị từng ngày
	for date, messages in chat_by_date.items():
		st.sidebar.subheader(f"📅 {date}")  # Hiển thị ngày

		# Hiển thị tin nhắn trong ngày
		for message in messages:
			message_time = datetime.fromisoformat(message["time"]).strftime("%H:%M:%S")  # HH:mm:ss
			if message["role"] == "user":
				st.sidebar.markdown(f"**[{message_time}] You:** {message['content']}")
			elif message["role"] == "assistant":
				st.sidebar.markdown(f"**[{message_time}] AI-Agent:** {message['content']}")

if __name__ == "__main__":
	fire.Fire(main)
