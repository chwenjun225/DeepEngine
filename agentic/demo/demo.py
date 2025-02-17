import streamlit as st 
import cv2 
import time
import numpy as np 
import matplotlib.pyplot as plt 

st.set_page_config(layout="wide", page_title="AI-Agent PHM")

st.sidebar.title("Upload Video")
st.sidebar.title("Realtime Camera")
video_file =  st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

start_time = time.time()
sensor_data = []
timestamps = []

def generate_vibration_data():
	elapsed_time = time.time() - start_time 

	if elapsed_time < 20:
		return np.random.uniform(10, 20) + np.sin(time.time())
	else:
		return 0.05 ** 2 + np.random.uniform(80, 100)

if "chat_history" not in st.session_state:
	st.session_state.chat_history = []

col1, col2 = st.columns([2.5, 2.5])

with col1:
	st.header("AI-Agent Prognostic and Health Management")
	if video_file:
		video_bytes = video_file.read()
		video_path = "temp_video.mp4"
		with open(video_path, "wb") as f:
			f.write(video_bytes)
			st.video(video_path)
			st.header("CNC Machine Vibration Tracking")
			graph_placeholder = st.empty()

			with col2:
				st.header("Chatbot")
				user_input = st.text_input("Enter prompt:")

				if st.button("Send"):
					if user_input:
						response = "**Danger!**\n\nGet far away of the CNC machine\n\nAnomaly detected in the CNC machine vibration data. Please check the machine immediately."
						st.session_state.chat_history.append(("🧑 User:", user_input))
						st.session_state.chat_history.append(("🤖 AI:", response))
						
						for _ in range(100):
							new_vibration = generate_vibration_data()
							sensor_data.append(new_vibration)
							timestamps.append(time.time() - start_time)

							if len(sensor_data) > 300:
								sensor_data.pop(0)
								timestamps.pop(0)
							
							fig, ax = plt.subplots(figsize=(6, 3)) 
							ax.plot(timestamps, sensor_data, color="red", linestyle="-", marker="o", markersize=1)
							ax.set_ylim([0, 100])
							ax.set_xlabel("Time (s)")
							ax.set_ylabel("Vibration Level")
							ax.set_title("Real-time Prediction CNC Machine Vibration Data")
							ax.grid(True)

							graph_placeholder.pyplot(fig)
							time.sleep(0.1)

					chat_area = st.container()
					with chat_area:
						for role, text in st.session_state.chat_history[-5:]:
							st.markdown(f"**{role}** {text}")

# Report to worker, If you recognize any accident from vibration data