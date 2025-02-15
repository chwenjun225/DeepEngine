import streamlit as st
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Streamlit Page Configuration
st.set_page_config(layout="wide", page_title="AI-Agent Demo")

# Sidebar: Upload Video File
st.sidebar.title("📂 Upload Video")
video_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# Function to generate vibration data
start_time = time.time()
sensor_data = []
timestamps = []

def generate_vibration_data():
	elapsed_time = time.time() - start_time  # Time since app started

	if elapsed_time < 20:
		return np.random.uniform(0.2, 1.0) + np.sin(time.time())  # Normal vibration
	else:
		return 0.05 ** 2 + np.random.uniform(0.1, 0.2)  # Parabolic increase

# Chatbot messages history
if "chat_history" not in st.session_state:
	st.session_state.chat_history = []

# Layout: Three Columns (Equal Width)
col1, col2 = st.columns([2.5, 2.5])  # Adjusted to fit better on the screen

with col1:
	st.header("Camera 360")
	
	if video_file:
		video_bytes = video_file.read()
		video_path = "temp_video.mp4"
		with open(video_path, "wb") as f:
			f.write(video_bytes)
		st.video(video_path)

	# Sensor Vibration Graph (Real-time)
	st.header("CNC Machine Vibration Tracking")
	graph_placeholder = st.empty()  # Placeholder for real-time graph

	for _ in range(100):  # Simulate real-time updates
		new_vibration = generate_vibration_data()
		sensor_data.append(new_vibration)
		timestamps.append(time.time() - start_time)

		# Keep history but limit display range for better visualization
		if len(sensor_data) > 200:  # Limit display to last 200 points
			sensor_data.pop(0)
			timestamps.pop(0)

		# Create figure
		fig, ax = plt.subplots(figsize=(6, 3))  # Smaller figure for better fit
		ax.plot(timestamps, sensor_data, color="red", linestyle="-", marker="o", markersize=3)
		ax.set_xlabel("Time (s)")
		ax.set_ylabel("Vibration Level")
		ax.set_title("Real-time CNC Machine Vibration Data")
		ax.grid(True)

		# Update graph in Streamlit
		graph_placeholder.pyplot(fig)

		time.sleep(0.2)  # Simulate sensor reading every 200ms

with col2:
	st.header("💬 AI Chatbot")
	user_input = st.text_input("AI-Agent PHM", "")

	if st.button("Send"):
		if user_input:
			# Simulating chatbot response (Replace with actual AI model call)
			response = "**Danger!**\n\nGet far away of the machine\n\nAnomaly detected in the CNC machine vibration data. Please check the machine immediately."
			st.session_state.chat_history.append(("🧑 User:", user_input))
			st.session_state.chat_history.append(("🤖 AI:", response))

	# Display Chat History
	chat_area = st.container()
	with chat_area:
		for role, text in st.session_state.chat_history[-5:]:  # Show only last 5 messages
			st.markdown(f"**{role}** {text}")

























# import streamlit as st
# import cv2
# import os
# import numpy as np
# import time
# import matplotlib.pyplot as plt

# # Streamlit Page Configuration
# st.set_page_config(layout="wide", page_title="AI-Agent Demo")

# # Sidebar: Upload Video File
# st.sidebar.title("Upload Video")
# video_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# # Function to generate vibration data
# start_time = time.time()

# def generate_vibration_data():
#     elapsed_time = time.time() - start_time  # Time since app started

#     if elapsed_time < 30:
#         # Normal vibration before 30 seconds
#         return np.random.uniform(0.2, 1.0) + np.sin(time.time())
#     else:
#         # Parabolic increase after 30 seconds (simulating an anomaly)
#         return 0.05 * (elapsed_time - 30) ** 2 + np.random.uniform(0.1, 0.2)

# # Chatbot messages history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Layout: Columns
# col1, col2 = st.columns([3, 2])  # Left (Video + Graph) | Right (Chatbot)

# with col1:
#     st.header("Camera Reatime")
#     if video_file:
#         video_bytes = video_file.read()
#         video_path = "temp_video.mp4"
#         with open(video_path, "wb") as f:
#             f.write(video_bytes)
#         st.video(video_path)

#     # Sensor Vibration Graph (Real-time)
#     st.header("PHMTracking Vibration Sensor Data")
#     sensor_data = []
#     timestamps = []
#     graph_placeholder = st.empty()  # Placeholder for real-time graph

#     for _ in range(100):  # Simulate 100 real-time updates
#         sensor_data.append(generate_vibration_data())
#         timestamps.append(time.time() - start_time)

#         if len(sensor_data) > 50:  # Keep only last 50 data points
#             sensor_data.pop(0)
#             timestamps.pop(0)

#         # Create figure
#         fig, ax = plt.subplots()
#         ax.plot(timestamps, sensor_data, color="red", linestyle="-", marker="o")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Vibration Level")
#         ax.set_title("Real-time Machine Vibration Data")
#         ax.grid(True)

#         # Update graph in Streamlit
#         graph_placeholder.pyplot(fig)

#         time.sleep(0.2)  # Simulate sensor reading every 200ms

# with col2:
#     st.header("💬 AI Chatbot")
#     user_input = st.text_input("AI-Agent PHM", "")

#     if st.button("Send"):
#         if user_input:
#             # Simulating chatbot response (Replace with AI model call)
#             response = f"🤖 AI: I'm processing '{user_input}'... (This is a placeholder response)"
#             st.session_state.chat_history.append(("🧑 User:", user_input))
#             st.session_state.chat_history.append(("🤖 AI:", response))

#     # Display Chat History
#     for role, text in st.session_state.chat_history:
#         st.markdown(f"**{role}** {text}")




# import streamlit as st
# import cv2
# import os 
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# import requests 
# import json 

# # Streamlit Page Configuration
# st.set_page_config(layout="wide", page_title="AI-Agent Demo")

# # Sidebar: Upload Video File
# st.sidebar.title("Upload Video")
# video_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# # Function to generate fake real-time vibration sensor data
# def generate_vibration_data():
#     return np.random.uniform(0.2, 1.0) + np.sin(time.time())

# # Chatbot messages history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Layout: Columns
# col1, col2 = st.columns([3, 2])  # Left (Video + Graph) | Right (Chatbot)

# with col1:
#     st.header("📹 Video Display")
#     if video_file:
#         # Load Video with OpenCV
#         video_bytes = video_file.read()
#         video_path = "temp_video.mp4"
#         with open(video_path, "wb") as f:
#             f.write(video_bytes)
#         st.video(video_path)

#     # Sensor Vibration Graph (Real-time)
#     st.header("PHM Real-time Vibration Sensor Data")
#     sensor_data = []
#     timestamps = []

#     graph_placeholder = st.empty()  # Placeholder for real-time graph

#     for _ in range(100):  # Simulate 100 real-time updates
#         sensor_data.append(generate_vibration_data())
#         timestamps.append(time.time())

#         # Keep only the last 50 data points for smooth visualization
#         if len(sensor_data) > 50:
#             sensor_data.pop(0)
#             timestamps.pop(0)

#         # Create figure
#         fig, ax = plt.subplots()
#         ax.plot(timestamps, sensor_data, color="red", linestyle="-", marker="o")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Vibration Level")
#         ax.set_title("Real-time Machine Vibration Data")
#         ax.grid(True)

#         # Update graph in Streamlit
#         graph_placeholder.pyplot(fig)

#         time.sleep(0.2)  # Simulate sensor reading every 200ms

# with col2:
#     st.header("💬 AI Chatbot")
#     user_input = st.text_input("Ask something...", "")

#     if st.button("Send"):
#         if user_input:
#             # Simulating chatbot response (Replace with AI model call)
#             response = f"🤖 AI: I'm processing '{user_input}'... (This is a placeholder response)"
#             st.session_state.chat_history.append(("🧑 User:", user_input))
#             st.session_state.chat_history.append(("🤖 AI:", response))

#     # Display Chat History
#     for role, text in st.session_state.chat_history:
#         st.markdown(f"**{role}** {text}")

# # Run Streamlit with:
# # streamlit run filename.py
