import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

# Streamlit Page Configuration
st.set_page_config(layout="wide", page_title="AI-Agent Demo")

# Sidebar: Upload Video File
st.sidebar.title("📂 Upload Video")
video_file = st.sidebar.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# Initialize Global Variables
start_time = time.time()
sensor_data = []
timestamps = []

# Function to generate vibration data
def generate_vibration_data():
    elapsed_time = time.time() - start_time  # Time since app started

    if elapsed_time <= 20:
        return np.random.uniform(10, 20)  # Normal vibration stays between 10-20
    else:
        return np.random.uniform(40, 90)  # Abnormal vibration spikes to 40-90

# Chatbot messages history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Layout: Two Columns
col1, col2 = st.columns([2.5, 2.5])

with col1:
    st.header("📷 Camera 360")

    if video_file:
        video_bytes = video_file.read()
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        st.video(video_path)

    # Real-time CNC Machine Vibration Graph
    st.header("📊 CNC Machine Vibration Tracking")
    graph_placeholder = st.empty()

    for _ in range(200):  # Simulate real-time updates
        new_vibration = generate_vibration_data()
        current_time = time.time() - start_time

        # Append data in order
        sensor_data.append(new_vibration)
        timestamps.append(current_time)

        # Keep only the last 50 points for smooth visualization
        if len(sensor_data) > 50:
            sensor_data = sensor_data[-50:]
            timestamps = timestamps[-50:]

        # Ensure timestamps remain increasing
        sorted_indices = np.argsort(timestamps)
        sorted_timestamps = np.array(timestamps)[sorted_indices]
        sorted_sensor_data = np.array(sensor_data)[sorted_indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(sorted_timestamps, sorted_sensor_data, color="red", linestyle="-", marker="o", markersize=3)
        ax.set_xlim([max(0, current_time - 50), current_time + 2])  # Keep x-axis moving forward
        ax.set_ylim([0, 100])  # Keep consistent y-axis
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
            response = "**🚨 Danger!**\n\n⚠️ Get far away from the machine!\n\nAnomaly detected in CNC machine vibration data. Please check the machine immediately."
            st.session_state.chat_history.append(("🧑 User:", user_input))
            st.session_state.chat_history.append(("🤖 AI:", response))

    # Display Chat History
    chat_area = st.container()
    with chat_area:
        for role, text in st.session_state.chat_history[-5:]:  # Show only last 5 messages
            st.markdown(f"**{role}** {text}")
