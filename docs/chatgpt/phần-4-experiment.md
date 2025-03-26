```python 
import matplotlib.pyplot as plt
import pandas as pd

# Sample performance data
data = {
    "Agent": ["Visual Agent", "Research Agent", "Reasoning Agent"],
    "Response Time (ms)": [35, 120, 210],
    "Accuracy (%)": [91, None, 98],
}

df = pd.DataFrame(data)

# Create bar charts for response time and accuracy
fig, ax1 = plt.subplots(figsize=(8, 5))

# Bar chart for response time
ax1.bar(df["Agent"], df["Response Time (ms)"], color="#4A90E2")
ax1.set_ylabel("Response Time (ms)", color="#4A90E2")
ax1.set_title("Agent Performance Overview")
ax1.tick_params(axis='y', labelcolor="#4A90E2")

# Secondary axis for accuracy
ax2 = ax1.twinx()
ax2.plot(df["Agent"], df["Accuracy (%)"], color="#E94E77", marker='o', linewidth=2, label="Accuracy")
ax2.set_ylabel("Accuracy (%)", color="#E94E77")
ax2.tick_params(axis='y', labelcolor="#E94E77")

plt.tight_layout()
plt.grid(True)
plt.show()
```

Here’s a chart showing **Agent Performance** with response time (bars) and accuracy (line). You can use this directly in your slide to clearly explain how each agent contributes in both speed and reliability. Let me know if you'd like the raw image to insert!