import requests

url = "http://localhost:8080/completion"
data = {
    "prompt": "Can I build AI-Agentic with DeepSeek-R1?",
    "n_predict": 200
}

response = requests.post(url, json=data)
print(response.json())
