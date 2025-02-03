from openai import OpenAI
client = OpenAI(
	base_url="http://localhost:2025/v1",
	api_key="token-abc123",
)

completion = client.chat.completions.create(
	model="/home/chwenjun225/Projects/Foxer/models/DeepSeek-R1-Distill-Qwen-1.5B/",
	messages=[
		{"role": "user", "content": "Hello!"}, 
        {"role": "user", "content": "Explain what is AI-Agent?"}
	]
)

print(completion.choices[0].message)
