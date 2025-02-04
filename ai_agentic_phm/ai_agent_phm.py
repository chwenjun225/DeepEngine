# Smaller models inherit larger models' reasoning patterns, showcasing the distillation process's effectiveness.
from openai import OpenAI

path_model = "/home/chwenjun225/Projects/Foxer/notebooks/DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct/1_finetuned_DeepSeek-R1-Distill-Qwen-1.5B_finetune_CoT_ReAct"

client = OpenAI(
	base_url="http://localhost:2025/v1",
	api_key="token-abc123",
)

completion = client.chat.completions.create(
	model=path_model,
	messages=[
		{"role": "user", "content": "Hello!"}, 
        {"role": "user", "content": "Explain what is AI-Agent?"}
	]
)

print(completion.choices[0].message)
