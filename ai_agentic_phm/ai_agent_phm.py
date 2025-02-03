from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

app = Flask(__name__)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="../models/DeepSeek-R1-Distill-Qwen-1.5B")

@app.route('/generate', methods=['POST'])
def generate():
	data = request.get_json()
	prompts = data.get('prompts', [])

	outputs = llm.generate(prompts, sampling_params)

	# Prepare the outputs.
	results = []

	for output in outputs:
		prompt = output.prompt
		generated_text = output.outputs[0].text
		results.append({
			'prompt': prompt,
			'generated_text': generated_text
		})

	return jsonify(results)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=2025, debug=True)