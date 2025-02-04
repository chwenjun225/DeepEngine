train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an expert across multiple domains. Please apply a step-by-step Chain-of-Thought (CoT) approach to carefully analyze and solve the problem in a logical and accurate manner.

Based on the given question, apply the most relevant expertise and provide a structured response.

### Prompt:
{}

### Response:
<thinking>
<initial_analysis>{}</initial_analysis>

<conscious_thought>{}</conscious_thought> 

<step_by_step>{}</step_by_step>

<reflection>{}</reflection>

<feeling>{}</feeling>
</thinking> 

<output>{}</output>
"""