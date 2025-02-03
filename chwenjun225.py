def formatting_prompts_func(examples, train_prompt_style, eos_token):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + eos_token
        texts.append(text)
    return {
        "text": texts,
    }