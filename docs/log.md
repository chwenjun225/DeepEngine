# 2025-01-20:20:51:30,601 
hf (pretrained=/home/chwenjun225/.llama/checkpoints/Llama3.1-8B-Instruct/hf), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto (64)
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|arc_challenge |      1|none  |     0|acc       |↑  |0.5188|±  |0.0146|
|              |       |none  |     0|acc_norm  |↑  |0.5520|±  |0.0145|
|arc_easy      |      1|none  |     0|acc       |↑  |0.8199|±  |0.0079|
|              |       |none  |     0|acc_norm  |↑  |0.7963|±  |0.0083|
|hellaswag     |      1|none  |     0|acc       |↑  |0.5901|±  |0.0049|
|              |       |none  |     0|acc_norm  |↑  |0.7923|±  |0.0040|
|lambada_openai|      1|none  |     0|acc       |↑  |0.7320|±  |0.0062|
|              |       |none  |     0|perplexity|↓  |3.4015|±  |0.0731|
|openbookqa    |      1|none  |     0|acc       |↑  |0.3380|±  |0.0212|
|              |       |none  |     0|acc_norm  |↑  |0.4300|±  |0.0222|
|piqa          |      1|none  |     0|acc       |↑  |0.7998|±  |0.0093|
|              |       |none  |     0|acc_norm  |↑  |0.8079|±  |0.0092|
|winogrande    |      1|none  |     0|acc       |↑  |0.7403|±  |0.0123|
