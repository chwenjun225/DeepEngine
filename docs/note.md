# https://learning.oreilly.com/library/view/llm-engineers-handbook/9781836200079/Text/Chapter_01.xhtml#_idParaDest-34

## The problem with building ML systems
Building production-ready ML systems is much more than just training a model. From an engineering point of view, training the model is the most straightforward step in most use cases. However, training a model becomes complex when deciding on the correct architecture and hyperparameters. That’s not an engineering problem but a research problem.

At this point, we want to focus on how to design a production-ready architecture. Training a model with high accuracy is extremely valuable, but just by training it on a static dataset, you are far from deploying it robustly. We have to consider how to do the following:

Ingest, clean, and validate fresh data
Training versus inference setups
Compute and serve features in the right environment
Serve the model in a cost-effective way
Version, track, and share the datasets and models
Monitor your infrastructure and models
Deploy the model on a scalable infrastructure
Automate the deployments and training

## Listing the technical details of the LLM Twin architecture
Until now, we defined what the LLM Twin should support from the user’s point of view. Now, let’s clarify the requirements of the ML system from a purely technical perspective:

On the data side, we have to do the following:
Collect data from LinkedIn, Medium, Substack, and GitHub completely autonomously and on a schedule
Standardize the crawled data and store it in a data warehouse
Clean the raw data
Create instruct datasets for fine-tuning an LLM
Chunk and embed the cleaned data. Store the vectorized data into a vector DB for RAG.
For training, we have to do the following:
Fine-tune LLMs of various sizes (7B, 14B, 30B, or 70B parameters)
Fine-tune on instruction datasets of multiple sizes
Switch between LLM types (for example, between Mistral, Llama, and GPT)
Track and compare experiments
Test potential production LLM candidates before deploying them
Automatically start the training when new instruction datasets are available.
The inference code will have the following properties:
A REST API interface for clients to interact with the LLM Twin
Access to the vector DB in real time for RAG
Inference with LLMs of various sizes
Autoscaling based on user requests
Automatically deploy the LLMs that pass the evaluation step.
The system will support the following LLMOps features:
Instruction dataset versioning, lineage, and reusability
Model versioning, lineage, and reusability
Experiment tracking
Continuous training, continuous integration, and continuous delivery (CT/CI/CD)
Prompt and system monitoring

## Nên sử dụng `poetry` thay vì `conda`
Other tools similar to `Poetry` are `Venv` and `Conda` for creating virtual environments. Still, they lack the dependency management option. Thus, you must do it through Python’s default `requirements.txt` files, which are less powerful than `Poetry’s lock` files. Another option is `Pipenv`, which feature-wise is more like `Poetry` but slower, and `uv`, which is a replacement for `Poetry` built in `Rust`, making it blazing fast. `uv` has lots of potential to replace Poetry, making it worthwhile to test out: https://github.com/astral-sh/uv.

# TODO: https://learning.oreilly.com/library/view/llm-engineers-handbook/9781836200079/Text/Chapter_02.xhtml#_idParaDest-44