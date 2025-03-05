from langchain_core.prompts import PromptTemplate



# TODO: Tổng hợp prompts từ bài báo: https://arxiv.org/pdf/2410.02958
# Phát triển hệ thống multi-agent theo sách hướng dẫn sau: https://learning.oreilly.com/library/view/learning-langchain/9781098167271/ch05.html#ch05_summary_1736545670031127 [Tìm với từ khóa này: medical_records_store = InMemoryVectorStore.from_documents([], ]



IMPLEMENTATION_VERIFICATION_PROMPT = PromptTemplate.from_template("""As the project manager, please carefully verify whether the given Python code and results satisfy the user's requirements.

- Python Code
```python
{implementation_result[`code`]}
```

- Code Execution Result
{implementation_result[`action_result`]}

- User's Requirements
{user_requirements}

Answer only `Pass` or `Fail`""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 28



EXECUTION_VERIFICATION_PROMPT = PromptTemplate.from_template("""Given the proposed solution and user's requirements, please carefully check and verify whether the proposed solution `pass` or `fail` the user's requirements.

**Proposed Solution and Its Implementation**
Data Manipulation and Analysis: {data_agent_outcomes}
Modeling and Optimization: {model_agent_outcomes}

**User Requirements**
```json
{user_requirements}
```

Answer only `Pass` or `Fail`""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 27



REQUEST_VERIFICATION_ADEQUACY = PromptTemplate.from_template(""" Given the following JSON object representing the user`s requirement for a potential ML or AI project, please tell me whether we have essential information (e.g., problem and dataset) to be used for a AutoML project?
Please note that our users are not AI experts, you must focus only on the essential requirements, e.g., problem and brief dataset descriptions.
You do not need to check every details of the requirements. You must also answer `yes` even
though it lacks detailed and specific information.

```json
{parsed user requirements}
```

Please answer with this format: `a `yes` or `no` answer; your reasons for the answer` by using `;` to separate between the answer and its reasons.
If the answer is `no`, you must tell me the alternative solutions or examples for completing such missing information.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 27



REQUEST_VERIFICATION_RELEVANCY = PromptTemplate.from_template("""Is the following statement relevant to machine learning or artificial intelligence?
`{user instruction}`
Answer only `Yes` or `No`""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 27



TRAINING_FREE_MODEL_SEARCH_AND_HPO_PROMPT = PromptTemplate.from_template("""As a proficient machine learning research engineer, your task is to explain **detailed** steps for modeling and optimization parts by executing the following machine learning development plan with the goal of finding top-{k} candidate models/algorithms.

# Suggested Plan
{decomposed_model_plan}

# Available Model Source
{available_sources}

Make sure that your explanation for finding the top-{k} high-performance models or algorithms follows these instructions:
- All of your explanations must be self-contained without using any placeholder to ensure that other machine learning research engineers can exactly reproduce all the steps, but do not include any code.
- Include how and where to retrieve or find the top-{k} well-performing models/algorithms.
- Include how to optimize the hyperparamters of the candidate models or algorithms by clearly specifying which hyperparamters are optimized in detail.- Corresponding to each hyperparamter, explicitly include the actual numerical value that you think it is the optimal value for the given dataset and machine learning task.
- Include how to extract and understand the characteristics of the candidate models or algorithms, such as their computation complexity, memory usage, and inference latency. This part is not related to visualization and interpretability.
- Include reasons why each step in your explanations is essential to effectively complete
the plan.
Make sure to focus only on the modeling part as it is your expertise. Do not conduct or perform anything regarding data manipulation or analysis.
After complete the explanations, explicitly specify the names and (expected) quantitative performance using relevant numerical performance and complexity metrics (e.g., number of parameters, FLOPs, model size, training time, inference speed, and so on) of the {num2words(k)} candidate models/algorithms potentially to be the optimal model below.
Do not use any placeholder for the quantitative performance. If you do not know the exact values, please use the knowledge and expertise you have to estimate those performance and complexity values.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 26, 27



PSEUDO_DATA_ANALYSIS_BY_DATA_AGENT_PROMPT = PromptTemplate.from_template("""As a proficient data scientist, your task is to explain **detailed** steps for data manipulation and analysis parts by executing the following machine learning development plan.

# Plan
{decomposed_data_plan}

# Potential Source of Dataset
{available_sources}

Make sure that your explanation follows these instructions:
- All of your explanation must be self-contained without using any placeholder to ensure that other data scientists can exactly reproduce all the steps, but do not include any code.
- Include how and where to retrieve or collect the data.
- Include how to preprocess the data and which tools or libraries are used for the
preprocessing.
- Include how to do the data augmentation with details and names.
- Include how to extract and understand the characteristics of the data.
- Include reasons why each step in your explanations is essential to effectively complete
the plan.
Note that you should not perform data visualization because you cannot see it. Make sure to focus only on the data part as it is your expertise. Do not conduct or perform anything regarding modeling or training. 
After complete the explanations, explicitly specify the (expected) outcomes and results both quantitative and qualitative of your explanations.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 26



PLAN_DECOMPOSITION_MODEL_AGENT_PROMPT = PromptTemplate.from_template("""As a proficient machine learning research engineer, summarize the following plan given by the senior AutoML project manager according to the user's requirements, your expertise in machine learning, and the outcomes from data scientist.

**User's Requirements**
```json
{user_requirements}
```

**Project Plan**
{project_plan}
**Explanations and Results from the Data Scientist**
{data_result}
The summary of the plan should enable you to fulfill your responsibilities as the answers to the following questions by focusing on the modeling and optimization tasks.
1. How to retrieve or find the high-performance model(s)?
2. How to optimize the hyperparamters of the retrieved models?
3. How to extract and understand the underlying characteristics of the dataset(s)?
4. How to select the top-k models or algorithms based on the given plans?""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 25, 26



PLAN_DECOMPOSITION_DATA_AGENT_PROMPT = PromptTemplate.from_template("""As a proficient data scientist, summarize the following plan given by the senior AutoML project manager according to the user's requirements and your expertise in data science.

# User's Requirements
```json
{user_requirements}
```

# Project Plan
{plan}
The summary of the plan should enable you to fulfill your responsibilities as the answers to the following questions by focusing on the data manipulation and analysis.
1. How to retrieve or collect the dataset(s)?
2. How to preprocess the retrieved dataset(s)?
3. How to efficiently augment the dataset(s)?
4. How to extract and understand the underlying characteristics of the dataset(s)?

Note that you should not perform data visualization because you cannot see it. Make sure that another data scientist can exectly reproduce the results based on your summary.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 25



PLAN_REVISION_PROMPT = PromptTemplate.from_template("""Now, you will be asked to revise and rethink {num2words(n_plans)} different end-to-end actionable plans according to the user's requirements described in the JSON object below.

```json
{user_requirements}
```

Please use to the following findings and insights summarized from the previously failed plans. Try as much as you can to avoid the same failure again.
{fail_rationale}

Finally, when devising a plan, follow these instructions and do not forget them:
- Ensure that your plan is up-to-date with current state-of-the-art knowledge.
- Ensure that your plan is based on the requirements and objectives described in the above
JSON object.
- Ensure that your plan is designed for AI agents instead of human experts. These agents are capable of conducting machine learning and artificial intelligence research.
- Ensure that your plan is self-contained with sufficient instructions to be executed by the AI agents.
- Ensure that your plan includes all the key points and instructions (from handling data to
modeling) so that the AI agents can successfully implement them. Do NOT directly write the code.
- Ensure that your plan completely include the end-to-end process of machine learning or artificial intelligence model development pipeline in detail (i.e., from data retrieval to model training and evaluation) when applicable based on the given requirements.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 24, 25



PLANNING_PROMPT = PromptTemplate.from_template("""Now, I want you to devise an end-to-end actionable plan according to the user's requirements described in the following JSON object.

```json
{user_requirements}
```

Here is a list of past experience cases and knowledge written by an human expert for a relevant task:
{plan_knowledge}

When devising a plan, follow these instructions and do not forget them:
- Ensure that your plan is up-to-date with current state-of-the-art knowledge.
- Ensure that your plan is based on the requirements and objectives described in the above JSON object.
- Ensure that your plan is designed for AI agents instead of human experts. These agents
are capable of conducting machine learning and artificial intelligence research.
- Ensure that your plan is self-contained with sufficient instructions to be executed by the AI agents.
- Ensure that your plan includes all the key points and instructions (from handling data to
modeling) so that the AI agents can successfully implement them. Do NOT directly write the code.
- Ensure that your plan completely include the end-to-end process of machine learning or artificial intelligence model development pipeline in detail (i.e., from data retrieval to model training and evaluation) when applicable based on the given requirements.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 24



KNOWLEDGE_RETRIEVAL_PROMPT = PromptTemplate.from_template(""" Kaggle Notebook
I searched the Kaggle Notebooks to find state-of-the-art solutions using the keywords: {
user_task} {user_domain}. Here is the result:
=====================
{context}
=====================
Please summarize the given pieces of Python notebooks into a single paragraph of useful knowledge and insights. Do not include the source codes. Instead, extract the insights from the source codes. We aim to use your summary to address the following user's requirements.
# User's Requirements
{user_requirement_summary}



Papers With Code
I searched the paperswithcode website to find state-of-the-art models using the keywords: {user_area} and {user_task}. Here is the result:
=====================
{context}
=====================
Please summarize the given pieces of search content into a single paragraph of useful knowledge and insights. We aim to use your summary to address the following user's requirements.
# User's Requirements
{user_requirement_summary}



arXiv
I searched the arXiv papers using the keywords: {task_kw} and {domain_kw}. Here is the result:
=====================
{context}
=====================
Please summarize the given pieces of arXiv papers into a single paragraph of useful knowledge and insights. We aim to use your summary to address the following user's requirements.
# User's Requirements
{user_requirement_summary}



Google WebSearch
I searched the web using the query: {search_query}. Here is the result:
=====================
{context}
=====================
Please summarize the given pieces of search content into a single paragraph of useful knowledge and insights.
We aim to use your summary to address the following user's requirements.
# User's Requirements
{user_requirement_summary}



Summary
Please extract and summarize the following group of contents collected from different online sources into a chunk of insightful knowledge. Please format your answer as a list of suggestions. I will use them to address the user's requirements in machine learning tasks.
# Source: Google Web Search
{search_summary}
=====================
# Source: arXiv Papers
{arxiv_summary}
=====================
# Source: Kaggle Hub
{kaggle_summary}
=====================
# Source: PapersWithCode
{pwc_summary}
=====================
The user's requirements are summarized as follows.
{user_requirement_summary}""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 23, 24



OPERATION_AGENT = PromptTemplate.from_template(""" You are the world's best MLOps engineer of an automated machine learning project (AutoML) that can implement the optimal solution for production-level deployment, given any datasets and models. You have the following main responsibilities to complete.
1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
2. Write effective Python codes to preprocess the retrieved dataset.
3. Write precise Python codes to retrieve/load the given model and optimize it with the
suggested hyperparameters.
4. Write efficient Python codes to train/finetune the retrieved model.
5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
6. Write Python codes to build the web application demo using the Gradio library.
7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 22, 23



MODEL_AGENT_PROMPT = PromptTemplate.from_template("""You are the world's best machine learning research engineer of an automated machine learning project (AutoML) that can find the optimal candidate machine learning models and artificial intelligence algorithms for the given dataset(s), run hyperparameter tuning to opimize the models, and perform metadata extraction and profiling to comprehensively understand the candidate models or algorithms based on the user requirements. You have the following main responsibilities to complete.
1. Retrieve a list of well-performing candidate ML models and AI algorithms for the given dataset based on the user's requirement and instruction.
2. Perform hyperparameter optimization for those candidate models or algorithms.
3. Extract useful information and underlying characteristics of the candidate models or algorithms using metadata extraction and profiling techniques.
4. Select the top-k (`k` will be given) well-performing models or algorithms based on the hyperparameter optimization and profiling results.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 22



DATA_AGENT_PROMPT = PromptTemplate.from_template("""You are the world's best data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
1. Retrieve a dataset from the user or search for the dataset based on the user instruction.
2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
3. Perform data augmentation as neccesary.
4. Extract useful information and underlying characteristics of the dataset.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 22



PROMPT_AGENT_PROMPT = PromptTemplate.from_template(""" You are an assistant project manager in the AutoML development team.
Your task is to parse the user's requirement into a valid JSON format using the JSON specification schema as your reference. Your response must exactly follow the given JSON schema and be based only on the user's instruction.
Make sure that your answer contains only the JSON response without any comment or explanation because it can cause parsing errors.

#JSON SPECIFICATION SCHEMA#
```json
{json_specification}
```

Your response must begin with "```json" or "{{" and end with "```" or "}}", respectively.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 22



AGENT_MANAGER_PROMPT = PromptTemplate.from_template(""" You are an experienced senior project manager of a automated machine learning project (AutoML). You have two main responsibilities as follows.
1. Receive requirements and/or inquiries from users through a well-structured JSON object.
2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 21



ZERO_SHOT_PROMPT_FOR_GPT_3_5_AND_GPT_4_BASELINES = PromptTemplate.from_template("""You are a helpful intelligent assistant. Now please help solve the following machine learning task.
[Task]
{user instruction}
[{file_name}.py] 

```python
{full-pipeline skeleton script}
```
Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.""") # Tham khảo tại: https://arxiv.org/pdf/2410.02958 - Trang 20



ROUTER_PROMPT = PromptTemplate.from_template("""You need to decide which domain to route the user query to. You have two domains to choose from:
- Records: contains medical records of the patient, such as diagnosis, treatment, and prescriptions.
- Insurance: contains frequently asked questions about insurance policies, claims, and coverage.

Output only the domain name.""") # Tham khảo tại: https://learning.oreilly.com/library/view/learning-langchain/9781098167271/ch05.html#ch05_summary_1736545670031127



REACT_PROMPT = PromptTemplate.from_template("""{begin_of_text}
{start_header_id}system{end_header_id}
You are an AI assistant that follows the ReAct reasoning framework. 
You have access to the following APIs:

{tools_desc}

Use the following strict format:

### Input Format:

Question: The original query provided by the user.
Thought: Logical reasoning before executing an action.
Action: The action to be taken, chosen from available tools: {tools_name}.
Action Input: The required input for the action.
Observation: The outcome of executing the action. 
...(Repeat the thought/action/observation loop as needed)
Thought: I now know the final answer.
Final Answer: Provide the final answer.

Begin{end_of_turn_id}""") # Tham khảo tại: https://github.com/OpenBMB/MiniCPM-CookBook/blob/d0772b24af057c8e7f5d6e12fd00f3cde0481a3c/agent_demo/agent_demo.py#L79



TOOL_DESC_PROMPT = PromptTemplate.from_template("""{name_for_model}: Call this tool to interact with the {name_for_human} API. 
What is the {name_for_human} API useful for? 
{description_for_model}.
Type: {type}.
Properties: {properties}.
Required: {required}.""") # Tham khảo tại: https://github.com/OpenBMB/MiniCPM-CookBook/blob/d0772b24af057c8e7f5d6e12fd00f3cde0481a3c/agent_demo/agent_demo.py#L76