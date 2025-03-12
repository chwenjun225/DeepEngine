# Multi AI-Agentics for automated training and inference AI-Narrow
**Trần Văn Tuấn - Feb. 27, 2025**
## Workflow
![Agent System Workflow](images/multi_agent_workflow.jpg)
## 1. Introduction 
### 1.1 Problem Statement
- Với tốc độ phát triển nhanh chóng của trí tuệ nhân tạo (AI), hạn chế của các AI-Agent hiện tại là khó khăn trong việc thích nghi và xử lý các tác vụ phức tạp trong môi trường động. Hiện tại, nhiều AI-Agent hoạt động theo các quy tắc tĩnh, khó cải thiện khả năng tư duy, thích nghi nhanh chóng và tự tối ưu hóa.
- Dự án này đề xuất việc tích hợp AI-Narrow vào AI-Agent nhằm tối ưu hóa quá trình xử lý tác vụ. AI-Agent sẽ học hỏi từ AI-Narrow để nâng cao hiệu suất, giúp hệ thống trở nên thông minh và chính xác hơn.
### 1.2 Project Goals and Objective
- Xây dựng hệ thống AI-Agent tự tạo ra các AI-Narrow để hoàn thành task được giao cho.
- Tối ưu hóa AI-Agent theo mô hình học tích hợp (Federated Learning).
- Đánh giá hiệu quả AI-Agent dựa trên benchmarking với các AI truyền thống.
### 1.3 Keywords
- AI-Agent, AI-Narrow, Adaptive Learning, Reinforcement Learning, Federated Learning, Multi-Agent System.
### 1.4 Background and Justification of the Project
- AI-Agent hiện tại còn bị giới hạn về **khả năng tự cải thiện và thích nghi**.
- Việc **tích hợp AI-Narrow** vào Multi AI-Agentics giúp hệ thống tự động tối ưu hóa, tự học hỏi với mục tiêu phát triển thành một hệ thống Multi-AGI.
### 1.4 Novelty and Innovation
- Multi AI-Agentics động bộ tự học hỏi và nâng cao hiệu quả từ AI-Narrow.
- Hệ thống được thiết kế và có tính thích nghi cao.
## 2. Related Works
- ...
## Reference
1. D. B. Acharya, K. Kuppan and B. Divya, "Agentic AI: Autonomous Intelligence for Complex Goals—A Comprehensive Survey," in IEEE Access, vol. 13, pp. 18912-18936, 2025, doi: 10.1109/ACCESS.2025.3532853. keywords: {Artificial intelligence;Surveys;Ethics;Reinforcement learning;Hands;Adaptation models;Medical services;Automation;Translation;Systematic literature review;Agentic AI;autonomous systems;human-AI collaboration;adaptability;governance frameworks;ethical AI}. URL https://ieeexplore.ieee.org/abstract/document/10849561

2. Alex Sheng. From Language Models to Practical Self-Improving Computer Agents . URL https://arxiv.org/abs/2404.11964

3. Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023. URL: https://arxiv.org/abs/2302.04761

4. Adam Santoro, Sergey Bartunov, Matthew M. Botvinick, Daan Wierstra, and Timothy P. Lillicrap. One-shot learning with memory-augmented neural networks. CoRR, abs/1605.06065, 2016. URL http://arxiv.org/abs/1605.06065.

5. Juergen Schmidhuber. Goedel machines: Self-referential universal problem solvers making provably optimal self-improvements. Lecture Notes in Computer Science-Adaptive Agents and Multi-Agent Systems II, 3394, 2006. URL https://arxiv.org/abs/cs/0309048

6. Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL https://arxiv.org/abs/2303.11366

7. Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions, 2023. URL: https://arxiv.org/abs/2212.10509

8. Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents, 2023a. URL: https://arxiv.org/abs/2207.01206

9. Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023b. URL: https://arxiv.org/abs/2210.03629

10. Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. Self-taught optimizer (stop): Recursively self-improving code generation, 2024. URL: https://arxiv.org/abs/2310.02304

11. Ishizaki, R. & Sugiyama, Mahito (forthcoming). Large Language Models: Assessment for Singularity. AI and Society. URL: https://philpapers.org/rec/ISHLLM

12. Ishibashi Y, Yano T, Oyamada M. Can Large Language Models Invent Algorithms to Improve Themselves?. arXiv preprint arXiv:2410.15639. 2024 Oct 21.

13. Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla , Olaf Wiest, Xiangliang Zhang. Large Language Model based Multi-Agents: A Survey of Progress and Challenges. URL: https://arxiv.org/pdf/2402.01680

14. Haoyuan Li, Hao Jiang, Tianke Zhang, Zhelun Yu, Aoxiong Yin, Hao Cheng, Siming Fu, YuhaoZhang, and Wanggui He. Traineragent: Customizable and efficient model training through llmpowered multi-agent system. arXiv preprint arXiv:2311.06622, 2023. URL: https://arxiv.org/abs/2311.06622

15. Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate, 2023. URL: https://arxiv.org/abs/2305.14325

16. Patara Trirat , Wonyong Jeong, Sung Ju Hwang. AUTOML-AGENT: AMULTI-AGENT LLM FRAMEWORK FOR FULL-PIPELINE AUTOML. URL https://arxiv.org/pdf/2410.02958

17. https://learning.oreilly.com/library/view/llm-engineers-handbook/9781836200079/

18. https://learning.oreilly.com/library/view/prompt-engineering-for/9781098153427/ch06.html#id95

19. https://learning.oreilly.com/library/view/learning-langchain/9781098167271/ch04.html

20. https://learning.oreilly.com/library/view/generative-ai-with/9781835083468/

21. Cách sử dụng nhiều Agent prompt: https://learning.oreilly.com/library/view/learning-langchain/9781098167271/ch05.html


## RETRIEVAL_AUGMENTED_PLANNING

state.messages["RAP"]["AI"][-1].content = """<|start_header_id|>assistant<|end_header_id|>\n\nBased on the provided JSON object, 
I will devise an actionable plan for image classification into species categories using a ResNet-50 Transfer Learning Convolutional Neural Network (CNN).


**Plan:**


### Step 1: Data Collection and Preprocessing
**Data Retrieval**: Collect a large dataset of images from various sources such as the Butterfly Image Classification dataset.
**Image Preprocessing**: Clean, resize, normalize, and preprocess the collected images to ensure they are in a suitable format for training a deep learning model. 
    This may involve data augmentation techniques like rotation, flipping, color jittering, etc.
**Data Labeling**: Label each image with its corresponding species category using a manual annotation process or by utilizing an existing labeling tool.


### Step 2: Model Selection and Training
**Model Selection**: Choose the ResNet-50 Transfer Learning Convolutional Neural Network (CNN) as the model for this task. 
    This is because it has been shown to perform well on image classification tasks.
**Model Pretraining**: Train the pre-trained ResNet-50 CNN using the collected dataset with a suitable batch size, number of epochs, and learning rate.
**Fine-tuning**: Fine-tune the pre-trained model on our specific dataset by adjusting the hyperparameters to optimize its performance.


### Step 3: Model Evaluation
**Model Evaluation Metrics**: Use metrics such as accuracy, precision, recall, F1-score, and ROC-AUC score to evaluate the trained model's performance.
**Hyperparameter Tuning**: Perform hyperparameter tuning using techniques like grid search or random search to optimize the model's performance.


### Step 4: Model Deployment
**Model Serving**: Deploy the trained model in a production-ready environment using a cloud-based platform such as AWS SageMaker, Google Cloud AI Platform, or Azure Machine Learning.
**Model Serving Model**: Integrate the deployed model into an application that can be used for image classification tasks.


### Step 5: Maintenance and Updates
**Model Monitoring**: Continuously monitor the performance of the deployed model using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC score.  
**Model Updates**: Update the model periodically to ensure it remains accurate and effective in handling new data.


### Step 6: Documentation and Sharing
**Documentation**: Document the entire machine learning pipeline, including data collection, preprocessing, model selection, training, evaluation, deployment, and maintenance steps.
**Sharing**: Share the documented pipeline with other stakeholders or researchers to facilitate collaboration and knowledge sharing.
**Code Example (Python):**
```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# Load the dataset
df = pd.read_csv('data.csv')
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('species', axis=1), df['species'], test_size=0.2, random_state=42)\n\n
# Preprocess data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the model architecture
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(len(set(y_train)), activation='softmax')(x)
# Define the model
model = Model(inputs=base_model.input, outputs=x)
# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))
# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Accuracy: {accuracy:.2f}')
# Save the model
model.save('species_classification_model.h5')
```
This code snippet demonstrates a basic example of how to implement a ResNet-50 CNN for image classification using Python. 
The model is trained on the Butterfly Image Classification dataset and evaluated on a test set.<|eot_id|><|end_of_text|>"""