# Multi AI-Agentics for automated training and inference AI-Narrow
**Trần Văn Tuấn - Feb. 27, 2025**
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
