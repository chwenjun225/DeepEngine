### **🚀 Chain-of-Thought (CoT) Prompt Template**
Chain-of-Thought (CoT) prompting giúp mô hình suy luận tốt hơn bằng cách hướng dẫn nó chia nhỏ các bước suy nghĩ trước khi đưa ra câu trả lời.

Dưới đây là một **template chuẩn cho CoT Prompting**, có thể điều chỉnh theo từng trường hợp cụ thể.

---

## **🔹 Basic Chain-of-Thought Prompt**
```plaintext
Let's think step by step.

Question: {user_question}

Step 1: First, let's analyze the question and identify key information.
Step 2: Next, we break the problem down into logical steps.
Step 3: We apply relevant knowledge to solve each part systematically.
Step 4: Finally, we combine our reasoning to arrive at the most accurate answer.

Now, let's begin:
```
🔥 **Mô tả**:  
- Dùng câu lệnh *Let's think step by step* để kích thích mô hình suy luận tuần tự.  
- Hướng dẫn mô hình suy nghĩ theo các bước rõ ràng.  

---

## **🔹 CoT for Math & Logical Problems**
```plaintext
Let's solve this problem step by step.

Problem: {math_problem}

Step 1: Identify given information and unknown variables.
Step 2: Determine the best approach to solve the problem.
Step 3: Perform calculations while checking for accuracy.
Step 4: Summarize the final answer with a clear explanation.

Now, let's compute:
```
📌 **Dùng khi cần mô hình suy luận toán học, logic hoặc tính toán chính xác**.  

---

## **🔹 CoT for Reasoning-Based Questions**
```plaintext
Let's break this down step by step.

Question: {user_query}

Step 1: Identify relevant details in the question.
Step 2: Consider different perspectives or possible answers.
Step 3: Use logical reasoning or factual knowledge to evaluate options.
Step 4: Provide a well-structured answer based on our findings.

Final Answer:
```
📌 **Dùng cho câu hỏi cần suy luận, như AI ethics, lập luận pháp lý, phân tích chính trị**.  

---

## **🔹 CoT for Coding/Algorithm Problems**
```plaintext
Let's analyze this coding problem step by step.

Task: {coding_problem}

Step 1: Understand the problem statement and constraints.
Step 2: Identify key concepts or algorithms that apply.
Step 3: Plan an efficient solution before implementation.
Step 4: Write and test the code to ensure correctness.

Now, let's develop the solution:
```
📌 **Dùng khi cần giải thích hoặc thiết kế thuật toán**.  

---

## **🔹 CoT for Research & Scientific Questions**
```plaintext
Let's break this research question down step by step.

Question: {scientific_query}

Step 1: Identify the core concept and relevant background knowledge.
Step 2: Explore potential explanations or theories related to the topic.
Step 3: Evaluate empirical evidence and research findings.
Step 4: Formulate a well-supported conclusion.

Final Explanation:
```
📌 **Dùng khi mô hình cần tổng hợp thông tin khoa học, nghiên cứu hoặc kiến thức chuyên sâu**.  

---

## **🔹 Example Use Case**
### **User Input:**
> *What are the advantages of solar energy over fossil fuels?*

### **AI Response with CoT Prompting:**
```plaintext
Let's break this down step by step.

Step 1: Identify relevant details in the question.
- The question asks about advantages, meaning we need to compare benefits.

Step 2: Consider different perspectives or possible answers.
- We can evaluate solar energy in terms of sustainability, cost, and environmental impact.

Step 3: Use logical reasoning or factual knowledge to evaluate options.
- Solar energy is renewable, while fossil fuels are finite.
- Solar energy has lower long-term costs due to free sunlight.
- Solar power reduces carbon emissions, whereas fossil fuels contribute to climate change.

Step 4: Provide a well-structured answer based on our findings.
- Solar energy is more sustainable, cost-effective in the long run, and environmentally friendly.

Final Answer:
Solar energy is a superior alternative to fossil fuels due to its renewable nature, long-term cost savings, and reduced environmental impact.
```

---

## **📌 Tổng kết**
| **Loại Câu Hỏi** | **Prompt CoT Tương Ứng** |
|-----------------|----------------------|
| **Câu hỏi chung** | "Let's think step by step..." |
| **Toán học/Logic** | "Let's solve this problem step by step..." |
| **Lập trình/Thuật toán** | "Let's analyze this coding problem step by step..." |
| **Khoa học/Nghiên cứu** | "Let's break this research question down step by step..." |

🚀 **Bạn muốn tối ưu thêm CoT Prompt theo lĩnh vực nào không?** 😃