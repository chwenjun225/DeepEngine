### **Ví dụ về Reflection trong LangGraph**
Trong **LangGraph**, một framework để xây dựng workflows cho AI với **LangChain**, ta có thể sử dụng **Reflection** để tạo vòng lặp giữa **creator prompt** và **reviser prompt**. Dưới đây là một ví dụ về cách thực hiện kỹ thuật này:

---

### **1. Cài đặt thư viện cần thiết**
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import Dict, List
```

---

### **2. Định nghĩa trạng thái**
Chúng ta sử dụng một dictionary để lưu trữ nội dung và phản hồi.

```python
class ReflectionState(Dict):
    input_text: str
    ai_response: str
    critique: str
    revised_response: str
```

---

### **3. Tạo các node xử lý**
- **Creator Node**: Sinh nội dung ban đầu.
- **Critique Node**: Đánh giá nội dung và phản hồi.
- **Reviser Node**: Sửa đổi dựa trên phản hồi.

```python
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)

def creator(state: ReflectionState) -> ReflectionState:
    """Tạo nội dung ban đầu từ đầu vào của người dùng."""
    prompt = [
        SystemMessage(content="Bạn là một AI tạo nội dung chuyên nghiệp."),
        HumanMessage(content=f"Viết một đoạn văn về chủ đề: {state['input_text']}")
    ]
    response = llm.invoke(prompt)
    state["ai_response"] = response.content
    return state

def critique(state: ReflectionState) -> ReflectionState:
    """Đưa ra phản hồi mang tính phản biện về nội dung đã tạo."""
    prompt = [
        SystemMessage(content="Bạn là một nhà phê bình nội dung AI. Hãy đánh giá đoạn văn sau và đề xuất cải thiện."),
        HumanMessage(content=state["ai_response"])
    ]
    critique = llm.invoke(prompt)
    state["critique"] = critique.content
    return state

def reviser(state: ReflectionState) -> ReflectionState:
    """Chỉnh sửa nội dung dựa trên phản hồi."""
    prompt = [
        SystemMessage(content="Bạn là một AI chỉnh sửa nội dung. Hãy cải thiện đoạn văn dưới đây dựa trên phản hồi."),
        HumanMessage(content=f"Đoạn văn gốc: {state['ai_response']}\n\nPhản hồi: {state['critique']}")
    ]
    revised = llm.invoke(prompt)
    state["revised_response"] = revised.content
    return state
```

---

### **4. Xây dựng đồ thị LangGraph**
```python
workflow = StateGraph(ReflectionState)

workflow.add_node("creator", creator)
workflow.add_node("critique", critique)
workflow.add_node("reviser", reviser)

workflow.add_edge("creator", "critique")
workflow.add_edge("critique", "reviser")
workflow.add_edge("reviser", END)

workflow.set_entry_point("creator")
graph = workflow.compile()
```

---

### **5. Chạy Workflow**
```python
input_state = {"input_text": "Tầm quan trọng của AI trong giáo dục"}
final_state = graph.invoke(input_state)

print("✍️ Nội dung ban đầu:\n", final_state["ai_response"])
print("\n🧐 Phản hồi chỉnh sửa:\n", final_state["critique"])
print("\n✅ Nội dung đã chỉnh sửa:\n", final_state["revised_response"])
```

---

### **🛠️ Giải thích cách hoạt động**
1. **Creator** sinh ra nội dung dựa trên đầu vào của người dùng.
2. **Critique** đánh giá nội dung và đưa ra phản hồi.
3. **Reviser** sửa đổi nội dung theo phản hồi.
4. Kết quả cuối cùng là một phiên bản cải thiện của nội dung ban đầu.

---

### **📌 Ứng dụng thực tế**
- Viết bài tự động với tự chỉnh sửa.
- Hỗ trợ AI tự đánh giá và cải thiện câu trả lời.
- Xây dựng chatbot phản hồi chính xác hơn.

Đây là cách áp dụng **Reflection** để giúp mô hình AI trở nên mạnh mẽ và hiệu quả hơn trong LangGraph. 🚀