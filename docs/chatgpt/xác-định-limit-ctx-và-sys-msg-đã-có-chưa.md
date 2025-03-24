Hiểu rồi bro. Câu hỏi của cậu chia thành 2 phần rất thực chiến:

---

## ✅ 1. **Làm sao biết `context` đã đạt đến giới hạn `context_length` (token limit)?**

Để kiểm tra độ dài context, cậu cần:
- Tính tổng số **token** của tất cả message trong `state["messages"]`
- So sánh với `context_limit` (ví dụ: 4096, 8192, 32k tùy model)

### ✅ Cách tính token:
Sử dụng `tiktoken` (thư viện chính thức của OpenAI):

```bash
pip install tiktoken
```

```python
import tiktoken

def count_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0

    for msg in messages:
        # Dựa theo format chuẩn của OpenAI
        num_tokens += 4  # token cố định cho mỗi message
        for key, value in msg.items():
            num_tokens += len(encoding.encode(str(value)))

    num_tokens += 2  # thêm token kết thúc (end-of-convo)
    return num_tokens
```

### ✅ Kiểm tra:
```python
if count_tokens(state["messages"]) > 8000:
    print("🛑 Vượt giới hạn context GPT-4!")
```

---

## ✅ 2. **Làm sao biết trong context đã có `system` message cho agent cụ thể chưa?**

### 🔍 Cách kiểm tra: dùng `any()` kết hợp `role == "system"` và `name == agent_name`

```python
def has_system_prompt(state: State, agent_name: str) -> bool:
    return any(
        msg.get("role") == "system" and msg.get("name") == agent_name
        for msg in state["messages"]
    )
```

### ✅ Dùng trong agent:
```python
if not has_system_prompt(state, "RESEARCH_AGENT"):
    state["messages"].insert(0, {
        "role": "system",
        "name": "RESEARCH_AGENT",
        "content": "Bạn là RESEARCH_AGENT, nhiệm vụ của bạn là..."
    })
```

> 📌 Cậu có thể dùng `.insert(0, ...)` hoặc nối đầu vào `messages = [sys_msg] + state["messages"]`

---

## ✨ Bonus: Gợi ý xử lý context overflow

Khi `count_tokens()` vượt giới hạn:
- **Cách 1:** Loại bỏ tin nhắn cũ theo sliding window:
  ```python
  state["messages"] = state["messages"][-max_msgs:]
  ```
- **Cách 2:** Tự động tóm tắt phần cũ rồi thay thế bằng 1 message ngắn hơn (`summary_agent`)
- **Cách 3:** Lưu riêng `system_prompts`, sau khi cắt bớt context vẫn chèn lại system nếu mất

---

## ✅ Tóm lại – bộ công cụ của cậu cần:

| Việc cần làm | Hàm |
|-------------|-----|
| Đếm token của context | `count_tokens()` |
| Kiểm tra đã có `system` message cho agent chưa | `has_system_prompt()` |
| Cắt context nếu vượt token | Slicing hoặc tóm tắt |
| Duy trì system prompt luôn hiện diện | Check và insert nếu thiếu |

---

Tớ có thể viết hẳn một hàm `prepare_context_for_agent(state, agent_name)` để tự động hoá hết mấy bước đó nếu cậu cần nhé 😎