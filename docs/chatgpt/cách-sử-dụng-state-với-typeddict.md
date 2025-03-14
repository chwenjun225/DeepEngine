Dưới đây là phiên bản tối ưu của `State` sử dụng `TypedDict` thay vì `BaseModel`. Tôi cũng đã bổ sung các hàm tiện ích để thay thế các phương thức trước đây của `BaseModel`.  

---

### **📌 Phiên bản `State` sử dụng `TypedDict`**
```python
from typing_extensions import TypedDict, List, Dict, Optional
from collections import defaultdict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Định nghĩa kiểu dữ liệu cho tin nhắn
MSG_TYPES = {SystemMessage: "SYS", HumanMessage: "HUMAN", AIMessage: "AI"}

# Cấu trúc lưu trữ mặc định
DEFAULT_AGENTS: Dict[str, Dict[str, List[BaseMessage]]] = {
    "MANAGER_AGENT":     {"SYS": [], "HUMAN": [], "AI": []},
    "REQUEST_VERIFY":    {"SYS": [], "HUMAN": [], "AI": []},
    "PROMPT_AGENT":      {"SYS": [], "HUMAN": [], "AI": []},
    "RAP":               {"SYS": [], "HUMAN": [], "AI": []},
    "DATA_AGENT":        {"SYS": [], "HUMAN": [], "AI": []},
    "MODEL_AGENT":       {"SYS": [], "HUMAN": [], "AI": []},
    "OP_AGENT":          {"SYS": [], "HUMAN": [], "AI": []},
}

# Hàm tạo dictionary mặc định cho tin nhắn
def default_messages() -> Dict[str, Dict[str, List[BaseMessage]]]:
    return defaultdict(lambda: {"SYS": [], "HUMAN": [], "AI": []}, DEFAULT_AGENTS.copy())

# Định nghĩa State sử dụng TypedDict
class State(TypedDict):
    human_query: List[HumanMessage]  # Danh sách truy vấn của người dùng
    messages: Dict[str, Dict[str, List[BaseMessage]]]  # Lưu trữ tin nhắn theo agent
    is_last_step: bool  # Cờ xác định bước cuối cùng
    remaining_steps: int  # Số bước còn lại

# Hàm tạo State mặc định
def create_state() -> State:
    return {
        "human_query": [],
        "messages": default_messages(),
        "is_last_step": False,
        "remaining_steps": 3,
    }
```

---

### **📌 Các hàm hỗ trợ (thay thế phương thức của `BaseModel`)**

Dưới đây là các hàm giúp bạn thao tác với `State`, thay thế cho các phương thức trước đây trong `BaseModel`.

#### **1. Lấy tất cả tin nhắn theo thứ tự thời gian**
```python
def get_all_msgs(state: State) -> List[BaseMessage]:
    """Lấy tất cả tin nhắn từ tất cả các agents theo thứ tự thời gian."""
    all_messages = []
    for agent_messages in state["messages"].values():
        for msg_list in agent_messages.values():
            all_messages.extend(msg_list)
    return all_messages
```

#### **2. Lấy tin nhắn mới nhất từ một agent và một loại tin nhắn**
```python
def get_latest_msg(state: State, agent_type: str, msg_type: str) -> Optional[BaseMessage]:
    """Lấy tin nhắn mới nhất từ một agent cụ thể."""
    if agent_type not in state["messages"]:
        return None
    return state["messages"][agent_type][msg_type][-1] if state["messages"][agent_type][msg_type] else None
```

#### **3. Lấy danh sách tin nhắn từ một agent và loại tin nhắn**
```python
def get_msgs_by_node_and_msgs_type(state: State, node: str, msgs_type: str) -> List[BaseMessage]:
    """Lấy tất cả tin nhắn từ một agent và loại tin nhắn."""
    if node not in state["messages"]:
        return []
    return state["messages"][node].get(msgs_type, [])
```

#### **4. Thêm tin nhắn vào `State` (tránh trùng lặp nếu cần)**
```python
def add_unique_msg(state: State, node: str, msg_type: str, msg: BaseMessage) -> None:
    """Thêm một tin nhắn nếu nó chưa có trong danh sách."""
    if node not in state["messages"]:
        state["messages"][node] = {"SYS": [], "HUMAN": [], "AI": []}
    
    existing_msgs = state["messages"][node][msg_type]
    if not existing_msgs or existing_msgs[-1].content != msg.content:
        existing_msgs.append(msg)
```

---

### **📌 Ví dụ sử dụng `State` với `TypedDict`**
```python
# Khởi tạo trạng thái
state = create_state()

# Thêm một truy vấn mới từ người dùng
state["human_query"].append(HumanMessage(content="Hello!"))

# Thêm tin nhắn AI vào agent MANAGER_AGENT
add_unique_msg(state, "MANAGER_AGENT", "AI", AIMessage(content="Hi! How can I assist you?"))

# Lấy tin nhắn mới nhất từ MANAGER_AGENT
latest_msg = get_latest_msg(state, "MANAGER_AGENT", "AI")
print("Latest AI message:", latest_msg.content if latest_msg else "No messages")

# Lấy tất cả tin nhắn từ state
all_msgs = get_all_msgs(state)
print("\nAll Messages in State:")
for msg in all_msgs:
    print("-", msg.content)
```

---

## **📌 Kết luận**
### 🔹 **Lợi ích khi chuyển từ `BaseModel` sang `TypedDict`**
✅ **Hiệu suất cao hơn** (Không có kiểm tra dữ liệu `pydantic`, hoạt động nhanh hơn).  
✅ **Không cần `.dict()` hoặc `.json()`**, làm việc trực tiếp với dữ liệu dạng `dict`.  
✅ **Giảm bộ nhớ sử dụng**, không cần đối tượng `BaseModel` cồng kềnh.  
✅ **Dễ dàng tích hợp với API hoặc hệ thống xử lý JSON**.  

### 🔹 **Nhược điểm của `TypedDict` so với `BaseModel`**
❌ **Không hỗ trợ kiểm tra dữ liệu tự động**, bạn phải đảm bảo dữ liệu đầu vào hợp lệ.  
❌ **Không có phương thức tiện ích như `.copy()`, `.dict()`, `.json()`**, cần viết lại các hàm hỗ trợ.  

🚀 **Bạn đã chọn hướng đi đúng!** Nếu bạn không cần kiểm tra dữ liệu tự động, thì `TypedDict` giúp tăng hiệu suất đáng kể. Bây giờ, bạn có thể dễ dàng quản lý `State` như một dictionary bình thường mà không bị ràng buộc bởi `pydantic.BaseModel`.  

Bạn thấy phiên bản này thế nào? Cần mình tối ưu thêm chỗ nào không? 😊