from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
	# Các tin nhắn có kiểu dữ liệu là "list". 
	# Hàm `add_messages` trong phần annotation 
	# xác định cách cập nhật trạng thái này.  
	# (Trong trường hợp này, nó **thêm tin nhắn 
	# mới vào danh sách** thay vì thay thế 
	# toàn bộ tin nhắn trước đó).
	messages: Annotated[list, add_messages]
