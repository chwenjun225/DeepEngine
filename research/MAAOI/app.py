import fire 



from langchain_core.messages import HumanMessage



from agentic import AGENTIC
from const_vars import QUERIES, CONFIG
from state import default_messages



def main() -> None:
	"""Xử lý truy vấn của người dùng và hiển thị phản hồi từ AI."""
	for i, user_query in enumerate(QUERIES, 1):
		print(f"\n👨_query_{i}:")
		print(user_query)
		print("\n🤖_response:")
		user_query = user_query.strip()
		if user_query.lower() == "exit": break
		state_data = AGENTIC.invoke(
			input={
				"human_query": [HumanMessage(user_query)], 
				"messages": default_messages()
			}, config=CONFIG
		)
		if not isinstance(state_data, dict): 
			raise ValueError("[ERROR]: app.invoke() không trả về dictionary.")
		if "messages" not in state_data: 
			raise ValueError("[ERROR]: Key 'messages' không có trong kết quả.")
		messages = state_data["messages"]
		print("\n===== [CONVERSATION RESULTS] =====\n")
		display_conversation_results(messages)
		print("\n===== [END OF CONVERSATION] =====\n")



def display_conversation_results(messages: dict) -> None:
	"""Hiển thị kết quả hội thoại từ tất cả các agent trong hệ thống."""
	if not messages:
		print("[INFO]: Không có tin nhắn nào trong hội thoại.")
		return
	for node, msgs in messages.items():
		print(f"\n[{node}]")
		if isinstance(msgs, dict):
			for msg_category, msg_list in msgs.items():
				if msg_list:
					print(f"  {msg_category}:")
					for msg in msg_list:
						content = getattr(msg, "content", "[No content]")
						print(f"\t- {content}")
		else:
			raise ValueError(f"`msgs` phải là một dictionary chứa danh sách tin nhắn, `msgs` hiện tại là: {msgs}")



if __name__ == "__main__":
	fire.Fire(main)