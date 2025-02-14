def planning(prompt_user, prompt_planning, rag_output):
	"""Xử lý thông tin từ RAG để tạo prompt phù hợp cho LLM."""
	planning_prompt = prompt_planning.invoke(
		{"question": prompt_user, "context": rag_output}
	)
	return planning_prompt