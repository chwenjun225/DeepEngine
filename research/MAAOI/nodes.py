from langchain_core.messages import AIMessage, SystemMessage



from state import State
from const import (
	LLM,
	DEFECT_REASONING_AGENT_PROMPT_MSG, 
)
from utils import (
	trim_context				,
	has_agent_got_sys_prompt	,
	has_agent_got_name_attr		,
	replace_message_content		,
	prepare_context				,
	get_latest_msg				,
)



def TEMPORAL_PATTERN_AGENT(state: State) -> State:
	"""Nhóm các loại lỗi lại với nhau và tạo prompt mô tả rõ ràng cho agent kế tiếp."""
	ai_msg = get_latest_msg(state, "VISION_AGENT_MSGS")
	ctx_frames_metadata = ai_msg.content if isinstance(ai_msg.content, list) else eval(ai_msg.content)

	label_map: dict[str, list[int]] = {}
	label_confidence: dict[str, list[float]] = {}

	for frame in ctx_frames_metadata:
		frame_id = frame["id"]
		for obj in frame["metadata"]:
			label = obj["label"]
			conf = obj["confidence"]

			label_map.setdefault(label, []).append(frame_id)
			label_confidence.setdefault(label, []).append(conf)

	description = "This is a summary of defect types detected across frames:"

	for label in sorted(label_map):
		frames = sorted(set(label_map[label]))
		avg_conf = sum(label_confidence[label]) / len(label_confidence[label])
		description += f"\n- The defect '{label}' was detected in frames {frames} with {len(label_confidence[label])} occurrences, and an average confidence of {avg_conf:.2f}."

	return {"TEMPORAL_PATTERN_AGENT_MSGS": [
		AIMessage(content=description, 
			name="TEMPORAL_PATTERN_AGENT_MSGS"
		)]}


# TEMPORAL_PATTERN_AGENT's output:
#
# Below is a summary of defect types detected across frames:
# - The defect 'missing_hole' was detected in frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] with 27 occurrences, and an average confidence of 0.72.
# - The defect 'mouse_bite' was detected in frames [4] with 1 occurrences, and an average confidence of 0.10.
# - The defect 'short' was detected in frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] with 12 occurrences, and an average confidence of 0.22.
# - The defect 'spur' was detected in frames [0, 1, 2, 4, 5, 6, 7, 8, 9] with 10 occurrences, and an average confidence of 0.27.

# Please analyze the trends, severity levels, and any potential correlations between these defects.
#



def DEFECT_REASONING_AGENT(state: State) -> State:
	"""Đánh giá lỗi, phát hiện pattern và gợi ý hành động QC."""
	summary = get_latest_msg(state, type_msgs="TEMPORAL_PATTERN_AGENT_MSGS")
	ai_msg = LLM.invoke([
		SystemMessage(DEFECT_REASONING_AGENT_PROMPT_MSG), 
		summary
	])
	return {"DEFECT_REASONING_AGENT_MSGS": [
		AIMessage(content=ai_msg.content, 
			name="DEFECT_REASONING_AGENT_MSGS"
		)]} ### Đã hoàn thành nhưng prompt sinh ra dài, cần tinh chỉnh lại 



def CRITICAL_ASSESSMENT_AGENT(state: State) -> State:
	"""Đánh giá mức độ lỗi."""
	return state 



def REPORT_GENERATOR_AGENT(state: State) -> State:
	"""Báo cáo kết quả cho người dùng.""" 
	return state 



def VISUAL_AGENT(state: State) -> State:
	"""Đưa ra tọa độ lỗi cuối cùng sau quá trình suy luận."""
	return state 
