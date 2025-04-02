import re 
import json


from langchain_core.messages import AIMessage, SystemMessage



from state import State
from const import (LLM, DEFECT_REASONING_AGENT_PROMPT_MSG, QC_JUDGEMENT_AGENT_PROMPT_MSG)
from utils import (get_latest_msg)



def TEMPORAL_PATTERN_AGENT(state: State) -> State:
	"""Nhóm các loại lỗi lại với nhau và tạo prompt mô tả rõ ràng cho agent kế tiếp."""
	prev_agent_msg = get_latest_msg(state, "VISION_AGENT_MSGS")
	ctx_frames_metadata = prev_agent_msg.content if isinstance(prev_agent_msg.content, list) else eval(prev_agent_msg.content)

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
# ================================== Ai Message ==================================
# Name: TEMPORAL_PATTERN_AGENT_MSGS

# This is a summary of defect types detected across frames:
# - The defect 'missing_hole' was detected in frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] with 27 occurrences, and an average confidence of 0.72.
# - The defect 'mouse_bite' was detected in frames [4] with 1 occurrences, and an average confidence of 0.10.
# - The defect 'short' was detected in frames [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] with 12 occurrences, and an average confidence of 0.22.
# - The defect 'spur' was detected in frames [0, 1, 2, 4, 5, 6, 7, 8, 9] with 10 occurrences, and an average confidence of 0.27.



def DEFECT_REASONING_AGENT(state: State) -> State:
	"""Đánh giá lỗi, phát hiện pattern và gợi ý hành động QC."""
	prev_agent_msg = get_latest_msg(state, type_msgs="TEMPORAL_PATTERN_AGENT_MSGS")
	ai_msg = LLM.invoke([
		SystemMessage(DEFECT_REASONING_AGENT_PROMPT_MSG), 
		prev_agent_msg
	])
	return {"DEFECT_REASONING_AGENT_MSGS": [
		AIMessage(content=ai_msg.content, 
			name="DEFECT_REASONING_AGENT_MSGS"
		)]} 
# ================================== Ai Message ==================================
# Name: DEFECT_REASONING_AGENT_MSGS
#
# - The defect 'tangle' was detected in frames [3, 4, 5, 6, 7, 8, 9] with 14 occurrences, and an average confidence of 0.35.
#
# 1. Most critical defects are 'missing_hole' (high confidence) and 'mouse_bite' (low confidence but severe impact).
# 2. There is a pattern where 'short', 'spur', and 'tangle' defects occur in frames [0-9] with varying frequencies, while 'mouse_bite' only occurs once.
# 3. Ignore low-confidence defects like 'mouse_bite'. Escalate 'missing_hole' due to high confidence and potential impact on product quality.
#
# Technical explanation:
# The defect detection summary indicates that the most critical defects are 'missing_hole' and 'mouse_bite', with 'missing_hole' being more severe due to its higher confidence. 
# The occurrence of multiple defects ('short', 'spur', 'tangle') across frames suggests a need for closer inspection, while low-confidence defects like 'mouse_bite' can be ignored unless further investigation is warranted.



def QC_JUDGEMENT_AGENT(state: State) -> State:
	"""Dựa trên thông tin được reasoning, đưa ra kết luận OK hoặc NG."""
	prev_agent_msg = get_latest_msg(state, type_msgs="DEFECT_REASONING_AGENT_MSGS")
	ai_msg = LLM.invoke([
		SystemMessage(QC_JUDGEMENT_AGENT_PROMPT_MSG), 
		prev_agent_msg
	])
	def extract_qc_judgement(text: str) -> str:
		"""Hàm tách OK/NG an toàn bằng regex."""
		match = re.search(r"\b(OK|NG)\b", text.strip(), re.IGNORECASE)
		if match:
			return match.group(1).upper()
		return "..."
	conclusion = extract_qc_judgement(ai_msg.content)
	return {"QC_JUDGEMENT_AGENT_MSGS": [
		AIMessage(content=conclusion, name="QC_JUDGEMENT_AGENT_MSGS"
	)]} 
# ================================== Ai Message ==================================
# Name: QC_JUDGEMENT_AGENT_MSGS
#
# NG



def VISUAL_AGENT(state: State) -> State:
	"""Đưa ra tọa độ lỗi cuối cùng sau quá trình suy luận."""
	ngok = get_latest_msg(state, "QC_JUDGEMENT_AGENT_MSGS").content
	ctx_frames_metadata = get_latest_msg(state, "VISION_AGENT_MSGS").content

	bbox_per_frame = {}
	for frame in ctx_frames_metadata:
		frame_id = frame["id"]
		objects = []
		for obj in frame["metadata"]:
			objects.append({
				"label": obj["label"],
				"bbox": [obj["bbox"]["x1"], obj["bbox"]["y1"], obj["bbox"]["x2"], obj["bbox"]["y2"]]
			})
		bbox_per_frame[frame_id] = objects

	result = {
		"ngok": ngok,
		"bbox": bbox_per_frame
	}
	return {"VISUAL_AGENT_MSGS": [
		AIMessage(content=str(result), name="VISUAL_AGENT_MSGS"
	)]} 
# ================================== Ai Message ==================================
# Name: VISUAL_AGENT_MSGS

# {
#   "ngok": "NG",
#   "bbox": {
#     "0": [
#       {
#         "label": "missing_hole",
#         "bbox": [
#           473,
#           263,
#           485,
#           277
#         ]
#       }, 
# ...