import re 
import json


from langchain_core.messages import AIMessage, SystemMessage



from state import State
from const import (
	LLM																					,
	DEFECT_REASONING_AGENT_PROMPT_MSG						,
	QUALITY_CONTROL_JUDGEMENT_AGENT_PROMPT_MSG	,
)
from utils import (get_latest_msg)



# ================================== Ai Message ==================================
# Name: VISION_AGENT
# 
# {'id': 0, 'metadata': [{'bbox': {'x1': 473, 'y1': 263, 'x2': 485, 'y2': 277}, 'confidence': 0.867, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 236, 'y1': 261, 'x2': 247, 'y2': 274}, 'confidence': 0.842, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 212, 'y1': 406, 'x2': 228, 'y2': 421}, 'confidence': 0.219, 'class_id': 3, 'label': 'short'}, {'bbox': {'x1': 160, 'y1': 486, 'x2': 170, 'y2': 493}, 'confidence': 0.185, 'class_id': 4, 'label': 'spur'}, {'bbox': {'x1': 123, 'y1': 302, 'x2': 133, 'y2': 314}, 'confidence': 0.086, 'class_id': 0, 'label': 'missing_hole'}]}
# {'id': 1, 'metadata': [{'bbox': {'x1': 472, 'y1': 265, 'x2': 484, 'y2': 279}, 'confidence': 0.822, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 236, 'y1': 262, 'x2': 247, 'y2': 276}, 'confidence': 0.795, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 122, 'y1': 304, 'x2': 134, 'y2': 317}, 'confidence': 0.523, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 212, 'y1': 410, 'x2': 228, 'y2': 426}, 'confidence': 0.211, 'class_id': 3, 'label': 'short'}, {'bbox': {'x1': 292, 'y1': 477, 'x2': 304, 'y2': 486}, 'confidence': 0.122, 'class_id': 4, 'label': 'spur'}]}
# {'id': 2, 'metadata': [{'bbox': {'x1': 236, 'y1': 263, 'x2': 247, 'y2': 277}, 'confidence': 0.837, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 122, 'y1': 303, 'x2': 135, 'y2': 317}, 'confidence': 0.813, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 472, 'y1': 267, 'x2': 484, 'y2': 281}, 'confidence': 0.686, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 424, 'y1': 388, 'x2': 436, 'y2': 408}, 'confidence': 0.406, 'class_id': 4, 'label': 'spur'}, {'bbox': {'x1': 212, 'y1': 407, 'x2': 228, 'y2': 422}, 'confidence': 0.335, 'class_id': 3, 'label': 'short'}]}
# {'id': 3, 'metadata': [{'bbox': {'x1': 235, 'y1': 263, 'x2': 246, 'y2': 277}, 'confidence': 0.831, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 122, 'y1': 303, 'x2': 135, 'y2': 317}, 'confidence': 0.793, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 472, 'y1': 268, 'x2': 483, 'y2': 280}, 'confidence': 0.785, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 462, 'y1': 418, 'x2': 476, 'y2': 433}, 'confidence': 0.37, 'class_id': 3, 'label': 'short'}, {'bbox': {'x1': 214, 'y1': 405, 'x2': 229, 'y2': 419}, 'confidence': 0.19, 'class_id': 3, 'label': 'short'}]}
# {'id': 4, 'metadata': [{'bbox': {'x1': 235, 'y1': 264, 'x2': 246, 'y2': 278}, 'confidence': 0.835, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 122, 'y1': 304, 'x2': 134, 'y2': 318}, 'confidence': 0.734, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 160, 'y1': 488, 'x2': 168, 'y2': 495}, 'confidence': 0.285, 'class_id': 4, 'label': 'spur'}, {'bbox': {'x1': 216, 'y1': 405, 'x2': 228, 'y2': 419}, 'confidence': 0.123, 'class_id': 3, 'label': 'short'}, {'bbox': {'x1': 201, 'y1': 446, 'x2': 209, 'y2': 455}, 'confidence': 0.096, 'class_id': 1, 'label': 'mouse_bite'}]}
# {'id': 5, 'metadata': [{'bbox': {'x1': 234, 'y1': 264, 'x2': 245, 'y2': 278}, 'confidence': 0.764, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 122, 'y1': 304, 'x2': 134, 'y2': 318}, 'confidence': 0.689, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 160, 'y1': 488, 'x2': 168, 'y2': 495}, 'confidence': 0.341, 'class_id': 4, 'label': 'spur'}, {'bbox': {'x1': 337, 'y1': 208, 'x2': 351, 'y2': 221}, 'confidence': 0.108, 'class_id': 4, 'label': 'spur'}, {'bbox': {'x1': 460, 'y1': 349, 'x2': 473, 'y2': 361}, 'confidence': 0.05, 'class_id': 3, 'label': 'short'}]}
# {'id': 6, 'metadata': [{'bbox': {'x1': 234, 'y1': 263, 'x2': 245, 'y2': 277}, 'confidence': 0.803, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 122, 'y1': 303, 'x2': 134, 'y2': 318}, 'confidence': 0.72, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 160, 'y1': 487, 'x2': 174, 'y2': 494}, 'confidence': 0.482, 'class_id': 4, 'label': 'spur'}, {'bbox': {'x1': 460, 'y1': 346, 'x2': 473, 'y2': 361}, 'confidence': 0.187, 'class_id': 3, 'label': 'short'}, {'bbox': {'x1': 335, 'y1': 208, 'x2': 352, 'y2': 222}, 'confidence': 0.12, 'class_id': 3, 'label': 'short'}]}
# {'id': 7, 'metadata': [{'bbox': {'x1': 121, 'y1': 303, 'x2': 134, 'y2': 318}, 'confidence': 0.736, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 233, 'y1': 263, 'x2': 245, 'y2': 277}, 'confidence': 0.725, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 215, 'y1': 407, 'x2': 229, 'y2': 422}, 'confidence': 0.483, 'class_id': 3, 'label': 'short'}, {'bbox': {'x1': 470, 'y1': 267, 'x2': 481, 'y2': 280}, 'confidence': 0.297, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 161, 'y1': 487, 'x2': 173, 'y2': 494}, 'confidence': 0.23, 'class_id': 4, 'label': 'spur'}]}
# {'id': 8, 'metadata': [{'bbox': {'x1': 470, 'y1': 267, 'x2': 481, 'y2': 280}, 'confidence': 0.825, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 233, 'y1': 264, 'x2': 245, 'y2': 278}, 'confidence': 0.764, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 121, 'y1': 304, 'x2': 133, 'y2': 318}, 'confidence': 0.587, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 161, 'y1': 487, 'x2': 169, 'y2': 494}, 'confidence': 0.388, 'class_id': 4, 'label': 'spur'}, {'bbox': {'x1': 459, 'y1': 338, 'x2': 472, 'y2': 354}, 'confidence': 0.095, 'class_id': 3, 'label': 'short'}]}
# {'id': 9, 'metadata': [{'bbox': {'x1': 469, 'y1': 267, 'x2': 481, 'y2': 280}, 'confidence': 0.878, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 233, 'y1': 264, 'x2': 244, 'y2': 278}, 'confidence': 0.734, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 122, 'y1': 304, 'x2': 133, 'y2': 318}, 'confidence': 0.542, 'class_id': 0, 'label': 'missing_hole'}, {'bbox': {'x1': 215, 'y1': 411, 'x2': 229, 'y2': 426}, 'confidence': 0.246, 'class_id': 3, 'label': 'short'}, {'bbox': {'x1': 160, 'y1': 487, 'x2': 169, 'y2': 494}, 'confidence': 0.201, 'class_id': 4, 'label': 'spur'}]}
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
# 
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



####################################################################################################################################################



def QUALITY_CONTROL_JUDGEMENT_AGENT(state: State) -> State:
	"""Dựa trên reasoning, đưa ra kết luận OK hoặc NG. Nếu không rõ, thử lại tối đa 3 lần."""
	prev_agent_msg = get_latest_msg(state, type_msgs="DEFECT_REASONING_AGENT_MSGS")
	def extract_qc_judgement(text: str) -> str:
		"""Hàm tách OK/NG bằng regex, không phân biệt hoa thường."""
		match = re.search(r"\b(OK|NG)\b", text.strip(), re.IGNORECASE)
		return match.group(1).upper() if match else ""
	conclusion = ""
	for _ in range(3):
		ai_msg = LLM.invoke([
			prev_agent_msg, 
			SystemMessage(QUALITY_CONTROL_JUDGEMENT_AGENT_PROMPT_MSG)
		])
		conclusion = extract_qc_judgement(ai_msg.content)
		if conclusion.upper() in {"OK", "NG"}: break
	if not conclusion: conclusion = "NG"
	return {
		"QUALITY_CONTROL_JUDGEMENT_AGENT_MSGS": [
			AIMessage(
				content=conclusion, 
				name="QUALITY_CONTROL_JUDGEMENT_AGENT_MSGS"
			)]}
# ================================== Ai Message ==================================
# Name: QUALITY_CONTROL_JUDGEMENT_AGENT_MSGS
#
# NG



def VISUAL_AGENT(state: State) -> State:
	"""Đưa ra tọa độ lỗi cuối cùng sau quá trình suy luận."""
	ngok = get_latest_msg(state, "QUALITY_CONTROL_JUDGEMENT_AGENT_MSGS").content
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
