from typing_extensions import (Annotated, TypedDict)



from langgraph.graph.message import add_messages 



class State(TypedDict):
	messages: Annotated[list[dict], add_messages]



class BBox(TypedDict):
	x1: int
	y1: int
	x2: int
	y2: int

class DetectionData(TypedDict):
	bbox: BBox
	confidence: float
	class_id: int
	label: str

SingleFrameDetections = list[DetectionData]

class FrameData(TypedDict):
	frame_id: int
	detections: SingleFrameDetections

class MultiFrameData(TypedDict):
	frames: Annotated[list[FrameData], add_messages]