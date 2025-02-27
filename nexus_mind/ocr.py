import numpy as np 
import cv2 
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image



vid_path = "/home/chwenjun225/projects/DeepEngine/nexus_mind/images/fuzetea_vid1.mp4"



ocr = PaddleOCR(use_angle_cls=True, lang='en') 



cap = cv2.VideoCapture(vid_path)
if not cap.isOpened():
	print(">>> Can not open camera")
	exit()
print(">>> Starting real-time OCR. Press 'q' to exit.")
while True:
	ret, frame = cap.read()
	if not ret:
		print(">>> Can't receive frame (stream end?). Exiting...")
		break 
	# Perform OCR on the current frame
	result = ocr.ocr(frame, cls=False)
	# Draw detected text on the frame
	for res in result:
		if res is not None:
			for line in res:
				box, (text, score) = line 
				box = np.array(box, dtype=np.int32)
				# Draw bounding box
				cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
				# Display text near the bounding box
				x, y = box[0]
				cv2.putText(frame, f"{text} ({score:.2f})", (x, y - 10), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	cv2.imshow("Research Demo AI-Agent create AI-Vision", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break 

cap.release()
# cv2.destroyAllWindows()
print(">>> OCR session ended.")