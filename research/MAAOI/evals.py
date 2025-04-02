import os 
import cv2 
import csv
import asyncio
from PIL import Image



from app import async_process_frames



### Config ###
DATASET_PATH = "/home/chwenjun225/projects/DeepEngine/research/MAAOI/evals"
MAX_IMAGES_PER_CLASS = 100
OUTPUT_CSV = "eval_results.csv"



def extract_frames_from_video(video_path:str, output_dir:str, interval:int=10) -> None:
	"""Trích xuất ảnh từ video."""
	os.makedirs(output_dir, exist_ok=True)
	cap = cv2.VideoCapture(video_path)
	frame_id = 0
	saved = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		if frame_id % interval == 0:
			filename = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
			cv2.imwrite(filename, frame)
			saved += 1
		frame_id += 1
	cap.release()
	print(f">>> Extracted {saved} frames to {output_dir}")



def evaluate_model(dataset_path="dataset_test", max_images=200):
	all_results = []
	async def evaluate_image(image_path, true_label):
		image = Image.open(image_path)
		processed_img, predicted, _ = await async_process_frames([image])
		return image_path, predicted, true_label

	tasks = []
	for label in ["OK", "NG"]:
		label_path = os.path.join(dataset_path, label)
		for filename in os.listdir(label_path)[:max_images]:
			image_path = os.path.join(label_path, filename)
			tasks.append(evaluate_image(image_path, label))

	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	results = loop.run_until_complete(asyncio.gather(*tasks))

	TP = FP = TN = FN = 0
	for fname, predicted, true_label in results:
		if true_label == "NG" and predicted == "NG":
			TP += 1
		elif true_label == "NG" and predicted == "OK":
			FN += 1
		elif true_label == "OK" and predicted == "OK":
			TN += 1
		elif true_label == "OK" and predicted == "NG":
			FP += 1

	total = TP + TN + FP + FN
	accuracy = (TP + TN) / total if total else 0
	precision = TP / (TP + FP) if TP + FP else 0
	recall = TP / (TP + FN) if TP + FN else 0
	f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

	print(f"\n[Evaluation Result]")
	print(f"True Positive (TP): {TP}")
	print(f"True Negative (TN): {TN}")
	print(f"False Positive (FP): {FP}")
	print(f"False Negative (FN): {FN}")
	print(f"Accuracy : {accuracy:.2%}")
	print(f"Precision: {precision:.2%}")
	print(f"Recall   : {recall:.2%}")
	print(f"F1-score : {f1:.2%}")



### Evaluation ###
async def evaluate_image(image_path: str, true_label: str):
	image = Image.open(image_path).convert("RGB")
	try:
		_, predicted_label, _ = await async_process_frames([image])
	except Exception as e:
		predicted_label = "ERROR"
	return {
		"filename": os.path.basename(image_path),
		"true_label": true_label,
		"predicted_label": predicted_label
	}

def compute_metrics(results):
	TP = FP = TN = FN = 0
	for r in results:
		pred, truth = r["predicted_label"], r["true_label"]
		if pred == "NG" and truth == "NG": TP += 1
		elif pred == "NG" and truth == "OK": FP += 1
		elif pred == "OK" and truth == "OK": TN += 1
		elif pred == "OK" and truth == "NG": FN += 1

	total = TP + TN + FP + FN
	accuracy = (TP + TN) / total if total else 0
	precision = TP / (TP + FP) if (TP + FP) else 0
	recall = TP / (TP + FN) if (TP + FN) else 0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

	return {
		"TP": TP, "TN": TN, "FP": FP, "FN": FN,
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1_score": f1
	}

def run_evaluation():
	all_tasks = []
	for label in ["OK", "NG"]:
		class_path = os.path.join(DATASET_PATH, label)
		image_files = sorted(os.listdir(class_path))[:MAX_IMAGES_PER_CLASS]
		for fname in image_files:
			fpath = os.path.join(class_path, fname)
			all_tasks.append(evaluate_image(fpath, label))

	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	results = loop.run_until_complete(asyncio.gather(*tqdm(all_tasks)))

	metrics = compute_metrics(results)

	### Save CSV
	with open(OUTPUT_CSV, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["filename", "true_label", "predicted_label"])
		writer.writeheader()
		writer.writerows(results)

	### Print summary
	print("\n📊 Evaluation Summary:")
	for key, value in metrics.items():
		if isinstance(value, float):
			print(f"{key}: {value:.2%}")
		else:
			print(f"{key}: {value}")

if __name__ == "__main__":
	run_evaluation()
