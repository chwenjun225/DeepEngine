import fire 
import os 
import glob 
import asyncio
import logging 
from datetime import datetime 
from PIL import Image 



from app import __detect_pcb_image__



async def eval_image_folder(
		ok_folder: str,
		ng_folder: str,
		batch_size: int = 10,
		max_workers: int = 4
) -> dict:
		"""
		Đánh giá hiệu suất mô hình trên tập ảnh trong thư mục OK và NG

		Args:
				ok_folder: Đường dẫn thư mục chứa ảnh OK
				ng_folder: Đường dẫn thư mục chứa ảnh NG
				batch_size: Số ảnh xử lý cùng lúc (mặc định: 10)
				max_workers: Số luồng xử lý tối đa (mặc định: 4)

		Returns:
				Dictionary chứa kết quả đánh giá chi tiết
		"""
		def get_image_paths(folder: str) -> list:
				"""Lấy danh sách các file ảnh hợp lệ trong thư mục"""
				extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
				return sorted([f for ext in extensions for f in glob.glob(os.path.join(folder, ext))])

		# 1. Khởi tạo kết quả đánh giá
		results = {
				'ok': {'total': 0, 'correct': 0, 'incorrect': []},
				'ng': {'total': 0, 'correct': 0, 'incorrect': []},
				'start_time': datetime.now().isoformat(),
				'duration': None
		}

		# 2. Lấy danh sách ảnh
		ok_images = get_image_paths(ok_folder)
		ng_images = get_image_paths(ng_folder)
		results['ok']['total'] = len(ok_images)
		results['ng']['total'] = len(ng_images)

		if not ok_images and not ng_images:
				raise ValueError("Không tìm thấy ảnh trong cả hai thư mục")

		# 3. Xử lý bất đồng bộ với Semaphore để giới hạn concurrent tasks
		semaphore = asyncio.Semaphore(max_workers)

		async def process_batch(image_paths: list, expected_status: str) -> dict:
				"""Xử lý một batch ảnh"""
				batch_results = {'correct': 0, 'incorrect': []}
				async with semaphore:
						tasks = [process_single_image(img, expected_status) for img in image_paths]
						batch_outputs = await asyncio.gather(*tasks, return_exceptions=True)
						
						for img_path, output in zip(image_paths, batch_outputs):
								if isinstance(output, Exception):
										batch_results['incorrect'].append({
												'image': img_path,
												'error': str(output),
												'predicted': 'ERROR'
										})
								elif output == expected_status:
										batch_results['correct'] += 1
								else:
										batch_results['incorrect'].append({
												'image': img_path,
												'error': None,
												'predicted': output
										})
				return batch_results

		async def process_single_image(img_path: str, expected_status: str) -> str:
				"""Xử lý một ảnh và trả về trạng thái dự đoán"""
				try:
						with Image.open(img_path) as img:
								_, product_status, _ = await __detect_pcb_image__(img.convert("RGB"))
								return product_status.upper()
				except Exception as e:
						logging.error(f"Lỗi xử lý ảnh {img_path}: {str(e)}")
						raise

		# 4. Xử lý các batch ảnh OK
		print(f"\nĐang đánh giá {len(ok_images)} ảnh OK...")
		for i in range(0, len(ok_images), batch_size):
				batch = ok_images[i:i + batch_size]
				batch_results = await process_batch(batch, "OK")
				results['ok']['correct'] += batch_results['correct']
				results['ok']['incorrect'].extend(batch_results['incorrect'])
				print(f"Đã xử lý {min(i + batch_size, len(ok_images))}/{len(ok_images)} ảnh OK")

		# 5. Xử lý các batch ảnh NG
		print(f"\nĐang đánh giá {len(ng_images)} ảnh NG...")
		for i in range(0, len(ng_images), batch_size):
				batch = ng_images[i:i + batch_size]
				batch_results = await process_batch(batch, "NG")
				results['ng']['correct'] += batch_results['correct']
				results['ng']['incorrect'].extend(batch_results['incorrect'])
				print(f"Đã xử lý {min(i + batch_size, len(ng_images))}/{len(ng_images)} ảnh NG")

		# 6. Tính toán các chỉ số tổng hợp
		results['duration'] = str(datetime.now() - datetime.fromisoformat(results['start_time']))

		# Tính accuracy
		for status in ['ok', 'ng']:
				total = results[status]['total']
				correct = results[status]['correct']
				results[status]['accuracy'] = round(correct / total * 100, 2) if total > 0 else 0.0

		# Tính tổng hợp
		total_images = results['ok']['total'] + results['ng']['total']
		total_correct = results['ok']['correct'] + results['ng']['correct']
		results['overall_accuracy'] = round(total_correct / total_images * 100, 2) if total_images > 0 else 0.0

		# 7. Xuất kết quả chi tiết
		print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
		print(f"Thư mục OK: {results['ok']['correct']}/{results['ok']['total']} "
					f"({results['ok']['accuracy']}%)")
		print(f"Thư mục NG: {results['ng']['correct']}/{results['ng']['total']} "
					f"({results['ng']['accuracy']}%)")
		print(f"Tổng thể: {total_correct}/{total_images} ({results['overall_accuracy']}%)")
		print(f"Thời gian thực thi: {results['duration']}")

		return results


def evaluate_image_folders(
		ok_folder: str = "/home/chwenjun225/projects/DeepEngine/evals/groundtruth_evaluations/OK",
		ng_folder: str = "/home/chwenjun225/projects/DeepEngine/evals/groundtruth_evaluations/NG",
		batch_size: int = 10,
		max_workers: int = 4
) -> dict:
		"""
		Giao diện đồng bộ để chạy eval_image_folder
		
		Args:
				ok_folder: Thư mục ảnh OK (mặc định: /path/to/OK)
				ng_folder: Thư mục ảnh NG (mặc định: /path/to/NG)
				batch_size: Kích thước batch (mặc định: 10)
				max_workers: Số worker tối đa (mặc định: 4)
				
		Returns:
				Dictionary kết quả đánh giá
		"""
		return asyncio.run(
				eval_image_folder(
						ok_folder=ok_folder,
						ng_folder=ng_folder,
						batch_size=batch_size,
						max_workers=max_workers
				)
		)



if __name__ == "__main__":
		fire.Fire(evaluate_image_folders)