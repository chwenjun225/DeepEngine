Đúng rồi bro, đó là một ý tưởng rất hay — dùng bounding box `(x1, y1, x2, y2)` để **crop ảnh**, rồi đưa phần đó vào **LLaMA3.2-Vision-Instruct** để mô tả chi tiết khu vực bị lỗi. Tớ sẽ hướng dẫn cậu **từng bước dễ hiểu** nhé:

---

### ✅ 1. Cắt ảnh từ bounding box

Giả sử cậu có ảnh dạng `PIL.Image` và bbox như `(122, 49, 126, 54)`:

```python
def crop_from_bbox(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
	"""Crop ảnh theo bounding box (x1, y1, x2, y2)."""
	x1, y1, x2, y2 = bbox
	return image.crop((x1, y1, x2, y2))
```

---

### ✅ 2. Resize hoặc padding (tuỳ chọn)

LLaMA Vision có thể yêu cầu ảnh input có kích thước đồng đều, ví dụ 224x224. Cậu có thể resize hoặc padding:

```python
def resize_or_pad(image: Image.Image, size=(224, 224)) -> Image.Image:
	return image.resize(size)
```

---

### ✅ 3. Encode ảnh thành base64 (cho prompt vision-instruct)

```python
def image_to_base64(pil_img: Image.Image) -> str:
	from io import BytesIO
	import base64
	buf = BytesIO()
	pil_img.save(buf, format="PNG")
	return base64.b64encode(buf.getvalue()).decode("utf-8")
```

---

### ✅ 4. Tạo prompt và gửi vào LLaMA

```python
def describe_defect_from_bbox(full_image: Image.Image, bbox: tuple[int, int, int, int]) -> str:
	cropped = crop_from_bbox(full_image, bbox)
	resized = resize_or_pad(cropped, size=(224, 224))
	base64_img = image_to_base64(resized)

	prompt = f"""You are a visual inspector. Please describe the defect in the following image region:
<image>{base64_img}</image>
"""
	return VISION_INSTRUCT_LLM.invoke(prompt).content
```

---

### 🧠 Gợi ý:

- Nếu YOLO detect nhiều `bbox`, cậu có thể **loop qua từng cái** và mô tả từng lỗi riêng biệt.
- Cũng có thể kết hợp lại thành `"defect 1: ...\ndefect 2: ..."`.

---

Muốn tớ giúp viết full ví dụ `detect -> crop -> describe` luôn không?