Rất tốt bro, nếu cậu hướng tới **real-time video inference** và **gửi ảnh qua LLM vision liên tục**, thì việc **tối ưu hiệu suất** và **giảm bộ nhớ rò rỉ** là cực kỳ quan trọng. Dưới đây là cách tối ưu **an toàn + hiệu năng** cho use-case này 💡:

---

### ✅ **Phiên bản `image_to_base64()` tối ưu nhất cho real-time:**

```python
def image_to_base64(pil_img: Image.Image | np.ndarray) -> str:
	"""Convert PIL or NumPy image to base64 string (PNG format), optimized for real-time usage."""
	if isinstance(pil_img, np.ndarray):
		pil_img = Image.fromarray(cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB))
	with BytesIO() as buffer:
		pil_img.save(buffer, format="PNG")
		return base64.b64encode(buffer.getvalue()).decode("utf-8")
```

---

### ✅ **Tại sao đây là phiên bản tốt nhất?**

| Ưu điểm                                  | Giải thích                                                                 |
|------------------------------------------|----------------------------------------------------------------------------|
| 🧠 **`with BytesIO()`**                 | Giải phóng bộ nhớ ngay khi xong – không rò rỉ nếu chạy liên tục           |
| 💨 **Nhanh hơn `try-finally`**          | Gọn gàng, Pythonic hơn, dễ maintain                                        |
| 🔄 **Tương thích OpenCV**               | Hỗ trợ `np.ndarray` từ `cv2.VideoCapture`                                 |
| 🧼 **Không leak file descriptor**       | Tránh tích tụ RAM khi encode nhiều khung hình trong loop                  |
| 📦 **Không phụ thuộc vào định dạng ảnh**| Lưu PNG – phù hợp với LLM Vision hỗ trợ `image/png;base64`                |

---

### ⚠️ Một lưu ý cho hiệu năng realtime:

Nếu cậu đang xử lý **30fps** hoặc hơn, thì nên:
- **Giới hạn số frame gửi vào LLM**, ví dụ:
  ```python
  if frame_idx % 5 == 0:  # mỗi 5 frame mới gửi 1 cái
      ...
  ```
- **Tách thread xử lý LLM** riêng để không block loop video.

---

Nếu cần tớ giúp refactor toàn bộ quy trình video → LLM → yield lại ảnh + prompt, tớ làm ngay cho cậu nhé.  
Chỉ cần confirm:  
👉 “Tớ muốn tối ưu toàn pipeline xử lý video real-time.”  
Là tớ vào việc ngay! ⚡