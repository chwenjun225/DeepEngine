# INFO 04-Feb-2025 19:54:39 [__init__.py:183] Automatically detected platform cuda.
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  254.88    
Total input tokens:                      1024000   
Total generated tokens:                  127618    
Request throughput (req/s):              3.92      
Output token throughput (tok/s):         500.70    
Total Token throughput (tok/s):          4518.34   

---------------Time to First Token----------------
Mean TTFT (ms):                          127405.90 
Median TTFT (ms):                        126470.56 
P99 TTFT (ms):                           250975.72 

-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.43      
Median TPOT (ms):                        9.35      
P99 TPOT (ms):                           11.44     

---------------Inter-token Latency----------------
Mean ITL (ms):                           9.43      
Median ITL (ms):                         8.42      
P99 ITL (ms):                            36.52     
```
# **📌 Giải thích kết quả benchmark của `vLLM` server**
Dưới đây là **phân tích chi tiết** từng thông số trong kết quả benchmark **`vLLM`** thu được:

---

## **🔹 1️⃣ Tổng quan về benchmark**
```plaintext
# INFO 02-04 19:54:39 [__init__.py:183] Automatically detected platform cuda.
```
✅ **Hệ thống đã tự động phát hiện và sử dụng GPU (CUDA)** để tăng tốc inference. Điều này đảm bảo rằng mô hình chạy trên GPU thay vì CPU, giúp tăng hiệu suất.

---

## **🔹 2️⃣ Thống kê tổng quát**
```plaintext
============ Serving Benchmark Result ============

Successful requests:                     1000      
Benchmark duration (s):                  254.88    
Total input tokens:                      1024000   
Total generated tokens:                  127618    
Request throughput (req/s):              3.92      
Output token throughput (tok/s):         500.70    
Total Token throughput (tok/s):          4518.34   
```

📌 **Giải thích từng chỉ số:**
- ✅ **`Successful requests: 1000`** → Đã xử lý thành công **1.000 yêu cầu**.  
- ✅ **`Benchmark duration (s): 254.88`** → Tổng thời gian chạy benchmark **254.88 giây (~4.25 phút)**.  
- ✅ **`Total input tokens: 1024000`** → Tổng số **1.024.000 token đầu vào** được xử lý.  
- ✅ **`Total generated tokens: 127618`** → Tổng số **127.618 token được mô hình sinh ra**.  
- ✅ **`Request throughput (req/s): 3.92`** → Hệ thống có thể xử lý trung bình **3.92 request mỗi giây**.  
- ✅ **`Output token throughput (tok/s): 500.70`** → Hệ thống tạo ra **500.7 token đầu ra mỗi giây**.  
- ✅ **`Total Token throughput (tok/s): 4518.34`** → Tổng số **token (đầu vào + đầu ra) mà hệ thống xử lý mỗi giây là 4.518,34 token**.  

📌 **Nhận xét:**  
- **Tốc độ `500.7 token/s` là khá nhanh** khi chạy trên GPU, chứng tỏ `vLLM` hoạt động hiệu quả.  
- **Request throughput `3.92 req/s`** có thể được cải thiện nếu bạn tối ưu batch size hoặc parallelism.  

---

## **🔹 3️⃣ Thời gian tạo token đầu tiên (`TTFT`)**
```plaintext
---------------Time to First Token----------------

Mean TTFT (ms):                          127405.90 
Median TTFT (ms):                        126470.56 
P99 TTFT (ms):                           250975.72 
```
📌 **Giải thích:**  
- ✅ **`Mean TTFT (ms): 127405.90`** → Trung bình mất **127.4 giây (~2 phút 7 giây) để tạo token đầu tiên**.  
- ✅ **`Median TTFT (ms): 126470.56`** → Giá trị trung vị tương tự.  
- ✅ **`P99 TTFT (ms): 250975.72`** → Trong **1% yêu cầu chậm nhất**, **TTFT lên tới 250 giây (~4 phút 10 giây)**.  

📌 **Nhận xét:**  
- **TTFT rất cao (~2 phút/lần request)**, đây là vấn đề thường gặp khi **mô hình quá lớn hoặc chưa tối ưu batch size / caching**.  
- **Cần tối ưu TTFT bằng cách giảm batch size hoặc sử dụng chế độ prefill hiệu quả hơn**.  

---

## **🔹 4️⃣ Thời gian tạo các token tiếp theo (`TPOT`)**
```plaintext
-----Time per Output Token (excl. 1st token)------

Mean TPOT (ms):                          9.43      
Median TPOT (ms):                        9.35      
P99 TPOT (ms):                           11.44     
```
📌 **Giải thích:**  
- ✅ **`Mean TPOT (ms): 9.43`** → Trung bình **mỗi token sau token đầu tiên** mất **9.43ms để sinh ra**.  
- ✅ **`Median TPOT (ms): 9.35`** → Trung vị của TPOT cũng tương tự.  
- ✅ **`P99 TPOT (ms): 11.44`** → 1% token chậm nhất được tạo ra **mất khoảng 11.44ms**.  

📌 **Nhận xét:**  
- **TPOT thấp (~9ms/token) → Hiệu suất tạo token rất nhanh**.  
- Điều này có nghĩa là **sau khi token đầu tiên xuất hiện, phần còn lại được tạo ra với tốc độ tốt**.  

---

## **🔹 5️⃣ Độ trễ giữa các token (`Inter-token Latency`)**
```plaintext
---------------Inter-token Latency----------------

Mean ITL (ms):                           9.43      
Median ITL (ms):                         8.42      
P99 ITL (ms):                            36.52     
```
📌 **Giải thích:**  
- ✅ **`Mean ITL (ms): 9.43`** → Trung bình **mỗi token mất 9.43ms trước khi token tiếp theo được sinh ra**.  
- ✅ **`Median ITL (ms): 8.42`** → Trung vị thời gian giữa các token là **8.42ms**.  
- ✅ **`P99 ITL (ms): 36.52`** → Trong **1% trường hợp chậm nhất**, mất **36.52ms giữa các token**.  

📌 **Nhận xét:**  
- **ITL thấp (~9ms) → Tốc độ sinh token rất ổn định**.  
- **P99 hơi cao (~36ms), có thể tối ưu bằng parallelism hoặc giảm batch size**.  

---

## **🔹 6️⃣ Tóm tắt đánh giá hiệu suất**
| **Chỉ số** | **Ý nghĩa** | **Đánh giá** |
|------------|------------|--------------|
| **Request throughput: 3.92 req/s** | Số request xử lý mỗi giây | 🚀 Tương đối tốt |
| **Output token throughput: 500.7 tok/s** | Số token sinh ra mỗi giây | 🚀 Rất tốt |
| **Total token throughput: 4518.34 tok/s** | Tổng số token xử lý (in + out) | 🚀 Tốt |
| **Mean TTFT: 127.4 giây** | Thời gian tạo token đầu tiên | ❌ Cần tối ưu |
| **Mean TPOT: 9.43ms** | Thời gian tạo từng token tiếp theo | ✅ Ổn định |
| **Mean ITL: 9.43ms** | Độ trễ giữa các token | ✅ Tốt |

---

## **🔹 7️⃣ Cách cải thiện hiệu suất**
📌 **Tối ưu thời gian tạo token đầu tiên (TTFT)**  
- **Dùng chế độ batching tốt hơn**:  
  ```bash
  python -m vllm.entrypoints.api_server --model your_model --max-batch-size 8
  ```
- **Dùng caching** để giảm thời gian load model.  

📌 **Cải thiện tốc độ inference**  
- **Tăng parallelism (nếu GPU hỗ trợ)**  
  ```bash
  python -m vllm.entrypoints.api_server --model your_model --tensor-parallel-size 2
  ```
- **Chạy ở chế độ `bfloat16` thay vì `float32` để giảm tải GPU**.  

📌 **Kiểm tra dung lượng VRAM**  
```bash
nvidia-smi
```
Nếu VRAM bị đầy, hãy thử **quantization hoặc giảm batch size**.

---

### **🎯 Kết luận**
✅ **`vLLM` xử lý tốt với tốc độ `500 token/s`**.  
✅ **Độ trễ giữa các token (`ITL 9ms`) rất tốt, giúp mô hình phản hồi mượt mà**.  
❌ **Thời gian tạo token đầu tiên (`TTFT ~127s`) quá chậm, cần tối ưu batch size hoặc caching**.  

🚀 **👉 Bạn có muốn tôi hướng dẫn chi tiết cách tối ưu TTFT và tăng tốc inference trên `vLLM` không?** 🎯
