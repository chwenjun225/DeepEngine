# AI-Agent for Prognostic and Health Management (PHM) 

## Agent Type 
Prognostic and Health Management (PHM) 

## Performance Measures 

1. Phát hiện và xử lý bất thường
- Phát hiện lỗi từ dữ liệu cảm biến (vibrations, nhiệt độ, âm thanh, hình ảnh).
- Đưa ra giải pháp khắc phục nhanh chóng và phù hợp với điều kiện vận hành.

2. Tăng cường an toàn và hiệu suất
- Đảm bảo các hệ thống máy móc hoạt động ổn định và an toàn.
- Giảm thiểu thời gian chết (downtime) và chi phí bảo trì.

3. Tự động hóa quy trình bảo trì
- Lên lịch bảo trì dự phòng dựa trên dữ liệu cảm biến.
- Đưa ra cảnh báo cho nhân viên kỹ thuật nếu cần sự can thiệp.

## Enviroment

1. Thiết bị vận hành công nghiệp
- Máy móc trong dây chuyền sản xuất, băng chuyền, động cơ, v.v.
- ...

2. Dữ liệu cảm biến từ hệ thống
- Tín hiệu từ nhiệt kế, đo độ rung, camera giám sát.
- ... 

3. Nhân viên kỹ thuật và vận hành 
- Kỹ sư giám sát và thực hiện các công việc bảo trì theo gợi ý từ hệ thống. 
- ... 

4. Quy tắc và tiêu chuẩn an toàn 
- Dựa trên quy định an toàn công nghiệp và yêu cầu của nhà máy. 

## Actuators 

1. Công cụ điều khiển băng chuyền hoặc máy móc (PLC)
- Thay đổi tốc độ băng chuyền, điều chỉnh áp xuất hoặc nhiệt độ. 
- ... 

2. Giao diện người dùng (HMI)
- Gửi thông báo hoặc báo cáo cho nhân viên kỹ thuật.
- ... 

3. Ngôn ngữ lập trình (Python, JavaScript, v.v.)
- Chạy các đoạn mã kiểm tra và điều chỉnh trạng thái hệ thống.

## Sensors 

1. Cảm biến đô độ rung, nhiệt độ, và áp suất
- Giám sát tình trạng của máy móc theo thời gian thực. 
- ... 

2. Camera giám sát
- Phân tích hình ảnh để phát hiện bất thường hoặc lỗi.

3. Hệ thống IoT
- Thu thập dữ liệu từ các thiết bị kết nối trong nhà máy. 
- ... 

4. Máy chủ quản lý dữ liệu
- Lưu trữ và xử lý thông tin từ các cảm biến. 
- ... 

## Mô hình sử dụng
- Llama-3.2-1B của Meta-AI: Khả năng phân tích dữ liệu và call tool (generate code) để kiểm soát và điều chỉnh hoạt động của hệ thống.

**Mô tả cách hoạt động**:

`<|system|>`:
- Bạn là một hệ thống trí tuệ nhân tạo chuyên về giám sát và bảo trì máy móc. Nhiệm vụ của bạn là nhận tín hiệu từ cảm biến, phát hiện bất thường và đưa ra giải pháp xử lý nhanh chóng.

`<|sensors|>`: 
- Dữ liệu cảm biến giả định:
```plaintext
* Độ rung: 0.78 mm/s (cao hơn mức tiêu chuẩn: 0.5 mm/s)  
* Tốc độbăng chuyền: 1.2 m/s (cao hơn mức tiêu chuẩn: 1.0 m/s)  
* Hình ảnh: Camera phát hiện đối tượng không rõ nằm trên băng chuyền.
```

`<|assistant|>`: 
```plaintext
- Xin chào, qua phân tích dữ liệu từ cảm biến và camera, tôi nhận thấy:
* Tốc độ băng chuyền đang cao hơn mức tiêu chuẩn (1.2 m/s so với 1.0 m/s).  
* Phát hiện vật thể không xác định trên băng chuyền.  
-> Tôi sẽ tạm thời giảm tốc độ băng chuyền về mức tiêu chuẩn và gửi cảnh báo để kiểm tra vật thể trên băng chuyền.
```

`<|ipython|>`:
- Thực hiện lệnh giảm tốc độ băng chuyền:
```python
set_conveyor_speed(speed=1.0)  # Đặt tốc độ băng chuyền về mức tiêu chuẩn
send_alert(">>> Phát hiện vật thể trên băng chuyền. Yêu cầu kiểm tra ngay.")
```
