# AI-Agent for Prognostic and Health Management (PHM) 

## I. Hiểu rõ chức năng cần triển khai

### 1. Agent Type 
Prognostic and Health Management (PHM) 

### 2. Performance Measures 

1. Phát hiện và xử lý bất thường
- Phát hiện lỗi từ dữ liệu cảm biến (vibrations, nhiệt độ, âm thanh, hình ảnh).
- Đưa ra giải pháp khắc phục nhanh chóng và phù hợp với điều kiện vận hành.

2. Tăng cường an toàn và hiệu suất
- Đảm bảo các hệ thống máy móc hoạt động ổn định và an toàn.
- Giảm thiểu thời gian chết (downtime) và chi phí bảo trì.

3. Tự động hóa quy trình bảo trì
- Lên lịch bảo trì dự phòng dựa trên dữ liệu cảm biến.
- Đưa ra cảnh báo cho nhân viên kỹ thuật nếu cần sự can thiệp.

### 3. Enviroment

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

### 4. Actuators 

1. Công cụ điều khiển băng chuyền hoặc máy móc (PLC)
- Thay đổi tốc độ băng chuyền, điều chỉnh áp xuất hoặc nhiệt độ. 
- ... 

2. Giao diện người dùng (HMI)
- Gửi thông báo hoặc báo cáo cho nhân viên kỹ thuật.
- ... 

3. Ngôn ngữ lập trình (Python, JavaScript, v.v.)
- Chạy các đoạn mã kiểm tra và điều chỉnh trạng thái hệ thống.

### 5. Sensors 

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

### 6. Mô hình sử dụng
- Llama-3.2-1B của Meta-AI: Khả năng phân tích dữ liệu và call tool (generate code) để kiểm soát và điều chỉnh hoạt động của hệ thống.

**Mô tả cách hoạt động**:

`<|system|>`:
- Bạn là một hệ thống trí tuệ nhân tạo chuyên về giám sát và bảo trì máy móc. Nhiệm vụ của bạn là nhận tín hiệu từ cảm biến, phát hiện bất thường và đưa ra giải pháp xử lý nhanh chóng.

`<|sensors|>`: 
- *Dữ liệu cảm biến giả định*:
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

## II. Đánh giá tính sáng tạo 

### 1. Nghiên cứu bằng sáng chế hiện có 

**Bảng 1**: Nghiên cứu, tổng hợp các bằng sáng chế liên quan đến PHM, các tài liệu dưới đây được tham khảo từ `Google Patent`, và từ Văn Phòng Cục Sở Hữu Trí Tuệ tại Mỹ, Trung Quốc, Đài Loan và Việt Nam. 

| STT | Patent No       | Date of Patent | Title | Comments |
|---- |-----------------|----------------|-------|----------|
| 01 | CN116680554B | 2024-04-19 | Hệ Thống Đo Lường và Cải Tiến Luồng Bệnh Nhân trong Các Hệ Thống Y Tế | Phương pháp truyền thống, `Chưa tích hợp LLMs, kỹ thuật nén mô hình, kết hợp IoT phần cứng` |
| 02 | CN114486262B | 2023-03-21 | Phương pháp dự đoán tuổi thọ còn lại của vòng bi dựa trên CNN-AT-LSTM | Đã tích hợp phương pháp học sâu CNN, Attention. `Chưa có kỹ thuật nén mô hình, kết hợp IoT phần cứng và Actuators đầu ra` |
| 03 | CN115291589B | 2024-10-01 | Phương pháp và hệ thống chẩn đoán lỗi ma trận D của thiết bị đo lường và kiểm soát mặt đất | Dựa trên phương pháp toán học truyền thống, không phải giải pháp AI | 
| 04 | US11609836B2 | 2023-03-21 | Phương pháp vận hành và thiết bị vận hành cho mô hình phát hiện và phân loại lỗi | Không áp dụng LLM, AI |
| 05 | TWI794907B   | 2023-03-01 | Prognostic and Health Management và phương pháp liên quan | Chưa kết hợp phần cứng actutors như PLC, chưa tích hợp AI mạnh mẽ |
| 06 | CN113344295B | 2023-02-14 | Phương pháp, hệ thống và môi trường dự đoán tuổi thọ còn lại của thiết bị dựa trên dữ liệu lớn công nghiệp | Phương pháp này sử dụng mô hình học máy, và các thuật toán dự đoán. Chưa kết hợp LLM, kỹ thuật cắt tỉa chứng cất trọng số mô hình |
| 07 | CN114925723B | 2023-04-07 | Phương pháp dự đoán tuổi thọ còn lại của vòng bi lăn bằng cách sử dụng encoder và decoder | Sử dụng CNN để trích xuất đặc trưng rồi đưa vào mạng encode-decode, thực hiện hồi quy tuyến tính trên chỉ số sức khoẻ thu thập được. `Chưa kết hợp phần cứng IOT, LLM, và các phương pháp tối ưu hoá mô hình` |
| 08 | TWI849573B   | 2024-07-21 | Hệ thống và phương pháp giám sát thông minh thiết bị máy móc | Phương pháp này đề xuất sử dụng 3 mô đun phần cứng lần lượt là `1. Cảm biến thu thập data; 2. Tính toán xử lý dữ liệu thô trên thiết bị biên; 3. Tính toán xử lý GPU server AI; 4. Truyền dữ liệu và hiển thị`, phương pháp này sử dụng mạng neural network. Tuy nhiên, đây là những điểm chưa có ở phương pháp này `1. Chưa có kết hợp Actuators (bộ phận thực hiện lệnh) là PLC; 2. Chưa áp dụng phương pháp nén mô hình AI; chưa áp dụng kiến trúc mô hình MoE` |
| 09 | CN115017826B | 2023-08-25 | Phương pháp dự đoán tuổi thọ còn lại của thiết bị sử dụng phương pháp học sâu xử lý trực tiếp dữ liệu thô từ cảm biến | Bằng sáng chế này đề xuất một giải pháp theo dõi quản lý sức khoẻ thiết bị bằng cách `dùng phương pháp học sâu, xử lý trực tiếp dữ liệu thô thu được từ cảm biến về`. `Tuy nhiên chưa kết hợp đầu ra Actutors và các thuật toán nén, chưng cất mô hình` |
| 10 | CN114169548B | 2023-04-07 | Hệ thống và phương pháp PHM quản lý và bảo trì cầu đường dựa trên BIM | Phương pháp sử dụng BIM trong xây dựng, `cùng với việc thu thập dữ liệu từ cảm biến đưa vào mạng neural để xử lý`. `Chưa áp dụng LLM, kỹ thuật nén mô hình, đầu ra tác động vật lý Actuators` |
| 11 | CN112785183B | 2022-04-05 | Khung hệ thống quản lý sức khỏe cho đội xe kiểu hợp nhất phân tầng | Ứng dụng cho theo dõi phương tiện giao thông cỡ lớn (đội xe tải, xe container, ...). Có tích hợp AI, và phần cứng nhưng chưa áp dụng kỹ thuật nén mô hình |
| 12 | CN113420849B | 2021-11-30 | Phương pháp, thiết bị và phương tiện huấn luyện mô hình gia tăng trực tuyến dựa trên học chủ động | `Đây là một bằng sáng chế phương pháp phần mềm nhằm hạn chế nhu cầu gán nhãn dữ liệu thủ công` |
| 13 | CN113076625B | 2022-03-29 | Hệ thống quản lý sức khỏe máy phát điện diesel tàu chiến và phương pháp hoạt động | Ý tưởng này sử dụng kết hợp các mô hình học sâu như CNN, RNN, DBN cùng với kết hợp các phần cứng như cảm biến. `Hiện tại chưa áp dụng LLM, và chưa áp dụng các thuật toán cắt tỉa trọng số LLM`|
| 14 | CN114266278B | 2024-02-20 | Phương pháp dự đoán thời gian sử dụng còn lại của thiết bị dựa trên mạng lưới chú ý kép | Bằng sáng chế này thể hiện một mô hình toán học attention để xử lý dữ liệu cảm biến từ nhiều nguồn, cùng với môdun BiLSTM để thể hiện mối quan hệ mô hình hoá giữa dữ liệu thu được và thời gian xử lý. `Bằng sáng chế tập trung vào một mô hình toán học attention network để xác định thời gian sử dụng còn lại của sản phẩm`. |

### 2. Phân tích điểm mới 
- Xác định các yếu tố sáng tạo (Ví dụ: cách kết hợp LLM và PHM hoặc tối ưu hoá quy trình dự đoán). 
- Lưu ý rằng phần mềm thuần tuý thường khó đạt được bằng sáng chế, nên cần kết hợp với phần cứng để tăng tính khả thi. 

## III. Kết hợp phần cứng
Xây dựng giải pháp tích hợp:
Đề xuất cách kết hợp phần mềm (PHM + LLM) với phần cứng (ví dụ: cảm biến, thiết bị IoT) để tăng khả năng đạt sáng chế.
Xác định lợi ích mà phần cứng mang lại cho chức năng (ví dụ: thu thập dữ liệu chính xác hơn, tăng tốc độ xử lý).

## IV. Đánh giá tính khả thi
Phân tích rủi ro:
Xem xét những thách thức khi kết hợp phần mềm và phần cứng (chi phí, khả năng triển khai, phức tạp kỹ thuật).
Lập báo cáo:
Tổng hợp đánh giá của bạn thành báo cáo chi tiết, bao gồm:
Tóm tắt chức năng.
Tính sáng tạo của chức năng.
Tính khả thi của việc đạt bằng sáng chế.
Đề xuất cải tiến để tăng khả năng thành công.

## V. Mô phỏng kịch bản sử dụng
- User hỏi đáp về dữ liệu lịch sử trong database. 
- 