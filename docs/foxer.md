# Foxer 

## Agent Type 
Office Assistance 

## Performance Measures 
1. An toàn 
2. Hợp pháp 
3. Tự động hoá các công việc văn phòng có tính lặp lại như: 
- Xử lý chấm công, bất thường, tăng ca
- Thu thập và chấm điểm báo cáo
- Nhắc nhở cuộc họp 
- Giải đáp thắc mắc cho người dùng 
- Tự động sắp xếp ngày nghỉ 
- ... 
4. Bảo vệ hệ thống thông tin của công ty khỏi các email giả mạo 

## Enviroment
1. Cấp trên, nhân viên văn phòng, kỹ sư, công nhân
2. Tâm trạng phù thuộc vào ngữ cảnh người sử dụng 
3. Luật lệ công ty 
4. Quyền lợi của người dùng và công ty 

## Actuators 
1. Chrome 
2. Selenium 
4. Ngôn ngữ lập trình (Python,... )

## Sensors 
1. Máy tính văn phòng 
2. Người dùng cung cấp tài liệu, câu hỏi, văn bản,... 
3. Cảm biến thời gian 
4. Bàn phím, microphone,...
5. Máy chủ, hệ thống công ty 

> Với các yêu cầu trên ta sử dụng mô hình Llama-3.1-8b của Meta-AI, có khả năng generate code điều khiển thao tác phần mềm. 

`<|system|>`: Bạn là một trợ lý văn phòng, giúp giải quyết các tác vụ văn phòng như chấm công, xử lý bất thường, nhắc nhở báo cáo,...

`<|user|>`: Xin chào, mã nhân viên của tôi là V123456, tôi gặp bất thường vào ngày 13 tháng 1 năm 2025, làm ơn giúp tôi giải quyết. 

`<|ipython|>`: Gọi API lấy dữ liệu bất thường theo mã nhân viên -> Hiển thị dữ liệu bất thường.

`<|assistant|>`: Giải thích dữ liệu bất thường -> Xử lý bất thường. 

`
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024
You are a helpful assistant<|eot_id|>

<|start_header_id|>user<|end_header_id|>
What is the capital of France?<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
lamvlaml
`