1. Lên kế hoạch và ý tưởng thực hiện dự án AI-Agent.
2. Tải mô hình llama3.1 về máy và thực hiện load nó với pytorch.
3. Everything Should Quantization & Pruning, Distillation (ESQP) 
4. Sử dụng guide này để làm AI chấm điểm báo cáo: https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/build_with_Llama_3_2.ipynb
5. Sử dụng Ai-Agent để tự động hỗ trợ vẽ layout nhà xưởng, văn phòng
6. Sử dụng README.md này để tạo ứng dụng demo chứng minh AI-Agent hoạt động trên điện thoại (Thiết bị biên, IOT): https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/inference/mobile_inference/android_inference
7. Sử dụng README.md này để tạo demo chứng minh AI-Agent hoạt động trên máy tính: https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/inference/local_inference
8. Sẽ thế nào nếu như ta áp dụng kiểu kiến trúc Mixture-of-Expert lên các thiết bị biên
9. Tuần sau có thể sử dụng báo cáo để trình bày `Define the Agent’s Purpose and Scope`

Dưới đây là bản dịch nội dung trong hình:

| Tên danh mục                         | Mô tả                                                                                             | Ứng dụng                              | Chi tiết thêm                                                                                   |
|--------------------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------|------------------------------------------------------------------------------------------------|
| Cắt tỉa tham số và lượng tử hóa       | Loại bỏ các tham số dư thừa không ảnh hưởng đáng kể đến hiệu suất.                                | Lớp tích chập và lớp kết nối đầy đủ   | Ổn định trong nhiều thiết lập, đạt hiệu suất tốt, hỗ trợ huấn luyện từ đầu hoặc từ mô hình đã huấn luyện. |
| Phân tích ma trận hạng thấp           | Sử dụng phân rã ma trận/tensor để ước tính các tham số quan trọng.                               | Lớp tích chập và lớp kết nối đầy đủ   | Quy trình chuẩn hóa, dễ thực hiện, hỗ trợ huấn luyện từ đầu hoặc từ mô hình đã huấn luyện.      |
| Bộ lọc tích chập chuyển đổi/nén       | Thiết kế các bộ lọc tích chập cấu trúc đặc biệt để tiết kiệm tham số.                            | Chỉ áp dụng cho lớp tích chập         | Thuật toán phụ thuộc vào ứng dụng, thường đạt hiệu suất tốt, chỉ hỗ trợ huấn luyện từ đầu.      |
| Truyền nội dung tri thức (Knowledge distillation) | Huấn luyện một mạng nơ-ron nhỏ gọn với tri thức được truyền từ một mô hình lớn hơn.             | Lớp tích chập và lớp kết nối đầy đủ   | Hiệu suất của mô hình nhạy cảm với ứng dụng và cấu trúc mạng, chỉ hỗ trợ huấn luyện từ đầu.     |
