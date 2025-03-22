# My answer

I. Nếu sử dụng 1 LLM:
Kết quả sẽ phụ thuộc vào hiểu biết tổng quát của mô hình, kết quả dựa trên dữ liệu đã training.

Ưu điểm: Hệ thống đơn giản, dễ phát triển.
Nhược điểm: 
- Cần tiêu tốn rất nhiều dữ liệu để training kiến thức cho mô hình.
- Không thông minh, không có khả năng suy nghĩ giải quyết vấn đề.

---

II. Nếu sử dụng Multi-Agent:
Kết quả sẽ phụ thuộc vào quá trình lập luận có quy củ của toàn hộ Agent trong hệ thống (RAG, PLANNING,...)

Ưu điểm: 
- Hệ thống có khả năng tự suy nghĩ và đưa ra hành động theo một quy trình khoa học đã chuẩn hóa (Chain-of-Thougt, Reasoning to Action, Reflection, Inner Monoluge,...).
- Khả năng lập luận, xác minh, quyết định, giải quyết vấn đề tốt hơn so với chỉ sử dụng 1 LLM.

Nhược điểm: 
- Hệ thống phức tạp, cần xây dựng nhiều thành phần phối hợp, dẫn đến độ trễ cao.
- Yêu cầu liên tục nghiên cứu Công nghệ Reasoning để khiến mô hình thông minh hơn. 


Nguyên lý phát hiện lỗi:
Mô hình Yolo có thể detect ảnh sai dù có huấn luyện bao nhiêu ảnh đi chăng nữa. Làm sao để khắc phục ???

=> Giải pháp: Cắt các điểm ảnh yolo detect được để đưa vào vision-agent để đưa ra mô tả 
(Lưu ý vision-agent chỉ đưa ra mô tả về các điểm ảnh mà yolo detect được chứ không đưa ra kết luận OK hay NG.)

2. Cuối cùng reasoning-agent sẽ lấy các mô tả từ vision-agent để lý luận xác minh lần cuối xem sản phẩm có thực sự bị NG hoặc OK không. 
