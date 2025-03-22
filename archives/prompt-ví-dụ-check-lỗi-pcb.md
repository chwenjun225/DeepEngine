Để làm rõ yêu cầu và giúp viết lại prompt mạch lạc hơn, chúng ta sẽ phân tích từng bước trong quá trình mô phỏng và xử lý yêu cầu.

### Phân tích yêu cầu:
1. **Lỗi cần kiểm tra trên bản mạch PCB**: 
   - Lỗi thiếu thiếc
   - Lỗi thừa thiếc
   - Lỗi xước bản mạch
   - Lỗi thừa vết dây dẫn trên bản mạch
   
2. **Sử dụng dữ liệu hình ảnh**: 
   - Cần ảnh của bản mạch PCB để tạo dữ liệu training cho mô hình vision.

3. **Sử dụng API GenerateSampleImageData**:
   - API này sẽ tạo dữ liệu training cho mô hình AI-vision dựa trên ảnh đầu vào.

4. **Huấn luyện mô hình vision**:
   - Sau khi có dữ liệu training, mô hình vision sẽ được huấn luyện để nhận diện các lỗi trên bản mạch PCB.

5. **Kiểm tra lỗi AOI (Automated Optical Inspection)**:
   - Sau khi huấn luyện xong, mô hình AI-vision sẽ được sử dụng để kiểm tra các lỗi trên bản mạch PCB.
   
6. **Xử lý dữ liệu từ AI-vision**:
   - Sau khi mô hình kiểm tra các lỗi, kết quả sẽ được trả về dưới dạng text. Bạn sẽ là người xử lý và phân tích dữ liệu này.

### Mục tiêu là làm cho prompt rõ ràng và dễ hiểu hơn, đặc biệt là về các bước yêu cầu và mục đích cuối cùng của tác vụ.

### Viết lại prompt:
**"Tôi cần một mô hình giúp tôi kiểm tra các lỗi sản phẩm trên bản mạch PCB. Các lỗi cần kiểm tra bao gồm: 1. Lỗi thiếu thiếc, 2. Lỗi thừa thiếc, 3. Lỗi xước bản mạch, 4. Lỗi thừa vết dây dẫn trên bản mạch. Dưới đây là đường dẫn tới ảnh dữ liệu của bản mạch. Bạn hãy sử dụng ảnh này để gọi đến API `GenerateSampleImageData` nhằm tạo dữ liệu huấn luyện cho mô hình AI-vision. Sau khi huấn luyện xong mô hình vision, bạn sử dụng mô hình này để kiểm tra các lỗi AOI trên bản mạch. Cuối cùng, bạn sẽ xử lý kết quả đầu ra dưới dạng text mà AI-vision trả về."**

### Giải thích cho các phần đã thay đổi:
1. **Làm rõ các lỗi cần kiểm tra**: Danh sách các lỗi được liệt kê rõ ràng và có cấu trúc dễ hiểu.
2. **Mô tả quy trình rõ ràng**: Các bước cần thực hiện (tạo dữ liệu, huấn luyện mô hình, kiểm tra lỗi, xử lý kết quả) được chia thành các câu đơn giản và dễ hiểu.
3. **API và xử lý kết quả**: Việc sử dụng API `GenerateSampleImageData` và sau đó huấn luyện mô hình AI-vision để kiểm tra lỗi được làm rõ, và yêu cầu xử lý kết quả cũng được nhấn mạnh ở cuối.

Hy vọng rằng bản viết lại này giúp bạn làm rõ các bước và yêu cầu của hệ thống multi-agent!
