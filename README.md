# RAYMOND - Tree-Based Reactive Agent System for Research

**RAYMOND** là một hệ thống multi-agent framework được thiết kế nhằm mục tiêu nghiên cứu và phát triển các thuật toán phản ứng (reactive algorithms) trong môi trường sản xuất tự động. Framework này tập trung vào tổ chức agent dạng cây (tree-based) nhằm cải thiện khả năng phối hợp và phản ứng thời gian thực giữa các agent trong hệ thống AI công nghiệp.

## 🚀 Mục tiêu

- Xây dựng mô hình agent phản ứng theo cấu trúc cây cho các tác vụ kiểm tra trực quan, lập kế hoạch hoặc điều phối robot.
- Kết hợp các mô hình học sâu như YOLO, Vision-Language Model (VLM) để cung cấp khả năng reasoning, phát hiện lỗi và mô tả lỗi.
- Hướng tới triển khai hệ thống phân tán, có khả năng mở rộng, dễ tích hợp vào môi trường công nghiệp.


## 🗂️ Cấu trúc thư mục

```
research/
├── MAAOI/             # Multi-agent AOI subsystem (các thí nghiệm riêng)
└── RAYMOND/           # Core framework
    ├── app.py         # Entry point khởi tạo và chạy các agents
    ├── const.py       # Các hằng số cấu hình
    ├── db.py          # Module lưu trữ/log dữ liệu (giả lập DB hoặc logging)
    └── README.md      # Hướng dẫn riêng cho thư mục RAYMOND
docs/                  # Tài liệu chi tiết (nếu có)
notebooks/             # Jupyter notebooks cho demo, thí nghiệm
from3rdparty/          # Thư viện/phần mở rộng bên thứ ba
.vscode/               # Cấu hình VS Code
.gitignore             # File bỏ qua khi push Git
download.py            # Module tải dữ liệu mẫu (nếu cần)
LICENSE                # Giấy phép sử dụng mã nguồn
README.md              # README chính của dự án
tranvantuan.sh         # Script hỗ trợ khởi động / thiết lập môi trường
```
## 🛠️ Hướng dẫn chạy

1. **Yêu cầu môi trường:**

   - Python >= 3.9
   - pip, venv hoặc Conda
   - Cài đặt thư viện yêu cầu:
     ```bash
     pip install -r requirements.txt  # nếu có
     ```

2. **Chạy Agent System:**
   ```bash
   cd research/RAYMOND
   python app.py
   ```

3. **(Tuỳ chọn)** Cấu hình thông qua `const.py`

---

## 🧪 Các tính năng chính

- **Tree-based Agent Coordination**: Phân tầng tác nhân (agent) thành cây để tối ưu quá trình phối hợp.
- **Reactive Planning**: Phản ứng nhanh với các sự kiện đầu vào theo kiểu hành vi.
- **Tích hợp Vision AI**: Hỗ trợ YOLO, Vision-Language Models.
- **Modular Design**: Dễ dàng mở rộng / plug-in các agent, module xử lý mới.

---

## 🤝 Đóng góp

1. Fork repo
2. Tạo nhánh (`git checkout -b feature/your-feature`)
3. Commit thay đổi (`git commit -am 'Add feature'`)
4. Push lên nhánh (`git push origin feature/your-feature`)
5. Mở Pull Request
