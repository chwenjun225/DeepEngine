# **AI Research: AutoML-MultiAgent for AOI Tasks – Project Summary**

## I. Quantifiable Goals

1. Complete a **prototype** of the reasoning and NG/OK classification system within **30 days**, targeting a **minimum accuracy of 99.5%** (as per the updated requirement).  
2. The project is currently being implemented **by one person**. Due to the system's complexity, adding more team members at this stage may not be efficient.  
3. An **additional 14 days** is required to optimize the reasoning logic and prompts for the agent, resulting in a **total estimated timeline of 45 days**.  
4. The system is designed to **automatically train and inspect** without manual intervention, providing **clear and explainable reasoning**, making it suitable for scalable and intelligent QA systems.

## II. Hardware Requirements

### 2.1. Minimum Configuration (single local inference machine)
- **CPU:** Intel i9-13900KF.  
- **RAM:** 64GB.  
- **SSD:** 1TB NVMe.  
- **HDD:** For storing logs and result data.  
- **GPU:** RTX 4090 – 24GB VRAM.

### 2.2. Model Deployment Options

#### Option 1 – High Performance (2 machines)
- **Reasoning Agent:** Deepseek-RL-14B → ~19–24GB VRAM.  
- **Vision Agent:** LLaMA-3.2-11B-Vision-Instruct → ~18–24GB VRAM.  
- Suitable for scenarios requiring parallel execution and high accuracy.

#### Option 2 – Cost-Efficient (1 machine)
- Both agents run on **a single RTX 4090 machine**, sharing the Vision Agent model.
- Total VRAM usage: **~21–24GB**, fully utilizing GPU resources while maintaining accuracy.

## III. Industrial Camera Requirements

- **Resolution:** Minimum 5MP, recommended 8MP–12MP.  
- **Lens focal length:** 16mm or 25mm depending on the PCB type.  
- **Frame rate:** ≥ 30 FPS (to avoid motion blur during movement).  
- **Interface:** USB 3.0 / GigE / CameraLink.  
- **Recommended brands:** Basler, Hikrobot, Dahua, IDS Imaging.  
- **Lighting:** Use ring LED or backlight to eliminate shadows and ensure image stability.

## IV. Manpower & Time Expect Completion

- **Manpower:** 1 person (fully familiar with the system and deployment process).  
- **Estimated total time:** 45 days.  
- Phase 1: 30 days to develop a complete prototype.  
- Phase 2: Additional 14 days to optimize reasoning logic and agent prompts.

<!-- 
# AI Research: AutoML-MultiAgent for AOI Tasks – Project Summary

## I. Mục tiêu định lượng (Quantifiable Goals)
1. Hoàn thiện **prototype** hệ thống reasoning và phân loại NG/OK trong **30 ngày**, với **độ chính xác tối thiểu 99.5%** (theo yêu cầu điều chỉnh mới).  
2. Hiện tại dự án được thực hiện **bởi một người**. Do hệ thống phức tạp, việc thêm người lúc này có thể không hiệu quả.  
3. Cần thêm **14 ngày** để tối ưu logic suy luận và prompt cho agent, **tổng thời gian dự kiến: 45 ngày**.  
4. Hệ thống hướng tới khả năng **huấn luyện và kiểm lỗi tự động**, không cần can thiệp thủ công, với **giải thích rõ ràng và dễ truy xuất**, phù hợp với các hệ thống QA thông minh và mở rộng.

## II. Yêu cầu phần cứng (Hardware Requirements)

### 2.1. Cấu hình tối thiểu (1 máy inference local)
- **CPU:** Intel i9-13900KF.  
- **RAM:** 64GB.  
- **SSD:** 1TB NVMe.  
- **HDD:** Lưu log và dữ liệu kết quả.  
- **GPU:** RTX 4090 – 24GB VRAM.

### 2.2. Phương án triển khai mô hình

#### Option 1 – Hiệu suất cao (2 máy)
- **Reasoning Agent:** Deepseek-RL-14B → ~ 19–24GB VRAM.  
- **Vision Agent:** LLaMA-3.2-11B-Vision-Instruct → ~ 18–24GB VRAM.  
- Phù hợp với yêu cầu xử lý song song, chính xác.

#### Option 2 – Tối ưu chi phí (1 máy)
- Cả hai agent cùng chạy trên **1 máy dùng RTX 4090**, chia sẻ mô hình Vision Agent.  
- Tổng VRAM sử dụng ~ **21–24GB**, tận dụng tối đa phần cứng GPU, đảm bảo kết quả đạt được chính xác.

## III. Yêu cầu về Camera công nghiệp

- **Độ phân giải:** Tối thiểu 5MP, khuyến nghị 8MP–12MP.
- **Ống kính:** 16mm hoặc 25mm tùy vào loại PCB.  
- **Tốc độ chụp:** ≥ 30 FPS (tránh mờ do chuyển động).  
- **Chuẩn kết nối:** USB 3.0 / GigE / CameraLink.  
- **Hãng khuyến nghị:** Basler, Hikrobot, Dahua, IDS Imaging.  
- **Chiếu sáng:** Đèn LED vòng hoặc backlight để loại bỏ bóng và ổn định ảnh.

## IV. Nhân sự và thời gian hoàn thành (Manpower & Time Expect Completion)

- **Nhân sự thực hiện:** 1 người (đã nắm toàn bộ hệ thống và quy trình triển khai)  
- **Tổng thời gian ước tính:** 45 ngày  
- Giai đoạn 1: 30 ngày để phát triển prototype hoàn chỉnh.
- Giai đoạn 2: 14 ngày bổ sung để tối ưu logic suy luận và prompt cho agent reasoning. -->