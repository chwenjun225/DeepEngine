# AI-Research: AutoML-MultiAgent for AOI tasks - Hardware requires

## Chinese Version (简体中文)

### **一、本地推理的最低硬件配置：**
- **CPU：** Intel i9-13900KF  
- **内存：** 64GB  
- **硬盘：** 1TB NVMe SSD  
- **HDD：** 用于保存日志与结果数据  
- **GPU：** RTX 4090 – 24GB 显存

---

### **二、模型部署方案：**

#### **方案一：高性能配置（使用 Deepseek-14B 作为推理引擎）**
- 需要 **两台独立电脑**：
  - 一台运行 **Reasoning Agent（Deepseek-RL-14B）** → 占用显存约 19–24GB  
  - 一台运行 **Vision Agent（LLaMA-3.2-11B-Vision-Instruct）** → 占用显存约 18–24GB  
- 适合追求高准确率、快速响应及大规模部署的场景。

#### **方案二：节省成本配置（单机运行）**
- 一台配备 **RTX 4090** 的主机即可同时运行 Reasoning 和 Vision Agent（LLaMA-3.2-11B） → 占用显存约 21–24GB  
- 在优化良好的情况下，依然可保证输出质量。  
- 充分利用 RTX 4090 的 GPU 资源（预计显存占用 20–24GB）。

---

### **三、工业相机配置建议：**
- **分辨率：** 至少 5MP，推荐 8MP–12MP  
- **镜头焦距：** 16mm 或 25mm（根据 PCB 类型选择）  
- **帧率：** ≥ 30 FPS，避免在流水线中出现运动模糊  
- **接口：** USB 3.0 / GigE / CameraLink（用于自动化 AOI 系统）  
- **推荐品牌：** Basler、海康威视（Hikrobot）、大华、IDS Imaging  
- **补光建议：** 配置 **环形 LED 灯或背光灯**，减少阴影，提升图像稳定性

---

### Pinyin (for pronunciation aid)

Yī, běndì tuīlǐ de zuìdī yìngjiàn pèizhì:  
- CPU: Intel i9-13900KF  
- Nèicún: 64GB  
- Yìngpán: 1TB NVMe SSD  
- HDD: yòng yú bǎocún rìzhì hé jiéguǒ  
- GPU: RTX 4090 – 24GB xiǎncún  

Fāng'àn yī: Gāo xìngnéng (Deepseek-14B)  
- Xūyào liǎng tái diànnǎo  
- Yī tái yùnxíng Reasoning Agent → 19–24GB xiǎncún  
- Yī tái yùnxíng Vision Agent → 18–24GB xiǎncún  

Fāng'àn èr: Jiéshěng chéngběn  
- Yī tái diànnǎo yùnxíng liǎng gè agent → 21–24GB xiǎncún  

Gōngyè xiàngjī:  
- Fēnbiànlǜ: 5–12MP  
- Jiàojù: 16mm huò 25mm  
- Zhēntǐ lǜ: ≥ 30 FPS  
- Jiēkǒu: USB 3.0 / GigE / CameraLink  
- Pǐnpái: Basler, Hikrobot, Dahua, IDS  
- Zhàomíng: Huánxíng LED huò bèiguāng

---

## **English Version**

### **I. Minimum System Requirements (for local inference):**
- **CPU:** Intel i9-13900KF  
- **RAM:** 64GB  
- **SSD:** 1TB NVMe  
- **HDD:** For storing logs and result data  
- **GPU:** RTX 4090 – 24GB VRAM

---

### **II. Model Deployment Options:**

#### **Option 1 – High Performance (using Deepseek-14B for reasoning):**
- Requires **2 separate machines**:
  - 1 for **Reasoning Agent: Deepseek-RL-14B** → ~19–24GB VRAM  
  - 1 for **Vision Agent: LLaMA-3.2-11B-Vision-Instruct** → ~18–24GB VRAM  
- Best suited for parallel execution, high accuracy, and production-scale deployment.

#### **Option 2 – Cost-efficient (single-machine setup):**
- A single **RTX 4090 machine** can host both the Reasoning and Vision Agents (running LLaMA-3.2-11B-Vision-Instruct) → ~21–24GB VRAM  
- Maintains output quality with proper optimization.  
- Maximizes GPU usage of the RTX 4090 (estimated usage: 20–24GB VRAM).

---

### **III. Industrial Camera Requirements:**
- **Resolution:** Minimum 5MP, recommended 8MP–12MP  
- **Lens focal length:** 16mm or 25mm depending on PCB type  
- **Frame rate:** ≥ 30 FPS to avoid motion blur on conveyor systems  
- **Interface:** USB 3.0 / GigE / CameraLink (for automated AOI setups)  
- **Recommended brands:** Basler, Hikrobot, Dahua, IDS Imaging  
- **Lighting:** Use **ring LED or backlight** to eliminate shadows and ensure image stability

---

## **Vietnam Version**
### **I. Cấu hình máy tính tối thiểu (cho hệ thống inference local):**
- CPU: Intel i9-13900KF  
- RAM: 64GB  
- SSD: 1TB NVMe  
- HDD: Dùng lưu log và dữ liệu kết quả
- GPU: **RTX 4090 – 24GB VRAM**

---

### **II. Các phương án triển khai mô hình:**

#### **Option 1 – Hiệu suất cao (sử dụng Deepseek-14B để reasoning):**
- Cần **2 máy tính độc lập**, cụ thể:
  - 1 máy host **Reasoning Agent: Deepseek-RL-14B** → ~19–24GB VRAM  
  - 1 máy host **Vision Agent: LLaMA-3.2-11B-Vision-Instruct** → ~18–24GB VRAM  
- Phù hợp nếu muốn xử lý song song với độ chính xác và tốc độ cao.

#### **Option 2 – Tối ưu chi phí phần cứng:**
- Chỉ cần **1 máy RTX 4090** có thể host cả Reasoning Agent và Vision Agent (chung LLaMA-3.2-11B-Vision-Instruct). → ~21–24GB VRAM  
- Đảm bảo chất lượng đầu ra được tối ưu tốt.
- Tận dụng tối đa phần cứng GPU-RTX4090 (20/24GB).

---

### **III. Yêu cầu về Camera công nghiệp:**

- **Loại:** Camera công nghiệp độ phân giải cao (≥ 5MP, lý tưởng 8MP–12MP).  
- **Tiêu cự ống kính:** Tuỳ loại PCB, nên chọn lens 16mm hoặc 25mm để chụp chi tiết vùng lỗi.  
- **Tốc độ chụp:** ≥ 30 FPS (đảm bảo không bị mờ khi di chuyển băng chuyền).  
- **Chuẩn kết nối:** USB 3.0 / GigE (Ethernet) / CameraLink (nếu tích hợp hệ thống AOI tự động).  
- **Hãng khuyến nghị:** Basler, Hikrobot, Dahua, IDS Imaging (tuỳ ngân sách).  
- **Chiếu sáng:** Cần **đèn LED vòng hoặc backlight**, tránh bóng đổ và đảm bảo tính ổn định hình ảnh.
