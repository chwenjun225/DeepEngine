# AI-Agent Project

## 1. Project Overview
The **AI-Agent** project is an advanced AI system designed to automate repetitive tasks such as:
- Weekly reminders
- Data collection and evaluation
- Performance scoring and reporting

This project leverages **Large Language Models (LLMs)** to achieve high accuracy and quality results while minimizing latency.

---

## 2. Objectives
- **Automate workflows**: Reduce manual effort by automating tasks with AI.
- **Improve accuracy**: Utilize cutting-edge LLMs to extract and analyze data with precision.
- **Deploy efficiently**: Optimize resource utilization for large-scale AI inference.

---

## 3. System Requirements

### 3.1 Hardware Requirements
The project requires high-performance hardware to support the deployment of LLMs (up to 70 billion parameters).

#### **Baseline Requirements**
| Component        | Specification                        |
|------------------|-------------------------------------|
| **GPU**          | NVIDIA A100 (80GB) / H100 (80GB)    |
| **CPU**          | Intel Xeon / AMD EPYC               |
| **RAM**          | 128GB - 256GB DDR4/DDR5             |
| **Storage**      | 2TB NVMe SSD (PCIe 4.0) + 8TB HDD   |
| **Power Supply** | 1600W - 2000W Platinum Rated PSU    |
| **Cooling**      | Liquid Cooling (CPU & GPU)          |
| **Motherboard**  | Supports multi-GPU configuration    |

#### **Workstation Configuration** *(Development phase)*
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: AMD Ryzen 9 7950X / Intel i9-13900K
- RAM: 128GB DDR5
- Storage: 2TB NVMe SSD

### 3.2 Software Requirements
| Software           | Version             |
|--------------------|---------------------|
| **Operating System** | Ubuntu 22.04 LTS    |
| **CUDA Toolkit**     | 12.0+               |
| **PyTorch**          | 2.0+                |
| **Transformers**     | HuggingFace library |
| **Python**           | 3.9+                |
| **Docker**           | 24.0+               |

---

## 4. Deployment Plan

### **4.1 Development Phase**
1. **Setup Environment**:
   - Install required drivers, CUDA Toolkit, and libraries.
   - Configure Python virtual environment and dependencies.

2. **Fine-tune Models**:
   - Use pre-trained LLaMA-2 / LLaMA-3 models for fine-tuning.
   - Optimize JSON extraction tasks and test inference latency.

3. **Testing**:
   - Validate model outputs using evaluation metrics (e.g., `llm-eval-harness`).
   - Benchmark resource utilization and accuracy.

### **4.2 Deployment Phase**
1. **Resource Allocation**:
   - Use centralized GPU infrastructure for deployment.
   - Allocate dedicated GPUs for long-term inference.

2. **Performance Monitoring**:
   - Monitor GPU usage, latency, and response time.

3. **Scaling**:
   - Integrate with cloud-based GPUs (if required) for scaling large-scale inference tasks.

---

## 5. Usage Guide
### **Run the Project**
1. Clone the repository:
   ```bash
   git clone <project-repo-url>
   cd AI-Agent-Project
   ```
2. Setup Python environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the inference script:
   ```bash
   python main.py --model llama-2-70b --task json_extraction
   ```

---

## 6. Future Improvements
- Integrate additional large-scale models (e.g., GPT-4, Falcon).
- Optimize resource usage for multi-GPU training.
- Explore distributed systems for larger-scale deployments.

---

## 7. Contact
For further details, please contact:
- **Project Lead**: Tran Van Tuan 陳文俊
- **Email**: trantuan22052k@gmail.com
- **Department**: Algorithm Development Team of AI Department 
