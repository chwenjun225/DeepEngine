Bạn đã xây dựng một **prompt chuẩn** cho mô hình **Llama3.2-1B-Instruct** với nhiệm vụ **trích xuất và chuyển đổi yêu cầu của người dùng thành JSON theo schema cho trước**. Dưới đây là một số **cải tiến và gợi ý** để làm cho prompt của bạn tốt hơn, chính xác hơn và ít bị lỗi hơn.

---

## **1. Cải tiến Prompt**
Dưới đây là một số thay đổi để:
- **Tối ưu khả năng sinh JSON hợp lệ**, giảm nguy cơ mô hình sinh ra lỗi cú pháp.
- **Giữ đúng định dạng JSON** bằng cách thêm ràng buộc cụ thể hơn.
- **Hướng dẫn mô hình rõ ràng hơn** về cách xử lý thông tin từ người dùng.

### **Cải tiến `PROMPT_AGENT_PROMPT`**
```python
PROMPT_AGENT_PROMPT = """{BEGIN_OF_TEXT}{START_HEADER_ID}SYSTEM{END_HEADER_ID}
You are an assistant project manager in the AutoML development team. 
Your task is to parse the user's requirement into a valid JSON format, strictly following the given JSON specification schema. 
You must adhere to these rules:
1. **ONLY generate JSON output**—no explanations, extra text, or formatting errors.
2. Ensure that **all required fields** from the schema are present in your output.
3. **Extract relevant information** from the user’s input and fill in the JSON fields accordingly.
4. Your JSON **MUST be well-formatted** and **syntactically correct**.

Your response must strictly follow the JSON schema below:
```json
{json_schema}
```

### Example 1:
**User query:** Build a deep learning model, potentially using CNNs or Vision Transformers, to detect defects in PCB (Printed Circuit Board) images. The model should classify defects into categories like missing components, soldering issues, and cracks. We have uploaded the dataset as 'pcb_defects_dataset'. The model must achieve at least 0.95 accuracy.

**AI response:**
```json
{
	"problem_area": "computer vision",
	"task": "defect detection",
	"application": "electronics manufacturing",
	"dataset_name": "pcb_defects_dataset",
	"data_modality": ["image"],
	"model_name": "Vision Transformer",
	"model_type": "deep learning",
	"hardware_cuda": true,
	"hardware_cpu_cores": 16,
	"hardware_memory": "64GB"
}
```

### Example 2:
**User query:** Develop a machine learning model, potentially using ResNet or EfficientNet, to inspect industrial products for surface defects (scratches, dents, discoloration). The dataset is provided as 'industrial_defects_images'. The model should achieve at least 0.97 accuracy.

**AI response:**
```json
{
	"problem_area": "computer vision",
	"task": "surface defect detection",
	"application": "industrial manufacturing",
	"dataset_name": "industrial_defects_images",
	"data_modality": ["image"],
	"model_name": "EfficientNet",
	"model_type": "deep learning",
	"hardware_cuda": true,
	"hardware_cpu_cores": 12,
	"hardware_memory": "32GB"
}
```

{END_OF_TURN_ID}{START_HEADER_ID}HUMAN{END_HEADER_ID}
{human_msg}{END_OF_TURN_ID}
{START_HEADER_ID}AI{END_HEADER_ID}
Let's begin. Your response must start with ```json and end with ```. No extra text is allowed.{END_OF_TURN_ID}
"""
```

---

### **Cải tiến `PARSE_JSON_PROMPT`**
```python
PARSE_JSON_PROMPT = """{BEGIN_OF_TEXT}{START_HEADER_ID}SYSTEM{END_HEADER_ID}
You are an AI project assistant. Your task is to extract and structure user requirements into a valid JSON format **strictly following** the given schema.

## Rules:
- **Only output JSON**—do not include any explanation or extra text.
- Ensure that **all fields** in the JSON schema are present.
- If the user's request lacks specific details, use **default values or placeholders** where appropriate.

### JSON Schema:
```json
{json_schema}
```

### Example:
**User query:** Build a machine learning model, potentially XGBoost or LightGBM, to classify banana quality as Good or Bad based on numerical information (size, weight, sweetness, softness, harvest time, ripeness, and acidity). We have uploaded the dataset as 'banana_quality.csv'. The model must achieve at least 0.98 accuracy.

**AI response:**
```json
{
	"problem_area": "tabular data analysis",
	"task": "classification",
	"application": "agriculture",
	"dataset_name": "banana_quality",
	"data_modality": ["tabular"],
	"model_name": "XGBoost",
	"model_type": "ensemble",
	"hardware_cuda": false,
	"hardware_cpu_cores": 8,
	"hardware_memory": "32GB"
}
```
{END_OF_TURN_ID}{START_HEADER_ID}HUMAN{END_HEADER_ID}
{human_msg}{END_OF_TURN_ID}
{START_HEADER_ID}AI{END_HEADER_ID}
Your response **must be a valid JSON object** that adheres strictly to the schema.{END_OF_TURN_ID}
"""
```

---

## **2. Những cải tiến quan trọng**
### ✅ **Bảo đảm đầu ra chỉ chứa JSON**
- Mình đã thêm **ràng buộc rõ ràng** rằng **không được sinh ra bất kỳ văn bản nào khác ngoài JSON**.
- Đoạn `{END_OF_TURN_ID}` giúp mô hình hiểu được **định dạng chat** và phân tách rõ ràng đầu vào/đầu ra.

### ✅ **Đảm bảo JSON có đầy đủ thông tin**
- Nếu **người dùng không cung cấp đủ thông tin**, mô hình sẽ **tạo placeholder hoặc mặc định**.
- **Không bỏ sót trường nào trong JSON schema**.

### ✅ **Làm rõ cách xử lý thông tin**
- Khi người dùng nhập yêu cầu, mô hình **phải chọn kiến trúc phù hợp** (**CNN, Vision Transformer, XGBoost,...**) dựa trên yêu cầu.

---

## **3. Một số Prompt mở rộng**
### **Prompt 1: Kiểm tra lỗi trên dây chuyền sản xuất ô tô**
#### **User query:**
> Develop an AI model using YOLOv8 or Faster R-CNN to detect manufacturing defects in automotive parts, such as scratches, dents, and misaligned components. The dataset is provided as 'car_factory_defects'. The model should achieve at least 0.96 accuracy.

#### **AI response:**
```json
{
	"problem_area": "computer vision",
	"task": "defect detection",
	"application": "automotive manufacturing",
	"dataset_name": "car_factory_defects",
	"data_modality": ["image"],
	"model_name": "YOLOv8",
	"model_type": "deep learning",
	"hardware_cuda": true,
	"hardware_cpu_cores": 24,
	"hardware_memory": "128GB"
}
```

---

### **Prompt 2: Phân loại sản phẩm nông sản**
#### **User query:**
> Build a machine learning model using LightGBM to classify fruit quality based on numerical and categorical attributes (color, weight, size, ripeness level). The dataset is 'fruit_quality_dataset'. Target accuracy is 0.95.

#### **AI response:**
```json
{
	"problem_area": "tabular data analysis",
	"task": "classification",
	"application": "agriculture",
	"dataset_name": "fruit_quality_dataset",
	"data_modality": ["tabular"],
	"model_name": "LightGBM",
	"model_type": "ensemble",
	"hardware_cuda": false,
	"hardware_cpu_cores": 8,
	"hardware_memory": "32GB"
}
```

---

## **Tóm lại**
- ✅ **Cải tiến prompt** để **rõ ràng hơn, chính xác hơn**.
- ✅ **Chỉ sinh JSON hợp lệ**, tránh lỗi cú pháp.
- ✅ **Bổ sung các prompt mẫu** cho bài toán kiểm tra chất lượng sản phẩm.

Bạn có muốn tùy chỉnh thêm để phù hợp với workflow của mình không? 🚀