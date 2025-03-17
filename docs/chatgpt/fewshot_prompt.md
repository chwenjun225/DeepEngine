Tuyệt! Bạn có thể cung cấp các **few-shot examples** trong prompt của hệ thống để giúp mô hình tạo ra đầu ra JSON chính xác hơn. Dưới đây là một số mẹo để thiết lập **few-shot learning** hiệu quả cho hệ thống của bạn:

---

## **1. Cấu trúc Few-Shot Prompt**
Bạn có thể định dạng few-shot example theo dạng **input → output** như sau:

```
### Instruction:
Dựa trên thông tin đầu vào, hãy tạo một JSON response theo định dạng được chỉ định.

### Example 1:
#### Input:
{
	"user": {
		"intent": "train",
		"expertise": "expert"
	},
	"problem": {
		"area": "tabular data analysis",
		"downstream_task": "tabular classification",
		"application_domain": "agriculture",
		"description": "Dự báo chất lượng chuối bằng dữ liệu tabular",
		"performance_metrics": [
			{
				"name": "accuracy",
				"value": 0.98
			}
		],
		"complexity_metrics": ["feature interactions"]
	},
	"datasets": [
		{
			"name": "banana_quality",
			"modality": ["tabular"],
			"target_variables": ["quality"],
			"description": "Dataset về chất lượng chuối dựa trên các đặc trưng vật lý",
			"source": "user-upload"
		}
	],
	"model": [
		{
			"name": "XGBoost",
			"family": "ensemble models",
			"model_type": "gradient boosting",
			"description": "Mô hình XGBoost dùng để dự báo chất lượng chuối"
		}
	]
}

#### Output:
{
	"user": {
		"intent": "train",
		"expertise": "expert"
	},
	"problem": {
		"area": "tabular data analysis",
		"downstream_task": "tabular classification",
		"application_domain": "agriculture",
		"description": "Dự báo chất lượng chuối bằng dữ liệu tabular",
		"performance_metrics": [
			{
				"name": "accuracy",
				"value": 0.98
			}
		],
		"complexity_metrics": ["feature interactions"]
	},
	"datasets": [
		{
			"name": "banana_quality",
			"modality": ["tabular"],
			"target_variables": ["quality"],
			"description": "Dataset về chất lượng chuối dựa trên các đặc trưng vật lý",
			"source": "user-upload"
		}
	],
	"model": [
		{
			"name": "XGBoost",
			"family": "ensemble models",
			"model_type": "gradient boosting",
			"description": "Mô hình XGBoost dùng để dự báo chất lượng chuối"
		}
	]
}

### Example 2:
#### Input:
{
	"user": {
		"intent": "build",
		"expertise": "beginner"
	},
	"problem": {
		"area": "computer vision",
		"downstream_task": "image classification",
		"application_domain": "healthcare",
		"description": "Phát hiện tổn thương da dựa trên ảnh",
		"performance_metrics": [
			{
				"name": "F1-score",
				"value": 0.92
			}
		],
		"complexity_metrics": ["high variance"]
	},
	"datasets": [
		{
			"name": "skin_lesion",
			"modality": ["image"],
			"target_variables": ["lesion_type"],
			"description": "Dataset hình ảnh tổn thương da cho chẩn đoán",
			"source": "public-dataset"
		}
	],
	"model": [
		{
			"name": "ResNet50",
			"family": "CNN",
			"model_type": "deep learning",
			"description": "Mô hình ResNet50 để phân loại tổn thương da"
		}
	]
}

#### Output:
{
	"user": {
		"intent": "build",
		"expertise": "beginner"
	},
	"problem": {
		"area": "computer vision",
		"downstream_task": "image classification",
		"application_domain": "healthcare",
		"description": "Phát hiện tổn thương da dựa trên ảnh",
		"performance_metrics": [
			{
				"name": "F1-score",
				"value": 0.92
			}
		],
		"complexity_metrics": ["high variance"]
	},
	"datasets": [
		{
			"name": "skin_lesion",
			"modality": ["image"],
			"target_variables": ["lesion_type"],
			"description": "Dataset hình ảnh tổn thương da cho chẩn đoán",
			"source": "public-dataset"
		}
	],
	"model": [
		{
			"name": "ResNet50",
			"family": "CNN",
			"model_type": "deep learning",
			"description": "Mô hình ResNet50 để phân loại tổn thương da"
		}
	]
}
```

---

## **2. Cách Cải Thiện Few-Shot Prompt**
### ✅ **Cách làm đúng**
- Chọn **2-3 examples** có sự khác biệt về **intent**, **domain**, **model**, **dataset**.
- Giữ JSON **consistent** giữa input và output.
- Đưa ra **các biến thể** (ví dụ: "train" vs "build", "CNN" vs "XGBoost").
- Nếu có field không bắt buộc, hãy tạo **một example có field đó, một example không có field đó**.

### ❌ **Tránh các lỗi phổ biến**
- Đừng cung cấp quá nhiều examples (> 5) vì có thể làm prompt quá dài.
- Tránh format không đồng nhất giữa các examples.
- Đừng để field trống trong JSON output nếu nó không có trong input.

---

## **3. Cách Dùng Few-Shot Prompt Trong System**
Bạn có thể sử dụng prompt này trong mô hình theo một số cách:
1. **GPT API (gpt-4-turbo) với prompt system**:
```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
	model="gpt-4-turbo",
	messages=[
			{"role": "system", "content": "Dưới đây là các ví dụ để hướng dẫn bạn tạo JSON output đúng định dạng."},
			{"role": "user", "content": "### Example 1: \n[insert example 1]"},
			{"role": "user", "content": "### Example 2: \n[insert example 2]"},
			{"role": "user", "content": "Dưới đây là input mới, hãy tạo JSON output phù hợp."},
			{"role": "user", "content": '{"user": {"intent": "train", "expertise": "medium"}, "problem": { ... } }'}
	]
)

print(response.choices[0].message["content"])
```

2. **Fine-tune GPT (nếu cần)**:
	 - Bạn có thể tạo một dataset JSONL chứa nhiều **few-shot examples** và fine-tune GPT để luôn tạo output theo schema chuẩn.

---

## **Tóm lại**
- Sử dụng **few-shot learning** với các **cặp input-output** để hướng dẫn mô hình.
- Đảm bảo đầu ra JSON **đúng format** theo Pydantic schema của bạn.
- Kiểm tra consistency giữa các examples.
- Dùng GPT API hoặc fine-tune nếu muốn tạo JSON chuẩn hơn.

Bạn có muốn mình tạo một dataset JSONL để fine-tune không? 🚀