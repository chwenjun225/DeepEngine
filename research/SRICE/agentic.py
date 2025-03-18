import torch 
import json 



MODEL_PATH = "/home/chwenjun225/.llama/checkpoints/Llama3.2-11B-Vision-Instruct"



with open(f"{MODEL_PATH}/params.json", "r") as f:
    model_config = json.load(f)



model = torch.load(
    f"{MODEL_PATH}/consolidated.00.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"
)



print("Mô hình đã tải thành công với số lượng tham số:", len(model.keys()))
