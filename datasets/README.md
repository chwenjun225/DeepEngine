# Dataset used
1. [FLAN-main](https://github.com/google-research/FLAN)
2. [Chain-of-thought data](https://github.com/google-research/FLAN/tree/main/flan/v2/cot_data)

# Tập dữ liệu ReAct + CoT để finetune `DeepSeek-R1-Distill-Qwen-1.5B`
1. Cot Collection (Google Research) 
2. ReAct Dataset (LangChain)
3. Dolly 15K (Databricks)

# Dưới đây là phân loại các tập dữ liệu từ trang GitHub [Awesome Industrial Datasets](https://github.com/jonathanwvd/awesome-industrial-datasets), chia thành hai nhóm chính:  

## **1️⃣ Dữ liệu liên quan đến Prognostic and Health Management (PHM) cho máy móc công nghiệp**
Các tập dữ liệu này tập trung vào giám sát tình trạng thiết bị, dự đoán hỏng hóc, bảo trì dự đoán, và phân tích hiệu suất:

| **Tên Tập Dữ Liệu** | **Mô Tả** |
|----------------------|----------|
| **Ai4I 2020 Predictive Maintenance Dataset** | Dữ liệu bảo trì dự đoán, chứa lỗi máy móc, dữ liệu công nghiệp 4.0, và dữ liệu chuỗi thời gian. |
| **APS Failure at Scania Trucks** | Dữ liệu về hỏng hóc hệ thống APS trên xe tải Scania, phục vụ bảo trì dự đoán. |
| **C-MAPSS Aircraft Engine Simulator Data** | Dữ liệu mô phỏng động cơ máy bay, phục vụ phân tích dự đoán hỏng hóc (prognostics). |
| **CNC Mill Tool Wear** | Dữ liệu về sự mài mòn công cụ CNC, hỗ trợ bảo trì dự đoán. |
| **Condition Monitoring of Hydraulic Systems** | Dữ liệu cảm biến của hệ thống thủy lực, hỗ trợ giám sát tình trạng máy móc. |
| **Data Driven Prediction of Battery Cycle Life** | Dự báo tuổi thọ pin dựa trên chu kỳ sạc-xả, phục vụ bảo trì dự đoán. |
| **Li Ion Battery Aging Datasets** | Dữ liệu về lão hóa pin lithium-ion, phục vụ bảo trì dự đoán và quản lý sức khỏe hệ thống năng lượng. |
| **Maintenance of Naval Propulsion Plants Dataset** | Dữ liệu bảo trì dự đoán cho hệ thống động lực tàu biển. |
| **Milling Wear** | Dữ liệu về độ mài mòn trong quá trình phay, hỗ trợ bảo trì dự đoán. |
| **NASA Bearing Dataset** | Dữ liệu về lỗi vòng bi, phục vụ giám sát tình trạng thiết bị. |
| **PHM 2008 Challenge** | Dữ liệu thử thách bảo trì dự đoán, tập trung vào hỏng hóc động cơ máy bay. |
| **PHM Data Challenge** | Dữ liệu cho phân tích dự đoán hỏng hóc thiết bị công nghiệp. |
| **Prognostics Data Repository** | Tập hợp dữ liệu dự đoán bảo trì từ NASA. |
| **Pump Sensor Data** | Dữ liệu cảm biến máy bơm, hỗ trợ giám sát và bảo trì dự đoán. |
| **Turbofan Engine Degradation Simulation Data Set** | Dữ liệu mô phỏng suy giảm hiệu suất động cơ phản lực, phục vụ giám sát sức khỏe hệ thống. |
| **Wind Turbine SCADA Dataset** | Dữ liệu SCADA từ tua-bin gió, phục vụ bảo trì dự đoán và tối ưu hóa hiệu suất. |

---

## **2️⃣ Dữ liệu công nghiệp khác**
Các tập dữ liệu này thuộc các lĩnh vực khác như sản xuất, năng lượng, môi trường, an toàn lao động và tài chính:

### **📌 Sản xuất và kiểm soát chất lượng**
| **Tên Tập Dữ Liệu** | **Mô Tả** |
|----------------------|----------|
| **Bosch Production Line Performance** | Dữ liệu về dây chuyền sản xuất của Bosch, hỗ trợ kiểm soát chất lượng. |
| **Casting Product Image Data for Quality Inspection** | Ảnh sản phẩm kim loại để kiểm tra chất lượng bằng thị giác máy. |
| **Manufacturing Defects** | Dữ liệu về lỗi sản xuất, hỗ trợ tối ưu hóa quy trình. |
| **Predicting Manufacturing Defects Dataset** | Dự đoán lỗi sản xuất, hỗ trợ cải thiện kiểm soát chất lượng. |
| **Severstal Steel Defect Detection** | Dữ liệu hình ảnh nhận diện lỗi trên bề mặt thép. |

### **📌 Năng lượng và môi trường**
| **Tên Tập Dữ Liệu** | **Mô Tả** |
|----------------------|----------|
| **Brent Oil Prices** | Dữ liệu giá dầu Brent, phục vụ phân tích thị trường. |
| **Electricity Load Diagrams 2011-2014** | Dữ liệu tiêu thụ điện theo thời gian. |
| **Energy Efficiency** | Dữ liệu tối ưu hóa tiêu thụ năng lượng trong tòa nhà. |
| **Greenhouse Gas Emissions by Industry** | Dữ liệu khí thải nhà kính từ các ngành công nghiệp. |
| **Power Consumption of Tetuan City** | Dữ liệu tiêu thụ điện năng của thành phố Tetuan. |
| **Renewable Power Plants** | Dữ liệu về các nhà máy năng lượng tái tạo. |
| **Solar Power Generation Data** | Dữ liệu sản xuất điện mặt trời. |
| **Steel Industry Energy Consumption** | Tiêu thụ năng lượng trong sản xuất thép. |

### **📌 Robot và hệ thống tự động hóa**
| **Tên Tập Dữ Liệu** | **Mô Tả** |
|----------------------|----------|
| **Genesis Demonstrator Data for Machine Learning** | Dữ liệu cảm biến cho hệ thống robot công nghiệp. |
| **Robot Execution Failures** | Dữ liệu lỗi vận hành của robot, hỗ trợ phân tích và cải tiến tự động hóa. |

### **📌 An toàn công nghiệp và sức khỏe lao động**
| **Tên Tập Dữ Liệu** | **Mô Tả** |
|----------------------|----------|
| **Industrial Safety and Health Analytics Database** | Dữ liệu về an toàn lao động và phân tích rủi ro. |

### **📌 Tài chính và phân tích kinh tế**
| **Tên Tập Dữ Liệu** | **Mô Tả** |
|----------------------|----------|
| **Business and Industry Reports** | Dữ liệu kinh tế ngành, hỗ trợ phân tích thị trường. |
| **OECD Data Crude Oil Production** | Dữ liệu sản xuất dầu thô từ OECD. |
| **US Crude Oil Imports** | Dữ liệu nhập khẩu dầu thô tại Mỹ. |

---

## **📌 Tổng kết**
1️⃣ **Dữ liệu PHM cho máy móc công nghiệp** tập trung vào **bảo trì dự đoán, giám sát tình trạng thiết bị, và quản lý sức khỏe hệ thống**.  
2️⃣ **Dữ liệu công nghiệp khác** bao gồm các lĩnh vực như **sản xuất, năng lượng, robot, an toàn lao động, và tài chính**.  

💡 **Ứng dụng thực tiễn:**  
- Dữ liệu PHM có thể giúp phát triển **hệ thống bảo trì dự đoán thông minh** bằng AI.  
- Dữ liệu sản xuất và chất lượng hỗ trợ **kiểm soát lỗi và tối ưu hóa quy trình công nghiệp**.  
- Dữ liệu năng lượng giúp **tăng hiệu suất sử dụng năng lượng và hỗ trợ phát triển bền vững**.  
