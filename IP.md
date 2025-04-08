### 1. Title
Multi-Agent System for Real-Time Automated Optical Inspection Using YOLO and Language Models

---

### 2. Field of the Invention
The present invention relates to the field of automated optical inspection (AOI) in electronics manufacturing, and more specifically, to real-time defect detection and classification using a multi-agent system combining computer vision, object detection models (YOLO), and large language models (LLMs).

---

### 3. Background
Traditional AOI systems rely on rule-based approaches or shallow machine learning models that are limited in adaptability and scalability. These systems often fail to handle ambiguous or novel defect patterns and require constant human tuning. Recent developments in deep learning and object detection (e.g., YOLO) have improved performance but still face challenges in explainability, integration, and real-time decision-making in production environments.

---

### 4. Summary
The invention proposes a novel multi-agent architecture for AOI that leverages:
- A vision agent using YOLO for real-time defect localization.
- A reasoning agent based on a large language model (LLM) to analyze detection metadata, generate insights, and guide decisions.
- A controller agent to orchestrate the workflow asynchronously and optimize decisions.

This system improves adaptability, reduces false positives, and enables human-in-the-loop feedback in a seamless production pipeline.

---

### 5. Brief Description of the Drawings
- Figure 1: System architecture of the multi-agent AOI framework
- Figure 2: YOLO object detection results with annotated defects
- Figure 3: Flow diagram of metadata reasoning by the LLM
- Figure 4: Interaction flow between vision agent, reasoning agent, and controller
- Figure 5: Example of NG/OK classification report with feedback loop

---

### 6. Detailed Description
The system comprises three core software agents:

**(a) Vision Agent:** Captures image/video input from AOI camera, processes using pretrained YOLOv8 model, and extracts bounding box metadata for detected objects/defects.

**(b) Reasoning Agent:** Receives metadata from vision agent, encodes the context, and performs reasoning using a fine-tuned large language model. It assesses defect severity, classifies NG/OK status, and explains reasoning when required.

**(c) Controller Agent:** Handles coordination between agents, manages inference timing, triggers alerts, and provides API endpoints for external integration (e.g., MES).

All agents operate asynchronously with optional streaming mode for real-time factory deployment. The system can be customized with different model checkpoints and fine-tuned prompts depending on PCB type or product line.

---

### 7. Claims
1. A multi-agent AOI framework comprising a vision agent, a reasoning agent, and a controller agent, wherein:
   - The vision agent employs a YOLO-based model to detect defects in PCB images.
   - The reasoning agent utilizes a large language model to classify defect severity based on detection metadata.
   - The controller agent synchronizes operation and provides integration with external systems.

2. The system of claim 1, wherein the agents communicate asynchronously and support real-time streaming.

3. The system of claim 1, wherein human operators can interact with the reasoning agent for explainability and quality control.

4. The system of claim 1, wherein the reasoning agent is fine-tuned for specific manufacturing use cases.

---

### 8. Abstract
A system and method for real-time automated optical inspection of printed circuit boards using a multi-agent framework. The system includes a YOLO-based vision agent for defect detection, a large language model-based reasoning agent for classification and explanation, and a controller agent for coordination. The agents operate asynchronously to provide accurate, explainable, and scalable defect detection suitable for production environments.




<!-- 

Note cần làm:
1. Tham khảo cách viết của: https://patentimages.storage.googleapis.com/68/f0/75/b31b89b81034a0/US20230260260A1.pdf

 -->


<!-- 

1. Abstract

**MÔ TẢ BẰNG LỜI CÁCH THUẬT TOÁN CHƯƠNG TRÌNH HOẠT ĐỘNG**

A machine learning method comprises:

S210. applying a constrastive learning model to a training image and an image mask to generate a foreground feature vector pair and a background feature pair, wherein the training image corresponds to the image mask.

S220. calculating a foreground loss and a background loss according to the foreground feature vector pair and the background feature vector pair.

S230. performing a weighted loss calculation on the foreground loss and the background loss by using a first weight and a second weight to generate a total loss, wherein the first weight corresponds to the foreground loss, and the second weight corresponds to the background loss .

S240. determine whether a recursion end condition is met according to the total loss.

S250. if YES, adjusting a parameter of a machine learning model by using the first encoder.

S260. if NO, adjusting ta parameter of the first encoder of the constrastive learning model by using the total loss, adjusting a parameter of the second encoder of the contrastive learning model by using the total loss and a preset multiple, capturing a new training image and a new training iamge and a new image mask corresponding to the new training image and the new image mask as the training image and the image mask.



MACHINE LEARNING METHOD AND DEVICE 

CROSS-REFERENCE TO RELATED APPLICATION

[0001]   This application claims priority to U.S.Provisional Application Ser.No
 -->