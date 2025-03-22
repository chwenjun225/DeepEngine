# AutoML-MultiAgent for AOI Tasks

## System Workflow Explanation  

### Step 1 – Reasoning Agent (Multi-Agent)
**EN:** Understands the user's request and automatically plans to train a suitable AI-Vision model based on the query content.  
**中文:** 理解用户需求，并根据查询内容自动规划训练一个合适的视觉 AI 模型。  
**拼音:** Lǐjiě yònghù xūqiú, bìng gēnjù cháxún nèiróng zìdòng guīhuà xùnliàn yī gè héshì de shìjué AI móxíng.

---

### Step 2 – Vision Agent
**EN:** Uses the trained AI-Vision model to detect visual defects on PCB products from images or video.  
**中文:** 使用刚训练完成的 AI-Vision 模型，从图像或视频中检测 PCB 产品的外观缺陷。  
**拼音:** Shǐyòng gāng xùnliàn wánchéng de AI-Vision móxíng, cóng túxiàng huò shìpín zhōng jiǎncè PCB chǎnpǐn de wàiguān quēxiàn.

---

### Step 3 – LLaMA-Vision-Instruct
**EN:** Analyzes the image and detection results provided by the Vision Agent.  
**中文:** 分析 Vision Agent 提供的图像和检测结果。  
**拼音:** Fēnxī Vision Agent tígōng de túxiàng hé jiǎncè jiéguǒ.

---

### Step 4 – DeepSeek-R1-7B-Instruct
**EN:** Processes the output from LLaMA and makes the final decision: **OK** or **NG**.  
**中文:** 进一步分析 LLaMA 的输出，并作出最终判断：OK 或 NG。  
**拼音:** Jìnyībù fēnxī LLaMA de shūchū, bìng zuòchū zuìzhōng pànduàn: OK huò NG.

---
