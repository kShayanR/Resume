# Resume

### Research Experience

#### ---Research Project 1---
**Title**: AI & Machine Learning in Malware detection 
**Role**: Research Assistant under the supervision of Dr. Reza Momen, (my university lecturer) 
**Date**: from September 2024 to February 2025

**Research Goals**: Exploring different ML-based approaches and methodologies in Cyber Security and Malware Detection, including processing and analyzing different data types alongside working on and deploying the AI programs for case studies and real world scenarios.

**Explanation (in details)**:
This Research project (AI/ML in Cyber Security and Malware detection) covered numerous techniques from Machine Learning-Based to Deep learning approaches plenty of data of different sorts and structures. These generally include:

**1. Machine Learning-Based Detection**

Types of Machine Learning Models Used:
1. Supervised Learning algorithms (Models are trained on labeled data - samples labeled as "malicious" or "benign"): Decision Trees, Random Forests, ​Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Logistic Regression
2. Unsupervised Learning algorithms (Used when labeled data is scarce. Models identify anomalies or clusters within data): Clustering (with K-means clustering), Anomaly detection models

Data Types and Categories used for Training and Prediction:
1. Static Analysis Features:
	- File Metadata: Information such as file size, type, permissions, and timestamps.
	- File Structure: Sections, imports, exports, and headers (e.g., the PE headers in Windows executables).
	- Disassembly Code and Instructions: The assembly-level code in binaries, which can be analyzed to detect potentially malicious code snippets.
	- String Analysis: Analysis of strings within the file, such as URLs, IP addresses, commands, or suspicious keywords.
2. Dynamic Analysis Features:
	- Behavioral Logs: Logs that capture how the file behaves in a sandbox or controlled environment, such as API calls, file operations, network connections, and registry modifications.
	- System Calls: A sequence of system calls that indicate interactions with the OS (e.g., accessing certain files or services).
	- Resource Usage: Observing CPU, memory, and I/O activities when the software runs, which may highlight abnormal patterns.
	- Network Traffic Analysis: Outbound and inbound connections made by the program, including IP addresses, ports, protocols, and any command-and-control (C2) traffic.

**2. Deep Learning and AI Techniques**

Deep learning (DL) builds upon machine learning but uses neural networks with multiple layers (hence "deep") to model complex patterns in large datasets. These networks can automatically extract features without the need for feature engineering, making them highly effective for detecting sophisticated malware patterns, such as polymorphic or metamorphic malware, which change their code to avoid detection.

Types of Deep Learning Models Used:
1. Convolutional Neural Networks (CNNs): Often used for image-like data (e.g., binary files converted into images or byte-level representations).
2. Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) Networks: Useful for analyzing sequences, such as system call sequences or API call logs, which contain temporal patterns.
3. Autoencoders: Used for anomaly detection by learning the "normal" data distribution and flagging anything that deviates significantly.
4. Graph Neural Networks (GNNs): Applied when data can be represented as graphs (e.g., function-call graphs or dependency graphs).
5. Transformers: Often used in natural language processing but applicable in cybersecurity for sequence modeling, making them suitable for analyzing logs or file sequences.

Data Types and Categories used for Training and Prediction:
Deep learning approaches often use similar data sources as machine learning models but are more flexible and can process raw, high-dimensional data without extensive feature engineering. Some common data inputs include:
1. Raw Binary Files:
	- Byte Sequences: Deep learning models can process the raw byte code of files. CNNs, for instance, can treat the byte sequences as image-like data and detect patterns indicative of malware.
	- Opcode Sequences: Operating codes (opcodes) extracted from binaries provide insight into the instructions that the file executes.
2. System Call Sequences (Dynamic Analysis):
	- RNNs or LSTM networks are particularly effective here, as they can model the sequence and frequency of system calls, capturing behavioral patterns of malware over time.
3. API Call Graphs and Dependency Graphs:
	- Graph Neural Networks (GNNs) are used to analyze relationships and dependencies within the malware. These models can detect complex interactions and embedded code structures often used in sophisticated malware.
4. Network Traffic Patterns:
	- Packet Data: Deep learning models can process packet data to identify C2 communication or other anomalous network traffic.
	- Time-Series Data: Transformer-based models or LSTMs can track sequential network behaviors that may indicate malware activity over a period.
5. Logs from Sandboxes and Runtime Environments:
	- Logs capturing events during dynamic analysis provide a sequence of actions or function calls that deep learning models can learn from. This can include registry changes, file modifications, and other runtime behaviors.
6. Natural Language Data (for Text-Based Malware or Phishing Detection):
	- Email and URL Analysis: Transformers and NLP models can analyze emails or URLs to detect phishing attempts or malicious links.
	- Textual Patterns: Machine learning models can analyze strings in the binary or the text content in scripts to detect social engineering malware.

**Findings**: Had successful achievements in AI-based Malware detection programs enhancement and development (Especially in Deep Learning criteria).

**Achievements**: Achieving notable results (in form of notable accuracy improvement in Malware detection) in Deep Learning-based Detector applications by utilizing Deep Neural Networks on various data types.

---

#### ---Research Project 2---
**Title**: Deep Learning in Aspect Extraction (ATE+ACD), a sub-field of Aspect-based Sentiment Analysis (ABSA)
**Role**: Research Assistant under the supervision of Dr. Masoud Dadgar (university lecturer & department chair/head) 
**Date**: from September 2024 to June 2025

**Explanation (in details)**:
This Research Project included exploring and utilizing a variety of approaches ranging from unsupervised statistical methods to supervised and semi-supervised deep learning models for enhancing Aspect Identification, a subtask of Aspect-based Sentiment Analysis (ABSA), which itself consists of two primary tasks (both sentence-level):
- Aspect Term Extraction (ATE) for both Implicit & Explicit Aspects
- Aspect Category Detection (ACD)
I started by trying out unsupervised techniques, the ones that didn't require training data. Some of the approaches I examined include: 
- Topic modeling with LDA
- Text vectorization (word-sentence embedding, TF-IDF, etc) + cosine similarity and a pre-defined knowledge base
- Zero-shot classification
- Ontology-based
- Conditional random field (CRF)
- FIN algorithm
- Rule-based: POS tagging + predefined dependency rules
- Named Entity Recognition (NER)

Further as I move on, I switched to supervised and semi-supervised SOTA (state-of the-art) deep learning models & approaches. The ones employed are: 
- Bidirectional Long Short-Term Memory (BiLSTM)
- Convolutional Neural Networks (CNNs)
- Graph Neural Networks (GNNs)
- Reinforcement Learning using Deep Q-Networks (DQNs)
- Transformer models (including both Discriminative and Generative variants)

As I continued examining a wide range of approaches for aspect extraction, highlighting the ones mentioned above, I managed to achieve notable improvements in accuracy regarding both ATE & ACD whilst utilizing SOTA Transformer models. Therefore, from a certain point on, I centralized my focus solely on these models to get my desired results. 

The ultimate SOTA results were achieved by utilizing Multi-Teacher Knowledge Distillation (MTKD) that is simply based on training multiple deep learning teacher models for each task and then transferring the knowledge of each individual teacher to a compact student model using knowledge distillation in order to have an all-in-one student model which has mastered the abilities of all its teachers in both individual tasks (ATE/ACD) as well as joint ACD+ATE.
