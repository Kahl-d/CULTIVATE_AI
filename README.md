# CULTIVATE_AI

# 🌱 CULTIVATE – AI for Understanding Cultural Themes in STEM  

## 📌 Overview  
CULTIVATE is an AI-driven research initiative that applies **Large Language Models (LLMs)** for **Cultural Capital Theme (CCT) identification** in educational narratives. Using **transformer-based models, fine-tuning techniques, and retrieval-augmented methods**, the project enhances the understanding of how **cultural assets impact STEM retention**.  

<img width="954" alt="Screenshot 2025-02-06 at 3 00 30 PM" src="https://github.com/user-attachments/assets/ad7138a6-2193-4274-9be7-3e3936d8882d" />



We explore multiple **BERT-based architectures, hyperparameter tuning strategies, domain adaptation techniques, and embedding visualizations** to optimize model performance in classifying **Cultural Capital Themes (CCTs)** from student essays and reflective journaling.  

---

## 🏗️ Core AI Methodologies  

### **1️⃣ Transformer Models & Variants Used**
- **BERT Family**:
  - **BERT (Base & Large)** – Baseline for contextual embeddings.
  - **RoBERTa** – Robustly optimized BERT pretraining approach.
  - **DeBERTa** – Advanced embeddings with disentangled attention.
- **T5 Variants**:
  - **T5 (Base & Large)** – Text-to-text format for classification and generation.
  - **Sentence-T5** – Fine-tuned for CCT identification.
- **DistilBERT** – Efficient, lightweight model for rapid inference.

---

### **2️⃣ Fine-Tuning & Hyperparameter Optimization**
We implemented extensive fine-tuning of **BERT, DeBERTa, and T5-based models**, optimizing hyperparameters such as:  
✅ **Batch Size**  
✅ **Learning Rate Scheduling** (Linear decay, Cosine Annealing)  
✅ **Dropout Regularization** (to prevent overfitting)  
✅ **Optimizer Variants** (AdamW, Adafactor)  
✅ **Warm-Up Steps & Weight Decay** for improved convergence  

Fine-tuning was conducted with **Hugging Face Transformers**, leveraging pre-trained weights and further adapting models on domain-specific datasets.

---

### **3️⃣ Domain Adaptation**
To improve cultural theme recognition, we applied **domain-adaptive pretraining techniques**, including:  
- **Masked Language Modeling (MLM) on domain-specific STEM narratives**  
- **Continued Pretraining on reflective journaling datasets**  
- **Adapter Layers for incremental learning without full model retraining**  

This allowed our models to better capture **culturally embedded expressions in STEM education**.

---

### **4️⃣ Downstream Classification Tasks**
After fine-tuning, models were tested for **CCT classification**, where **BERT-based models** were evaluated on:  
✅ **Multi-label Classification** of 11 Cultural Capital Themes  
✅ **Zero-shot Learning Capabilities** on unseen prompts  
✅ **Few-shot Adaptation** using labeled CCT datasets  
✅ **Cross-domain Generalization** for student essays vs. academic texts  

We utilized **F1-scores, Precision-Recall curves, and Confusion Matrices** to benchmark model performance.

---

### **5️⃣ Retrieval-Augmented Generation (RAG) for CCT Expansion**
To **enhance AI explainability**, we integrated **RAG (Retrieval-Augmented Generation)** to:  
- Retrieve relevant **historical and contextual CCT references** for justification.  
- Generate **context-aware explanations** for AI decisions.  
- Improve **theme identification accuracy** by augmenting model-generated responses with **retrieved external knowledge**.

RAG helped **bridge LLM reasoning with real-world cultural knowledge**, improving interpretability.

---

### **6️⃣ Knowledge Graphs for CCT Relationships**
To model relationships between **Cultural Capital Themes**, we:  
✅ Constructed **Knowledge Graphs** of **CCT interdependencies**.  
✅ Used **Prompt Engineering and DOMAIN ADAPTATION** to refine embeddings.  
✅ Linked **student narratives to cultural capital concepts** through entity recognition.

This approach allowed **better structural representation** of **how different cultural strengths impact STEM persistence**.
<img width="1171" alt="Screenshot 2025-02-06 at 3 38 42 PM" src="https://github.com/user-attachments/assets/ec5bcfc4-3d1e-47f9-8c2d-51e20df130d9" />

---

### **7️⃣ Embedding Visualization & t-SNE Analysis**
We applied **t-SNE & PCA** on learned **CCT embeddings** to:  
- **Cluster student narratives** based on CCT affinity.  
- **Visualize high-dimensional text representations** to identify **overlapping vs. distinct themes**.  
- Understand how **semantic distances** between CCT categories evolved **before and after fine-tuning**.  
<img width="450" alt="Picture12" src="https://github.com/user-attachments/assets/5fd4c038-928d-4090-b073-de8edc1e959e" />

---

## 🎯 Model Evaluation & Metrics  
To assess the **real-world impact of CULTIVATE’s AI models**, we used:  

📊 **Evaluation Metrics:**  
✅ **F1-score & Precision-Recall Curves** – To measure classification accuracy.  
✅ **AUC-ROC Analysis** – To evaluate binary theme detection performance.  
✅ **Per-Class Accuracy Breakdown** – To identify underperforming themes.  

📌 **Key Findings:**  
🚀 **Fine-tuned DeBERTa achieved the highest accuracy (XX%)** in detecting **Cultural Capital Themes**.  
🚀 **RAG-enhanced models improved explanation generation, reducing theme misclassification errors by XX%**.  
🚀 **Knowledge Graph integration helped uncover hidden relationships between STEM experiences and cultural capital concepts**.  

---

## 🔍 Future Enhancements  
🔹 **Expand domain-adaptive pretraining on multi-source cultural datasets**  
🔹 **Integrate semi-supervised learning to improve generalization**  
🔹 **Develop interactive annotation tools to refine CCT datasets**  
🔹 **Enhance retrieval mechanisms for more robust AI-assisted cultural insights**  

---

## 🏆 Acknowledgments  
This project is conducted at **San Francisco State University**, under the guidance of **Professor Anagha Kulkarni**.  

---

## 📜 Citation  
If you find this research useful, please cite:  

```bibtex
@article{Khan2024CULTIVATE,
  author = {Khalid Mehtab Khan},
  title = {CULTIVATE – AI for Understanding Cultural Themes in STEM},
  journal = {San Francisco State University Research},
  year = {2024},
  url = {https://github.com/khalidmehtabkhan/CULTIVATE}
}

