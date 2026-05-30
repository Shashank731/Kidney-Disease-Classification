# 🩺 Kidney Disease Classification

An end-to-end Deep Learning and MLOps project for classifying kidney CT scan images into four categories: **Cyst, Normal, Stone, and Tumor** using a fine-tuned DenseNet121 model. The project includes data pipelines, experiment tracking, model versioning, and deployment through a Streamlit web application.

## 🚀 Features

* Kidney CT Scan Classification
* Transfer Learning with DenseNet121
* Experiment Tracking using MLflow
* Pipeline Management using DVC
* Interactive Streamlit Web Application
* End-to-End MLOps Workflow

---

## 📊 Dataset

* Kidney CT Scan Dataset
* 9,955 Images
* 4 Classes:

  * Cyst
  * Normal
  * Stone
  * Tumor

---

## 🏗️ Project Workflow

```text
Data Ingestion
      ↓
Data Splitting
      ↓
Model Preparation
      ↓
Model Training
      ↓
Evaluation
      ↓
Deployment
```

### Inference Pipeline

```text
CT Scan Image
      ↓
Preprocessing
      ↓
DenseNet121
      ↓
Prediction
      ↓
Confidence Score
```

---

## 🧠 Model

* Architecture: DenseNet121
* Transfer Learning: ImageNet Pretrained Weights
* Modified Classifier: 1024 → 4 Classes
* Optimizer: Adam
* Loss Function: CrossEntropyLoss

---

## 📈 Results

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 98.30% |
| Validation Accuracy | 97.47% |
| Validation Loss     | 0.0817 |

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Streamlit
* MLflow
* DVC
* Git & GitHub

---

## 📂 Project Structure

```text
src/
├── components/
├── pipeline/
├── utils/

artifacts/
configs/
research/
models/
app.py
main.py
dvc.yaml
```

---

## ⚠️ Limitation

The model is trained only on kidney CT scans and does not detect out-of-distribution images (e.g., faces, cars, animals).

---

## 🔮 Future Work

* Grad-CAM Explainability
* OOD Detection
* Probability Dashboard
* Model Monitoring

---

## 👨‍💻 Author

Shashank Saraswat

Machine Learning | Deep Learning | Computer Vision | MLOps
