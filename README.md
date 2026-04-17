# Parkinson's Disease Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Coming%20Soon-green.svg)](https://xgboost.readthedocs.io/)

---

## 📌 Project Overview

This project focuses on **early detection of Parkinson’s Disease (PD)** using voice-based biomedical features and machine learning models.

The workflow includes:

* Data preprocessing and cleaning
* Class imbalance handling (SMOTETomek)
* Feature scaling
* Model training using **Gradient Boosting**
* Hyperparameter tuning using **GridSearchCV (cross-validation)**
* Model evaluation using multiple performance metrics
* Visualization of results (ROC curve, confusion matrix, feature importance)

---

## 📊 Dataset

### 1. Public Dataset

* **Source**: https://archive.ics.uci.edu/ml/datasets/parkinsons
* 195 voice recordings, 22 biomedical features

### 2. Institutional Dataset

* Research: *“Develop integrated diagnosis and prediction model of Parkinson's disease”*
* Conducted at **TMU Shuang-Ho Hospital**
* IRB Approval: **TMU-JIRB Protocol No. N201801043**
* ❗ Not publicly available due to privacy restrictions

### 🎯 Target Variable

* `status`

  * `1` → Parkinson’s Disease
  * `0` → Healthy Control

---

## ⚙️ Machine Learning Pipeline

1. **Data Cleaning**

   * Numeric feature selection
   * Missing value handling

2. **Train-Test Split**

   * Stratified split to preserve class distribution

3. **Feature Scaling**

   * StandardScaler applied to training data

4. **Class Balancing**

   * SMOTETomek applied **only to training set**

5. **Model Training**

   * Gradient Boosting Classifier

6. **Hyperparameter Tuning**

   * GridSearchCV (5-fold cross-validation)
   * Optimized using ROC-AUC

7. **Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * ROC-AUC
   * MCC

---

## 📈 Results

### 🔍 Best Model Parameters

```
learning_rate = 0.1
max_depth     = 4
n_estimators  = 100
```

### 📊 Cross-Validation Performance

* ROC-AUC (CV): **0.9886**

### 📊 Test Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9565 |
| Precision | 0.9737 |
| Recall    | 0.9737 |
| F1-score  | 0.9737 |
| ROC-AUC   | 0.9868 |
| MCC       | 0.8487 |

### 🔢 Confusion Matrix

```
[[ 7  1]
 [ 1 37]]
```

---

## 📊 Visualizations

The project includes:

* Confusion Matrix
* ROC Curve
* Feature Importance
* Performance Metrics Bar Chart

Generated using:

```
src/visualizer.py
```

---

## 📁 Project Structure

```
parkinsons-xgboost/
│
├── src/
│   ├── data_utils.py
│   ├── visualizer.py
│
├── notebooks/
│   ├── data_utils.ipynb
│   ├── model_trainer.ipynb
│   ├── visualizer.ipynb
│
├── models/
│   ├── gb_pd_alldata_v1.pkl
│   ├── scaler_pd_alldata_v1.pkl
│
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Clone repository

```bash
git clone https://github.com/yourusername/parkinsons-xgboost.git
cd parkinsons-xgboost
```

### 2. Create environment

```bash
conda create -n pd-env python=3.9
conda activate pd-env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run project

```bash
jupyter lab
```

Run:

* notebooks/model_trainer.ipynb
* notebooks/visualizer.ipynb

---

## 🚀 Future Work

* Add **XGBoost model comparison**
* Integrate **FastAI CNN (spectrogram-based)**
* Expand dataset for better generalization
* Deploy model via **FastAPI**
* Real-time inference system

---

## 📌 Key Takeaways

* Proper class imbalance handling is critical
* Cross-validation improves reliability
* ROC-AUC and MCC are better for imbalanced data
* Voice biomarkers are promising for PD detection

---

## 📬 Contact

* Email: [mirnadanisat@gmail.com](mailto:mirnadanisat@gmail.com)
* LinkedIn: https://www.linkedin.com/in/mirnadanisatandjung/
