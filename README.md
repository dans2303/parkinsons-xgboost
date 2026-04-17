# Parkinson's Disease Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Implemented-success.svg)](https://xgboost.readthedocs.io/)

---

## 📌 Project Overview

This project focuses on **early detection of Parkinson’s Disease (PD)** using voice-based biomedical features and machine learning models.

The workflow includes:

* Data preprocessing and cleaning
* Class imbalance handling (SMOTETomek)
* Feature scaling
* Model training using **Gradient Boosting and XGBoost**
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
   * XGBoost Classifier (for comparison)

6. **Hyperparameter Tuning**

   * GridSearchCV (5-fold cross-validation)
   * Optimized using ROC-AUC

7. **Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * ROC-AUC
   * Matthews Correlation Coefficient (MCC)

---

## 📈 Results

### 🔍 Best Model Parameters

**Gradient Boosting**

```
learning_rate = 0.1
max_depth     = 4
n_estimators  = 100
```

**XGBoost**

```
learning_rate = 0.1
max_depth     = 4
n_estimators  = 100
```

---

### 📊 Cross-Validation Performance

* Gradient Boosting ROC-AUC (CV): **0.9886**

---

### 📊 Test Performance Comparison

| Model             | Accuracy | ROC-AUC | MCC    |
| ----------------- | -------- | ------- | ------ |
| Gradient Boosting | 0.9565   | 0.9868  | 0.8487 |
| XGBoost           | 0.9783   | 0.9704  | 0.9233 |

---

### 🔢 Confusion Matrix

**Gradient Boosting**

```
[[ 7  1]
 [ 1 37]]
```

**XGBoost**

```
[[ 7  1]
 [ 0 38]]
```

---

### 🧠 Interpretation

XGBoost achieved higher accuracy and MCC, indicating stronger classification performance and better handling of class imbalance.

Gradient Boosting achieved slightly higher ROC-AUC, suggesting better probability separation between classes.

This demonstrates a trade-off between classification accuracy and probabilistic discrimination, highlighting the importance of evaluating multiple metrics for imbalanced datasets.

---

## 📊 Visualizations

The project includes visualization for both models:

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

* Compare additional models (e.g., Random Forest, LightGBM)
* Integrate **FastAI CNN (spectrogram-based approach)**
* Expand the dataset for better generalization
* Deploy model via **FastAPI**
* Build a real-time inference system

---

## 📌 Key Takeaways

* Proper class imbalance handling is critical
* Cross-validation improves reliability
* ROC-AUC and MCC are essential for imbalanced datasets
* Ensemble models (GB & XGBoost) perform strongly on tabular biomedical data
* Voice biomarkers show strong potential for non-invasive PD detection

---

## 📬 Contact

* Email: [mirnadanisat@gmail.com](mailto:mirnadanisat@gmail.com)
* LinkedIn: https://www.linkedin.com/in/mirnadanisatandjung/
