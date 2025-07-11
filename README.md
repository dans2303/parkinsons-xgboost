# Parkinson's Disease Detection Using XGBoost

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green.svg)](https://xgboost.readthedocs.io/)

This project detects Parkinson's disease from voice measurements using XGBoost and includes preprocessing, class balancing (SMOTETomek), bootstrapping, tuning, and evaluation.

---

## Dataset

- **Public Dataset**:  
  [UCI ML Repository - Parkinson’s](https://archive.ics.uci.edu/ml/datasets/parkinsons)  
  A commonly used dataset with 195 voice recordings and 22 biomedical voice features.

- **Institutional Dataset**:  
  An additional dataset was used from a research project titled  
  _“Develop integrated diagnosis and prediction model of Parkinson's disease”_  
  conducted at **TMU Shuang-Ho Hospital**.  
  This dataset was approved under TMU-JIRB Protocol No. **N201801043**.  
  **Note**: Due to privacy policies and IRB restrictions, this dataset is not publicly available.

- **Target Variable**:  
  `status` — 1 indicates Parkinson’s disease, 0 indicates healthy control.

---

## Pipeline Overview

- Data cleaning and type filtering
- SMOTETomek class balancing
- Bootstrapping (optional)
- Feature scaling
- Model tuning via GridSearchCV
- Evaluation metrics and visualization

---

## How to Run

1. Clone the repo and navigate into it
2. Activate your conda environment:
