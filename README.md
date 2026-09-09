# 🏦 Bank Fraud Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

> **End-to-end machine learning project for identifying potentially fraudulent banking transactions using Python.**

---

## 📌 Project Overview

Financial institutions process millions of transactions every day. Detecting fraudulent activity manually across such a large volume of transactions is extremely difficult.

This project explores how **Machine Learning classification techniques** can be applied to historical banking transaction data to identify patterns associated with fraudulent activity.

The project follows an end-to-end data science workflow:

**Raw Transaction Data → Data Exploration → Feature Engineering → Model Training → Model Evaluation → Threshold Optimisation → Fraud Prediction**

The main objective is not simply to maximise accuracy, but to build an approach that can effectively identify fraudulent transactions while controlling the number of legitimate transactions incorrectly flagged as fraud.

---

# 🎯 Business Problem

Banks need to identify potentially fraudulent transactions quickly while avoiding excessive false alerts.

A fraud detection system therefore needs to answer:

> **Can historical transaction data be used to train a machine learning model that identifies transactions with a high probability of being fraudulent?**

The challenge is particularly difficult because fraudulent transactions represent only a **very small proportion of total transactions**.

This creates a highly **imbalanced classification problem**.

---

# 📊 Dataset

The project uses a large banking transaction dataset containing approximately:

- **6.36 million transactions**
- **11 variables**
- Fraud and legitimate transaction labels
- No missing values in the analysed dataset

The target variable is:

```text
isFraud
