# 🏦 Financial Risk Assessment Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Project Overview

This project is an AI-powered financial risk assessment tool designed to streamline the loan underwriting process. By leveraging machine learning on a dataset of **400,000 financial records**, the platform automates decision-making to:

1.  **Predict Loan Eligibility (Classification):** Determines if an applicant is _Eligible_, _High Risk_, or _Not Eligible_.
2.  **Calculate Max Safe EMI (Regression):** Predicts the maximum monthly EMI an applicant can safely afford based on their financial profile.

The system features a **dual-model architecture** (XGBoost & Random Forest), **MLflow** for experiment tracking, and a production-ready **Streamlit** dashboard for real-time inference.

---

## 📸 Application Demo

### 1. Real-Time Risk Assessment

_Enter applicant details and get instant eligibility status + max safe EMI limit._

### 2. Interactive Analytics

_Visual insights into loan portfolios, risk distribution, and income patterns._

---

## 🛠️ Tech Stack

| Component               | Tools Used                           |
| :---------------------- | :----------------------------------- |
| **Frontend**            | Streamlit, Plotly, Seaborn           |
| **Machine Learning**    | Scikit-Learn, XGBoost, Random Forest |
| **Experiment Tracking** | MLflow                               |
| **Data Processing**     | Pandas, NumPy                        |
| **Deployment**          | Streamlit Cloud / Local Host         |

---

## 📂 Project Structure

```text
Financial_Risk_Platform/
│
├── src/
│   ├── preprocessing.py   # Feature Engineering & Cleaning Pipeline
│   ├── train.py           # Model Training Script with MLflow Logging
│   ├── eda.py             # Exploratory Data Analysis & Visualization
│   └── generate_data.py   # Script to generate the synthetic 400k dataset
│
├── models/                # Saved models (PKL files)
├── mlruns/                # MLflow experiment logs
├── app.py                 # Main Streamlit Application
├── requirements.txt       # Project Dependencies
└── README.md              # Documentation


🚀 How to Run Locally

Clone the Repository

git clone [https://github.com/SanjaiSK13/FINANCIAL_RISK_PLATFORM.git](https://github.com/SanjaiSK13/FINANCIAL_RISK_PLATFORM.git)

cd FINANCIAL_RISK_PLATFORM

2. Install Dependencies

pip install -r requirements.txt

3. Train the Models

Run the training pipeline to train the models and save the best ones to the models/ folder.

python src/train.py

4.Launch the Web App

streamlit run app.py


💡 Key Features

Dual-Pipeline Prediction: Runs both classification and regression models simultaneously for a complete risk profile.

Smart Feature Engineering: Automatically calculates Debt-to-Income (DTI) ratio, disposable income, and EMI coverage ratios.

Safety Meter: A visual gauge chart that instantly shows if the requested loan amount is within the safe limit.
```
