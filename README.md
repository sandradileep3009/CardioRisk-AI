# ❤️ CardioRisk AI — Explainable Cardiovascular Risk Stratification Platform

## Overview

CardioRisk AI is an end-to-end machine learning platform that predicts 10-year cardiovascular disease risk using clinical and lifestyle features.

The system goes beyond traditional binary classification by generating calibrated risk probabilities, uncertainty estimates, and explainable predictions through an interactive Streamlit application.

⚠️ This project is an AI prototype for research and educational purposes, not a medical diagnosis system.

---

# Key Features

## 🧠 Machine Learning Risk Prediction

- Built a Random Forest classification model for cardiovascular risk prediction
- Implemented data preprocessing and feature scaling using Scikit-learn
- Generated probability-based risk predictions instead of only class labels


## 📊 Model Calibration

- Applied Platt Scaling using `CalibratedClassifierCV`
- Improved reliability of predicted risk probabilities
- Added calibration curve analysis to evaluate probability quality


## 🔍 Explainable AI

- Integrated SHAP for model interpretability
- Identifies feature contributions influencing cardiovascular risk predictions


## 📈 Risk Stratification

Converts model probability into actionable categories:

- Low Risk
- Medium Risk
- High Risk


## 📉 Model Evaluation

Evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- ROC-AUC Score
- ROC Curve
- Calibration Curve


## 🖥️ Streamlit Deployment

Built an interactive dashboard that provides:

- Patient risk assessment
- Risk probability
- Risk category
- Confidence interval estimation


---

# Tech Stack

**Languages**
- Python

**Machine Learning**
- Scikit-learn
- Random Forest
- Probability Calibration

**Explainability**
- SHAP

**Data Science**
- Pandas
- NumPy

**Visualization**
- Matplotlib
- Seaborn

**Deployment**
- Streamlit


---

# Project Workflow

```
Clinical Data

↓

Data Cleaning & Preprocessing

↓

Feature Scaling

↓

Random Forest Model

↓

Platt Calibration

↓

Risk Probability Prediction

↓

SHAP Explainability

↓

Streamlit Dashboard
```

---

# Project Structure

```
Heart_disease_prediction/

├── code.py
├── app.py
├── diabetes.csv
├── cardio_risk_model.pkl
├── cardio_scaler.pkl
├── requirements.txt
└── README.md
```

---


# Resume Highlight

Developed an explainable cardiovascular risk stratification system using a calibrated Random Forest model with Platt scaling, SHAP-based interpretability, probability-based risk assessment, and Streamlit deployment for AI-driven clinical analytics.

---

# Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python code.py
```

Run application:

```bash
streamlit run app.py
```

---

# Resume Highlight

Developed an explainable cardiovascular risk stratification system using a calibrated Random Forest model with Platt scaling, SHAP-based interpretability, probability-based risk assessment, and Streamlit deployment for AI-driven clinical analytics.

---

# Author

Sandra Dileep  
Computer Science Student | AI/ML Enthusiast
