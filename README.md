# professional-version-of-Credit-Card-Fraud-Detection-project
🚀 Credit Card Fraud Detection

A Machine Learning project to detect fraudulent credit card transactions using supervised learning techniques.

📌 Overview

Credit card fraud is a major financial problem globally. This project builds a machine learning model that can identify fraudulent transactions with high recall and ROC-AUC performance.

The dataset is taken from Kaggle, containing anonymized transaction features (V1–V28), transaction time, amount, and the target column Class (0 = normal, 1 = fraud).

🎯 Project Goals

Detect fraudulent transactions accurately

Handle severe class imbalance

Build a model suitable for real-world use

Evaluate performance using proper metrics

🧠 Key Features

✔ Stratified train-test split
✔ Data scaling using StandardScaler
✔ SMOTE oversampling for imbalance
✔ Multiple ML models (Logistic Regression, Random Forest, Gradient Boosting)
✔ Evaluation using precision, recall, F1-score, and ROC-AUC
✔ Confusion matrix visualization
✔ Final model saved using joblib

📂 Dataset

Source: Kaggle Credit Card Fraud Detection dataset

Rows: 284,807

Fraud Percentage: ~0.17%

Target Column: Class

🛠️ Technologies Used

Python

Pandas, NumPy

scikit-learn

imbalanced-learn (SMOTE)

Matplotlib & Seaborn

joblib

🔧 Model Workflow
1️⃣ Load and explore data

Check shape, missing values, data balance, descriptive statistics.

2️⃣ Split data

Stratified train-test split to keep fraud ratio consistent.

3️⃣ Scaling

StandardScaler applied to numeric features.

4️⃣ Resampling (SMOTE)

Oversampling applied only to training data to balance fraud class.

5️⃣ Model Training

Models tested:

Logistic Regression

Random Forest

Gradient Boosting

6️⃣ Model Evaluation

Metrics:

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

7️⃣ Save Model

Saved using joblib.dump() for deployment.

📊 Results Summary

Random Forest performed the best.

High recall means the model catches most fraud cases.

Strong ROC-AUC, indicating excellent separation between classes.

SMOTE improved the detection of minority fraud cases.

🏁 Conclusion

This project successfully detects fraudulent transactions using machine learning.
It handles imbalanced data effectively and achieves strong real-world performance metrics.

Future improvements:

Hyperparameter tuning

Trying advanced models (XGBoost, LightGBM)

Real-time prediction API

Deployment using Flask/FastAPI

📎 Project Structure
├── credit_card_fraud_detection.py
├── README.md
└── model.pkl   (saved model)

❤️ Acknowledgement

Dataset provided by Kaggle.
Built by Vijay as part of improving ML project skills
