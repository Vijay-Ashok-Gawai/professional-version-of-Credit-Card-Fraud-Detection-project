# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
from imblearn.over_sampling import SMOTE

# %% Load data
DATA_PATH = r"C:\Users\Dell\Downloads\creditcard.csv.zip"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
df['Hours'] = np.floor(df['Time'] / 3600)

# %% Train/Test Split
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# %% Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# %% SMOTE
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# %% Evaluation function
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    metrics = {
        "train_acc": accuracy_score(y_train, model.predict(X_train)),
        "test_acc": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_test, y_pred_test, zero_division=0),
        "f1": f1_score(y_test, y_pred_test, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba_test) if y_proba_test is not None else None,
        "conf_matrix": confusion_matrix(y_test, y_pred_test)
    }
    print(f"\n=== {name} ===")
    print(f"Train Acc: {metrics['train_acc']:.4f} | Test Acc: {metrics['test_acc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Confusion Matrix:\n", metrics['conf_matrix'])
    return metrics

# %% Models
log_model = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)

log_metrics = evaluate_model(log_model, X_train_res, y_train_res, X_test_scaled, y_test, "LogisticRegression")
rf_metrics = evaluate_model(rf_model, X_train_res, y_train_res, X_test_scaled, y_test, "RandomForest")

# %% Select & Save Best Model
best_model_name = "log" if log_metrics["f1"] > rf_metrics["f1"] else "rf"
best_model = log_model if best_model_name == "log" else rf_model
joblib.dump({"model": best_model, "scaler": scaler, "features": X_train.columns.tolist()}, "best_fraud_model.joblib")
print(f"\nBest model: {best_model_name} | Saved as 'best_fraud_model.joblib'")

# %% Confusion Matrix Heatmap
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nDONE: Models trained & evaluated. Metrics printed above. Model saved as 'best_fraud_model.joblib'.")

