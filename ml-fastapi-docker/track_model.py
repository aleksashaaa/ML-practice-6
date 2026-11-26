
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Load dataset
column_names = [
    "checking_account", "duration_months", "credit_history",
    "purpose", "credit_amount", "savings_account",
    "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing",
    "existing_credits", "job", "maintenance_for",
    "telephone", "foreign_worker", "target"
]

df = pd.read_csv("german.data", sep=r"\s+", header=None, names=column_names)
df["target"] = df["target"].map({1: 0, 2: 1})

# Prepare test data
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load existing trained model
model = joblib.load("model.joblib")

# Generate predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0)
}

print("Model Evaluation:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

mlflow.set_experiment("German Credit Experiment")

# Log experiment and register model
with mlflow.start_run(run_name="Logistic Regression"):

    # Log model parameters
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("preprocessing", "StandardScaler + OneHotEncoder")

    # Log evaluation metrics
    mlflow.log_metrics(metrics)

    # Log saved model file as an artifact
    mlflow.log_artifact(
        str("model.joblib"),
        artifact_path="saved_model"
    )

    # Log and register model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="GermanCreditModel"
    )

print("\nMLflow tracking completed!")
print("Model registered as GermanCreditModel")
