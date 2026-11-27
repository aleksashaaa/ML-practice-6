
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import os
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

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
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "GermanCreditModel"

mlflow.set_tracking_uri(TRACKING_URI)
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
    mlflow.log_artifact("model.joblib", artifact_path="saved_model")

    # Log and register model (a new version is created on every run)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(3),
        registered_model_name=MODEL_NAME,
    )

# Point the "champion" alias at the newest version so the API knows what to serve
client = MlflowClient()
latest = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
client.set_registered_model_alias(MODEL_NAME, "champion", str(latest))

print("\nMLflow tracking completed!")
print(f"Model registered as {MODEL_NAME} v{latest} (alias: champion)")