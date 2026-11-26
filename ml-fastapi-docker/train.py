
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load dataset
column_names = [
    "checking_account", "duration_months", "credit_history", "purpose", "credit_amount", "savings_account",
    "employment_since", "installment_rate", "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits", "job", "maintenance_for", "telephone", "foreign_worker", "target"
]

df = pd.read_csv('german.data', sep=r"\s+", header=None, names=column_names)

# Original 1 = good, 2 = bad
# Business target 0 = good, 1 = bad
df["target"] = df["target"].map({1: 0, 2: 1})

# 2. Split features and target
X = df.drop(columns="target")
y = df["target"]

# 3. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Identify numerical and categorical columns
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# 5. Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, num_cols),
    ("categorical", categorical_pipeline, cat_cols)
])

# 6. Create Logistic Regression pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

# 7. Train model
model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)

print("Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save trained model
joblib.dump(model, "ml-fastapi-docker/model.joblib")
print("Model saved successfully!")
