"""FastAPI service for German credit risk prediction.

The endpoint accepts human-readable category values (keys, labels or the legacy
UCI codes) and returns the prediction together with a probability and a
per-feature explanation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
DATA_PATH = BASE_DIR / "german.data"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="German Credit ML API",
    version="2.0.0",
    description=(
        "Credit risk scoring for the UCI Statlog German Credit dataset. "
        "Categorical fields accept human-readable keys such as "
        "`healthy_balance` or `past_delays`; legacy UCI codes like `A13` "
        "are still supported."
    ),
)


def _categorical_field(feature_name: str) -> Any:
    feature = features.FEATURES_BY_NAME[feature_name]
    keys = [category.key for category in feature.categories]
    return Field(
        ...,
        description=f"{feature.label}. One of: {', '.join(keys)}.",
        examples=[keys[0]],
    )


class CreditInput(BaseModel):
    """Applicant profile accepted by the model."""

    model_config = ConfigDict(extra="forbid")

    checking_account: str = _categorical_field("checking_account")
    duration_months: int = Field(
        ..., ge=4, le=72, description="Loan duration in months.", examples=[24]
    )
    credit_history: str = _categorical_field("credit_history")
    purpose: str = _categorical_field("purpose")
    credit_amount: int = Field(
        ..., ge=250, le=20000, description="Credit amount in DM.", examples=[3000]
    )
    savings_account: str = _categorical_field("savings_account")
    employment_since: str = _categorical_field("employment_since")
    installment_rate: int = Field(
        ..., ge=1, le=4, description="Installment rate level (1 = lowest)."
    )
    personal_status_sex: str = _categorical_field("personal_status_sex")
    other_debtors: str = _categorical_field("other_debtors")
    residence_since: int = Field(
        ..., ge=1, le=4, description="Years at the current address (capped at 4)."
    )
    property: str = _categorical_field("property")
    age: int = Field(..., ge=18, le=75, description="Age in years.", examples=[30])
    other_installment_plans: str = _categorical_field("other_installment_plans")
    housing: str = _categorical_field("housing")
    existing_credits: int = Field(
        ..., ge=1, le=4, description="Existing credits at this bank."
    )
    job: str = _categorical_field("job")
    maintenance_for: int = Field(
        ..., ge=1, le=2, description="Number of people to support."
    )
    telephone: str = _categorical_field("telephone")
    foreign_worker: str = _categorical_field("foreign_worker")

    @field_validator(*features.CATEGORICAL_NAMES, mode="before")
    @classmethod
    def _normalize_category(cls, value: Any, info: ValidationInfo) -> str:
        """Accept a friendly key, a human label or a legacy UCI code."""

        return features.resolve(info.field_name, value)


class Factor(BaseModel):
    """Contribution of one input to the predicted risk."""

    feature: str
    label: str
    value: str
    contribution: float
    direction: Literal["increases", "decreases"]
    strength: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = good credit risk, 1 = bad.")
    label: str
    risk_probability: float = Field(..., description="Probability of bad credit.")
    confidence: float = Field(..., description="Probability of the predicted class.")
    explanations: list[Factor] = Field(
        ..., description="Feature contributions, strongest first."
    )


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=features.COLUMN_NAMES)
    # Original file: 1 = good, 2 = bad. Business target: 0 = good, 1 = bad.
    df["target"] = df["target"].map({1: 0, 2: 1})
    return df


@lru_cache(maxsize=1)
def compute_metrics() -> dict[str, Any]:
    """Evaluate the saved model on the same 20% hold-out split used in training."""

    df = _load_dataset()
    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    return {
        "model": "Logistic Regression",
        "dataset": {
            "name": "UCI Statlog (German Credit Data)",
            "total": int(len(df)),
            "good": int((y == 0).sum()),
            "bad": int((y == 1).sum()),
        },
        "split": {"train": int(len(X_train)), "test": int(len(X_test))},
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        },
        "confusion_matrix": {
            "true_good_pred_good": int(tn),
            "true_good_pred_bad": int(fp),
            "true_bad_pred_good": int(fn),
            "true_bad_pred_bad": int(tp),
        },
    }


def _base_feature(transformed_name: str) -> str:
    """Map ``categorical__checking_account_A11`` back to ``checking_account``."""

    _, raw_name = transformed_name.split("__", 1)
    if raw_name in features.NUMERIC_NAMES:
        return raw_name
    for name in sorted(features.CATEGORICAL_NAMES, key=len, reverse=True):
        if raw_name.startswith(f"{name}_"):
            return name
    return raw_name


def _explain(row: dict[str, Any], payload: dict[str, Any]) -> list[Factor]:
    """Split the linear model decision into per-feature contributions."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    frame = pd.DataFrame([row], columns=list(features.FEATURES_BY_NAME))
    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    values = transformed[0]

    contributions: dict[str, float] = {}
    for name, value, coefficient in zip(
        preprocessor.get_feature_names_out(), values, classifier.coef_[0]
    ):
        base = _base_feature(str(name))
        contributions[base] = contributions.get(base, 0.0) + float(value * coefficient)

    strongest = max((abs(value) for value in contributions.values()), default=0.0) or 1.0

    factors: list[Factor] = []
    for feature_name, contribution in contributions.items():
        feature = features.FEATURES_BY_NAME[feature_name]
        factors.append(
            Factor(
                feature=feature_name,
                label=feature.label,
                value=features.display_value(feature_name, payload[feature_name]),
                contribution=round(contribution, 4),
                direction="increases" if contribution >= 0 else "decreases",
                strength=round(min(abs(contribution) / strongest, 1.0), 4),
            )
        )
    factors.sort(key=lambda factor: abs(factor.contribution), reverse=True)
    return factors


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "ML API is running", "docs": "/docs", "schema": "/features"}


@app.get("/features")
def get_features() -> dict[str, Any]:
    """Human-readable description of every model input."""

    return {"features": features.schema()}


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """Model quality on the held-out test split."""

    return compute_metrics()


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CreditInput) -> PredictionResponse:
    payload = data.model_dump()
    row = features.to_model_row(payload)

    frame = pd.DataFrame([row], columns=list(features.FEATURES_BY_NAME))
    probabilities = model.predict_proba(frame)[0]
    risk_probability = float(probabilities[1])
    prediction = int(risk_probability >= 0.5)

    return PredictionResponse(
        prediction=prediction,
        label="High credit risk" if prediction == 1 else "Low credit risk",
        risk_probability=round(risk_probability, 4),
        confidence=round(float(probabilities[prediction]), 4),
        explanations=_explain(row, payload),
    )
