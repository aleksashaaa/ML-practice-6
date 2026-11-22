
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="German Credit ML API")

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
model = joblib.load(MODEL_PATH)


class CreditInput(BaseModel):
    checking_account: str
    duration_months: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_account: str
    employment_since: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    residence_since: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    existing_credits: int
    job: str
    maintenance_for: int
    telephone: str
    foreign_worker: str


@app.get("/")
def home():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(data: CreditInput):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    return {"prediction": prediction}
