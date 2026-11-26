# German Credit ML API + Credit Risk Studio

Credit risk scoring for the [UCI Statlog German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
served by FastAPI and wrapped in a four-step Streamlit interface.

The model is a `LogisticRegression` pipeline (standard scaling + one-hot encoding).
The UI accepts human-readable categories such as *"Healthy balance (200 DM or more)"*
or *"Past payment delays"* instead of the original `A13` / `A33` codes.

## What you get

- **Guided wizard** — Applicant → Loan → Finances → Obligations → Review, with a live summary.
- **Readable inputs** — every category has a plain-English label and a tooltip.
- **Risk verdict** — probability of a bad credit, confidence and a risk scale.
- **Explanation** — the strongest factors behind the decision, with direction and weight.
- **Model metrics** — accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.
- **Backwards compatible API** — legacy UCI codes (`A11`, `A202`, …) still work.

## Project structure

```
ml-fastapi-docker/
├── app.py                 # Streamlit front-end (Credit Risk Studio)
├── main.py                # FastAPI service
├── features.py            # Single source of truth: labels, codes, ranges, defaults
├── train.py               # Training script
├── track_model.py         # MLflow tracking helper
├── model.joblib           # Trained pipeline
├── german.data            # Dataset
├── requirements.txt
├── Dockerfile
└── .streamlit/config.toml # Purple Streamlit theme
```

## Run locally

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```

In a second terminal, start the UI:

```bash
streamlit run app.py
```

- UI: <http://127.0.0.1:8501>
- Swagger docs: <http://127.0.0.1:8000/docs>

If the API runs elsewhere, point the UI at it:

```bash
set API_URL=http://127.0.0.1:8000
streamlit run app.py
```

## API

### `POST /predict`

Accepts human-readable category keys:

```json
{
  "checking_account": "healthy_balance",
  "duration_months": 24,
  "credit_history": "existing_repaid",
  "purpose": "furniture",
  "credit_amount": 3000,
  "savings_account": "100_to_500",
  "employment_since": "1_to_4y",
  "installment_rate": 2,
  "personal_status_sex": "male_single",
  "other_debtors": "none",
  "residence_since": 2,
  "property": "car_or_other",
  "age": 30,
  "other_installment_plans": "none",
  "housing": "rent",
  "existing_credits": 1,
  "job": "skilled",
  "maintenance_for": 1,
  "telephone": "registered",
  "foreign_worker": "no"
}
```

Response:

```json
{
  "prediction": 0,
  "label": "Low credit risk",
  "risk_probability": 0.0738,
  "confidence": 0.9262,
  "explanations": [
    {
      "feature": "foreign_worker",
      "label": "Foreign worker",
      "value": "No",
      "contribution": -0.5815,
      "direction": "decreases",
      "strength": 1.0
    }
  ]
}
```

`prediction` is `0` for a good credit risk and `1` for a bad one.
Positive contributions push the probability of a bad credit up.

Legacy codes are still accepted, for example `"checking_account": "A13"`.

### `GET /features`

Returns the full human-readable schema: labels, keys, legacy codes, ranges and defaults.

### `GET /metrics`

Returns the model metrics computed on the same 20% hold-out split used in training
(`random_state=42`, stratified).

### `GET /`

Health message with links to the docs and schema.

## Docker

The image serves the API:

```bash
docker build -t german-credit-api .
docker run -d -p 8000:8000 --name german-credit-container german-credit-api
```

The Streamlit UI is intended to run locally against that API.

## Dataset

Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset].
UCI Machine Learning Repository. DOI: [10.24432/C5NC77](https://doi.org/10.24432/C5NC77).
Licensed under CC BY 4.0.
