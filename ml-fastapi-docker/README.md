
# German Credit ML API

## Project Description

This project demonstrates the deployment of a Machine Learning model
using FastAPI and Docker.

The model uses Logistic Regression with preprocessing Pipeline
to predict credit classification based on customer information.

## Technologies

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- FastAPI
- Uvicorn
- Docker

## Project Structure

ml-fastapi-docker/

- main.py
- model.joblib
- requirements.txt
- Dockerfile
- README.md

## Run Locally

Install dependencies:

pip install -r requirements.txt

Start the API:

uvicorn main:app --reload

Open documentation:

http://127.0.0.1:8000/docs

## Run with Docker

Build Docker image:

docker build -t german-credit-api .

Run Docker container:

docker run -d -p 8000:8000 --name german-credit-container german-credit-api

Open API:

http://127.0.0.1:8000/

Swagger documentation:

http://127.0.0.1:8000/docs

## API Endpoints

### GET /

Returns a message confirming that the API is running.

### POST /predict

Accepts customer information and returns the model prediction.
