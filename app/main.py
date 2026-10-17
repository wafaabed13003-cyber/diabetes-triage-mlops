"""Prediction API for the Virtual Diabetes Clinic triage service.

Endpoints:
    GET  /health   -> {"status": "ok", "model_version": "v0.1"}
    POST /predict  -> {"prediction": <float>}

The model file and version are chosen with environment variables,
so the same code serves v0.1 and v0.2:
    MODEL_PATH     (default: models/model_v01.joblib)
    MODEL_VERSION  (default: v0.1)
"""

import os

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_v01.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

# Same order as the columns used during training
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

# Load the model once, when the service starts (not at every request)
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Diabetes progression API", version=MODEL_VERSION)


class PatientFeatures(BaseModel):
    """The 10 features of one patient. All are required numbers."""

    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class Prediction(BaseModel):
    prediction: float


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return a clear JSON error (HTTP 422) when the input is invalid."""
    details = [
        {
            "field": str(err["loc"][-1]) if err["loc"] else "body",
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid input", "details": details},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientFeatures):
    # One-row table with the columns in the training order
    row = pd.DataFrame([patient.model_dump()], columns=FEATURES)
    value = float(model.predict(row)[0])
    return {"prediction": value}
