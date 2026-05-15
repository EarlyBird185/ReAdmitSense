from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, numpy as np, pandas as pd

app = FastAPI()

# Allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your domain in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load model artifacts on startup
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
with open("feature_columns.json") as f:
    feature_columns = json.load(f)


class PatientInput(BaseModel):
    age: int                    # 0–9 ordinal (e.g. [60-70) = 6)
    gender: int                 # 1=Male, 0=Female
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    number_diagnoses: int
    diag_1: str                 # e.g. "Circulatory"
    insulin: str                # "No", "Steady", "Up", "Down"
    a1c: str                    # "None", "Norm", ">7", ">8"
    max_glu_serum: str          # "None", "Norm", ">200", ">300"
    number_outpatient: int
    number_emergency: int
    number_inpatient: int


@app.post("/predict")
def predict(patient: PatientInput):
    # Build a raw dict matching the pre-encoding structure your model saw
    raw = {
        "age": patient.age,
        "gender": patient.gender,
        "time_in_hospital": patient.time_in_hospital,
        "num_lab_procedures": patient.num_lab_procedures,
        "num_procedures": patient.num_procedures,
        "number_diagnoses": patient.number_diagnoses,
        "number_outpatient": patient.number_outpatient,
        "number_emergency": patient.number_emergency,
        "number_inpatient": patient.number_inpatient,
        "diag_1": patient.diag_1,
        "diag_2": "Other",   # not collected in chatbot — use mode
        "diag_3": "Other",
        "insulin": patient.insulin,
        "A1Cresult": patient.a1c,
        "max_glu_serum": patient.max_glu_serum,
    }

    # One-hot encode categoricals to match training columns
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df)

    # Align to exact training columns (fills missing dummies with 0)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale and predict
    X = scaler.transform(df)
    prob = model.predict_proba(X)[0][1]   # probability of readmission
    label = "HIGH" if prob >= 0.6 else "MODERATE" if prob >= 0.4 else "LOW"

    return {
        "risk_label": label,
        "probability": round(float(prob), 3),
        "percentage": round(float(prob) * 100, 1)
    }


@app.get("/health")
def health():
    return {"status": "ok"}