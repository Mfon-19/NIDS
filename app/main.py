from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import os
import joblib
import traceback
import io
import subprocess
from nids.pipelines import train as train_pipeline

# Paths
CWD = os.getcwd()
MODEL_DIR = os.path.join(CWD, "trained_model_files")
PIPELINE_PATH = os.path.join(MODEL_DIR, "rf_pipeline.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# Make sure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(title="NIDS RandomForest API", version="0.1.0")


class FlowsRequest(BaseModel):
    data: List[Dict[str, Any]]


# Utility functions

def ensure_artifacts():
    """Ensure the trained pipeline and encoder exist, else trigger training."""
    if os.path.exists(PIPELINE_PATH) and os.path.exists(ENCODER_PATH):
        return
    # Run training script
    try:
        train_pipeline.main()
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to train pipeline: {e}")


def load_artifacts():
    ensure_artifacts()
    pipeline = joblib.load(PIPELINE_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return pipeline, label_encoder


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_json(req: FlowsRequest):
    try:
        pipeline, label_encoder = load_artifacts()
        df = pd.DataFrame(req.data)

        # Column order / missing handling
        expected_cols = pipeline.named_steps["preprocessor"].feature_names_in_.tolist()
        missing_cols = [col for col in expected_cols if col not in df.columns]
        for col in missing_cols:
            df[col] = 0
        df = df[expected_cols]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        preds_numeric = pipeline.predict(df)
        probs = pipeline.predict_proba(df)
        preds_labels = label_encoder.inverse_transform(preds_numeric)
        prob_labels = label_encoder.classes_.tolist()

        response = [
            {
                "prediction": pred,
                "probabilities": {
                    prob_labels[i]: float(prob) for i, prob in enumerate(sample_probs)
                },
            }
            for pred, sample_probs in zip(preds_labels, probs)
        ]
        return {"results": response}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        pipeline, label_encoder = load_artifacts()

        expected_cols = pipeline.named_steps["preprocessor"].feature_names_in_.tolist()
        missing_cols = [col for col in expected_cols if col not in df.columns]
        for col in missing_cols:
            df[col] = 0
        df = df[expected_cols]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        preds_numeric = pipeline.predict(df)
        probs = pipeline.predict_proba(df)
        preds_labels = label_encoder.inverse_transform(preds_numeric)
        prob_labels = label_encoder.classes_.tolist()

        response = [
            {
                "prediction": pred,
                "probabilities": {
                    prob_labels[i]: float(prob) for i, prob in enumerate(sample_probs)
                },
            }
            for pred, sample_probs in zip(preds_labels, probs)
        ]
        return {"results": response}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) 