"""Pipeline inference helper.

This module supersedes the legacy **predict.py** script and offers a function
``predict`` that can be imported programmatically *and* a CLI entry-point::

    python -m nids.pipelines.predict data.csv

The model artefacts must already exist under ``trained_model_files`` – they can
be produced via ``nids.pipelines.train``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

__all__: list[str] = ["predict"]


def _load_artefacts() -> tuple[Any, Any]:
    """Return (pipeline, label_encoder) loaded from the *trained_model_files* dir."""
    cwd = os.getcwd()
    model_dir = Path(cwd) / "trained_model_files"
    pipeline_path = model_dir / "rf_pipeline.joblib"
    encoder_path = model_dir / "label_encoder.joblib"

    if not pipeline_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            "Trained artefacts not found. Run `nids.pipelines.train` first."
        )

    pipeline = joblib.load(pipeline_path)
    label_encoder = joblib.load(encoder_path)
    return pipeline, label_encoder


def predict(data_path: str) -> None:  # noqa: D401
    """Predict network flow labels for the CSV at *data_path* and print a summary."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    print(f"Loading data from: {data_path}")
    new_data = pd.read_csv(data_path)
    print(f"Data loaded successfully: {len(new_data)} rows.")

    new_data.columns = new_data.columns.str.strip()
    new_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    pipeline, label_encoder = _load_artefacts()

    # Ensure the dataframe contains all columns expected by the pipeline
    expected_cols = pipeline.named_steps["preprocessor"].feature_names_in_.tolist()
    missing_cols = [col for col in expected_cols if col not in new_data.columns]
    if missing_cols:
        print(f"Warning: {len(missing_cols)} expected columns missing in input. Filling with 0.")
        for col in missing_cols:
            new_data[col] = 0

    # Reorder columns and fill holes
    new_data = new_data[expected_cols]

    print("Making predictions …")
    preds_numeric = pipeline.predict(new_data)
    preds_labels = label_encoder.inverse_transform(preds_numeric)

    total = len(preds_labels)
    counts = pd.Series(preds_labels).value_counts()
    benign = counts.get("BENIGN", 0)
    attacks = total - benign

    print("\n--- Prediction Summary ---")
    print(f"Total Flows: {total}")
    print(f"BENIGN: {benign} ({benign / total * 100:.2f}%)")
    print(f"ATTACK: {attacks} ({attacks / total * 100:.2f}%)")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Predict network traffic labels using a trained pipeline."
    )
    parser.add_argument(
        "data_path", type=str, help="Path to the CSV file containing network traffic data."
    )
    args = parser.parse_args()
    predict(args.data_path) 