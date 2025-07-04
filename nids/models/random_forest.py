"""Standalone Random-Forest model training helper.

This module provides the training logic that was previously inside the root-level
`RandomForestClassifier.py` script.  It is kept self-contained on purpose so that
experiments can be run without touching the full pipeline.

Example
-------
>>> from nids.models.random_forest import train_standalone
>>> train_standalone("./archive")
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

__all__ = [
    "train_standalone",
]


def _load_csv_concat(data_dir: str | os.PathLike[str]) -> pd.DataFrame:
    """Read all CSV files inside *data_dir* into a single DataFrame."""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in directory: {data_dir!s}. Is the path correct?"
        )

    _LOGGER.info("Loading %d CSV files …", len(csv_files))
    # Using a simple list comprehension here – dask is overkill for local experiments.
    df = pd.concat((pd.read_csv(fp) for fp in csv_files), ignore_index=True)
    _LOGGER.info("Dataset shape after concat: %s", df.shape)
    return df


def train_standalone(data_dir: str | os.PathLike[str], *, random_state: int = 42) -> None:
    """Train a Random-Forest on *data_dir* and persist artefacts under *trained_model_files*."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    df = _load_csv_concat(str(data_dir))

    # Basic cleaning -----------------------------------------------------------------
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    target_col: Final[str] = "Label"
    if target_col not in df.columns:
        raise KeyError(f"Missing required target column: {target_col}")

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Encode labels ------------------------------------------------------------------
    lbl = LabelEncoder()
    y = lbl.fit_transform(y_raw)

    num_cols = X.select_dtypes(include=np.number).columns
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    # Impute & scale numeric features only ------------------------------------------
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # Fit on train numeric columns
    X_train_num = X_train[num_cols]
    X_test_num = X_test[num_cols]

    X_train_num = imputer.fit_transform(X_train_num)
    X_train_num = scaler.fit_transform(X_train_num)

    X_test_num = imputer.transform(X_test_num)
    X_test_num = scaler.transform(X_test_num)

    X_train_processed = pd.DataFrame(X_train_num, columns=num_cols, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_num, columns=num_cols, index=X_test.index)

    # Train model --------------------------------------------------------------------
    model = _RFC(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train_processed, y_train)

    # Evaluate -----------------------------------------------------------------------
    y_pred = model.predict(X_test_processed)
    acc = accuracy_score(y_test, y_pred)
    _LOGGER.info("Accuracy on hold-out: %.4f", acc)
    _LOGGER.debug("\n%s", classification_report(y_test, y_pred, target_names=lbl.classes_))
    _LOGGER.debug("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    # Persist artefacts --------------------------------------------------------------
    out_dir = Path("trained_model_files"); out_dir.mkdir(exist_ok=True)
    joblib.dump(model, out_dir / "random_forest_model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(lbl, out_dir / "label_encoder.joblib")
    _LOGGER.info("Saved artefacts to %s", out_dir)


if __name__ == "__main__":  # pragma: no cover – CLI helper
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train a standalone RandomForest on CIC-IDS2017 CSV files."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=os.getenv("DATASET_PATH", "data/raw"),
        help="Directory containing CSVs (default: env DATASET_PATH or ./archive)",
    )
    cli_args = parser.parse_args()

    train_standalone(cli_args.data_dir) 