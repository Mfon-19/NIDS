"""Random-Forest Training Pipeline for the NIDS project.

This module is a direct refactor of the former **train_pipeline.py** script and
lives inside the package namespace so it can be imported without relying on
`PYTHONPATH` tricks.  It can also still be used as a CLI:

    python -m nids.pipelines.train --data-dir ./archive

Functions
---------
load_dataset            – Efficiently read multiple CSVs via Dask
build_pipeline          – Build preprocessing + RandomForest pipeline
hyperparameter_search   – Randomised search for best hyper-parameters
evaluate_model          – Evaluation metrics, plots, SHAP etc.
main                    – CLI entry-point
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import dask.dataframe as dd

__all__: list[str] = [
    "load_dataset",
    "build_pipeline",
    "hyperparameter_search",
    "evaluate_model",
    "main",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load all CSV files in *dataset_path* into a single pandas DataFrame."""
    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in directory: {dataset_path}. Is the path correct?"
        )

    print(f"[Data] Reading {len(csv_files)} CSV files with Dask …")
    ddf = dd.read_csv(csv_files, assume_missing=True)
    ddf = ddf.replace([np.inf, -np.inf], np.nan)
    df = ddf.compute()
    print(f"[Data] Loaded dataset with shape: {df.shape}")
    return df

# ---------------------------------------------------------------------------
# Pipeline & Hyper-parameters
# ---------------------------------------------------------------------------

def build_pipeline(numeric_features: list[str]) -> Pipeline:
    """Return a preprocessing + RandomForest pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
    ])

    clf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def hyperparameter_search(pipe: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    """Perform a RandomisedSearchCV and return the best estimator."""
    param_dist = {
        "classifier__n_estimators": [100, 200, 400, 800],
        "classifier__max_depth": [None, 10, 30, 50],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.3, 0.5],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=8,
        cv=2,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    search.fit(X_train, y_train)
    print(f"[Tuning] Best params: {search.best_params_}")
    return search.best_estimator_

# ---------------------------------------------------------------------------
# Evaluation / Interpretation helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    report_dir: str | os.PathLike[str],
) -> None:
    """Generate evaluation metrics, plots, and a minimal HTML report."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    # Accuracy & classification report
    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test_labels, y_pred_labels)
    print(f"[Eval] Accuracy: {acc:.4f}\n")
    print(cls_report)

    # Confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    cm_path = report_dir / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # ROC curve (macro-average) – may fail for multi-class situations
    roc_path: str | None = None
    try:
        RocCurveDisplay.from_estimator(model, X_test, label_encoder.transform(y_test_labels))
        roc_path = report_dir / "roc_curve.png"
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover – not critical
        print(f"[Eval] Skipping ROC curve: {exc}")

    # Permutation importance
    print("[Interpret] Computing permutation importance …")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    importances = pd.Series(result.importances_mean, index=model.feature_names_in_)
    imp_path = report_dir / "permutation_importance.png"
    importances.sort_values(ascending=False)[:20].plot.barh(figsize=(8, 10))
    plt.title("Top-20 Permutation Importances")
    plt.gca().invert_yaxis()
    plt.savefig(imp_path, bbox_inches="tight")
    plt.close()

    # SHAP summary
    try:
        print("[Interpret] Computing SHAP values (may take a while) …")
        explainer = shap.TreeExplainer(model.named_steps["classifier"])
        shap_values = explainer.shap_values(model.named_steps["preprocessor"].transform(X_test))
        shap.summary_plot(
            shap_values,
            model.named_steps["preprocessor"].transform(X_test),
            show=False,
        )
        shap_path = report_dir / "shap_summary.png"
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[Interpret] SHAP computation skipped: {exc}")

    # Minimal HTML report
    html_report = report_dir / "report.html"
    with open(html_report, "w", encoding="utf-8") as fp:
        fp.write("<h1>NIDS – Random Forest Model Report</h1>\n")
        fp.write(f"<p><strong>Run:</strong> {datetime.utcnow().isoformat()} UTC</p>\n")
        fp.write(f"<p><strong>Accuracy:</strong> {acc:.4f}</p>\n")
        fp.write("<h2>Classification Report</h2><pre>" + cls_report + "</pre>")
        for img in [cm_path, roc_path, imp_path, report_dir / "shap_summary.png"]:
            if img and Path(img).exists():
                fp.write(f"<h2>{Path(img).stem.replace('_', ' ').title()}</h2>")
                fp.write(f"<img src='{Path(img).name}' style='max-width:100%;'>\n")

    print(f"[Eval] Report generated at {html_report}")

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: D401
    """Train the pipeline via CLI or programmatic call."""
    parser = argparse.ArgumentParser(description="Train the Random-Forest NIDS pipeline")
    default_data_dir = os.getenv("DATASET_PATH", os.path.join(os.getcwd(), "data", "raw"))
    parser.add_argument(
        "--data-dir",
        default=default_data_dir,
        help="Directory containing CIC-IDS2017 CSV files (default: env DATASET_PATH or ./data/raw)",
    )
    args = parser.parse_args(argv)

    dataset_path = args.data_dir
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Provided data directory '{dataset_path}' does not exist.")

    df = load_dataset(dataset_path)
    df.columns = df.columns.str.strip()

    target_col: Final[str] = "Label"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Normalise labels
    df[target_col] = df[target_col].replace({"No Label": "BENIGN"}).fillna("ATTACK")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    pipeline = build_pipeline(numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.3,
        random_state=42,
        stratify=y_enc,
    )

    best_model = hyperparameter_search(pipeline, X_train, y_train)

    # Persist pipeline & encoder
    output_dir = Path("trained_model_files"); output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "rf_pipeline.joblib"
    encoder_path = output_dir / "label_encoder.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"[Save] Pipeline saved to {model_path}")
    print(f"[Save] LabelEncoder saved to {encoder_path}")

    # Evaluate / interpret
    report_dir = Path("reports") / datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    evaluate_model(best_model, X_test, y_test, label_encoder, report_dir)


if __name__ == "__main__":  # pragma: no cover
    main() 