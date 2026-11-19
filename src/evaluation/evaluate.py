"""Model evaluation module for PhishingShield.

Computes classification metrics for phishing detection (label 1 = Phishing,
label 0 = Legitimate), outputs the standardized evaluation report JSON, and provides
the compute_metrics function consumed by the training pipeline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.extractor import FEATURES

DEFAULT_MODEL_PATH = "artifacts/model/model.joblib"
DEFAULT_METADATA_PATH = "artifacts/model/model_metadata.json"
DEFAULT_TEST_PATH = "data/processed/test.csv"
DEFAULT_OUTPUT_PATH = "artifacts/reports/evaluation.json"


def compute_metrics(y_true: Any, y_pred: Any, y_proba: Any) -> dict[str, float]:
    """Compute performance metrics for the phishing class (positive label = 1).

    This function is imported by the training pipeline (train.py) and the evaluation
    CLI.

    Metrics computed:
        - accuracy: (TP + TN) / (TP + TN + FP + FN)
        - precision: TP / (TP + FP) for positive label 1
        - recall: TP / (TP + FN) for positive label 1
        - f1: Harmonic mean of precision and recall
        - roc_auc: Area under ROC curve computed from P(phishing) probabilities
        - false_positive_rate: FP / (FP + TN)

    Args:
        y_true: Ground truth binary labels (0 = legitimate, 1 = phishing).
        y_pred: Predicted binary labels (0 = legitimate, 1 = phishing).
        y_proba: Predicted probabilities for phishing class (label 1).

    Returns:
        dict[str, float]: Standardized dictionary containing scalar numeric metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=int).ravel()

    # Handle 1D or 2D probability inputs
    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        if y_proba_arr.ndim == 2:
            if y_proba_arr.shape[1] >= 2:
                y_proba_arr = y_proba_arr[:, 1]
            else:
                y_proba_arr = y_proba_arr.ravel()
        else:
            y_proba_arr = y_proba_arr.ravel()
    else:
        y_proba_arr = None

    # Compute confusion matrix with explicit labels [0, 1]
    # tn: legit correctly identified, fp: legit incorrectly marked as phish
    # fn: phish missed as legit, tp: phish correctly identified
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    precision = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0.0))
    recall = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0.0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0.0))

    denom_fpr = fp + tn
    false_positive_rate = float(fp / denom_fpr) if denom_fpr > 0 else 0.0

    # ROC-AUC calculation requires at least two classes present in y_true
    if y_proba_arr is not None and len(np.unique(y_true_arr)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
        except ValueError:
            roc_auc = 0.0
    else:
        roc_auc = 0.0

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
        "false_positive_rate": round(false_positive_rate, 6),
    }


def evaluate_model(
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    test_path: str = DEFAULT_TEST_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Load model artifact and test dataset, evaluate metrics, and save evaluation.json."""
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at '{model_path}'.")

    t_path = Path(test_path)
    if not t_path.exists():
        raise FileNotFoundError(f"Test split dataset not found at '{test_path}'.")

    # Load test split
    test_df = pd.read_csv(t_path)
    if "label" not in test_df.columns:
        raise ValueError(f"Label column missing from '{test_path}'.")

    missing_features = [f for f in FEATURES if f not in test_df.columns]
    if missing_features:
        raise ValueError(f"Features missing from '{test_path}': {missing_features}")

    X_test = test_df[FEATURES]
    y_test = test_df["label"].astype(int).values

    # Load model
    model = joblib.load(m_path)

    # Load metadata if available
    meta_path = Path(metadata_path)
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    model_name = metadata.get("model_name", type(model).__name__)
    model_version = str(metadata.get("version", "1"))
    threshold = float(metadata.get("threshold", 0.5))

    # Obtain class 1 probabilities
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)
        if hasattr(model, "classes_") and 1 in model.classes_:
            class_idx = list(model.classes_).index(1)
            y_proba = probas[:, class_idx]
        elif probas.shape[1] >= 2:
            y_proba = probas[:, 1]
        else:
            y_proba = probas.ravel()
    else:
        # Fallback for models without predict_proba
        y_proba = model.predict(X_test).astype(float)

    y_pred = (y_proba >= threshold).astype(int)

    # Compute standardized metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Confusion matrix elements
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Classification report as dict
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["legitimate", "phishing"],
        output_dict=True,
        zero_division=0.0,
    )

    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    report_payload = {
        "model_name": model_name,
        "model_version": model_version,
        "evaluated_at": now_utc,
        "test_rows": int(len(test_df)),
        "metrics": metrics,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": report_dict,
    }

    # Write output report JSON
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"\nEvaluation completed for {model_name} (version {model_version}) on {len(test_df)} test samples.")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Report written to: {output_path}\n")

    return report_payload


def main(argv: list[str] | None = None) -> None:
    """CLI runner for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate PhishingShield model performance.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to model artifact.")
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH, help="Path to model metadata.")
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH, help="Path to test CSV.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Path to save evaluation JSON.")
    args = parser.parse_args(argv)

    evaluate_model(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        test_path=args.test_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
