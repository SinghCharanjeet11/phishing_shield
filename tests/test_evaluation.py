"""Unit tests for the model evaluation module."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.evaluation.evaluate import compute_metrics, evaluate_model
from src.features.extractor import FEATURES


def test_compute_metrics_perfect_classification():
    """Test metrics calculation with 100% correct predictions."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.9, 0.95]

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_compute_metrics_all_negative_predictions():
    """Test metrics when model predicts all negative (0)."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]
    y_proba = [0.1, 0.2, 0.3, 0.4]

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["false_positive_rate"] == 0.0


def test_compute_metrics_all_positive_predictions():
    """Test metrics when model predicts all positive (1)."""
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 1, 1]
    y_proba = [0.6, 0.7, 0.8, 0.9]

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 1.0  # FP=2, TN=0 -> 2 / (2 + 0) = 1.0


def test_compute_metrics_confusion_matrix_and_fpr():
    """Test specific confusion matrix counts and false positive rate."""
    # TN=2, FP=1, FN=1, TP=2
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.3, 0.85, 0.9]

    metrics = compute_metrics(y_true, y_pred, y_proba)
    # FPR = FP / (FP + TN) = 1 / (1 + 2) = 1/3 = 0.333333
    assert abs(metrics["false_positive_rate"] - (1.0 / 3.0)) < 1e-4
    # Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3
    assert abs(metrics["precision"] - (2.0 / 3.0)) < 1e-4
    # Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3
    assert abs(metrics["recall"] - (2.0 / 3.0)) < 1e-4


def test_probability_2d_array():
    """Test that compute_metrics correctly handles 2D probability arrays."""
    y_true = [0, 1]
    y_pred = [0, 1]
    y_proba_2d = np.array([[0.8, 0.2], [0.1, 0.9]])

    metrics = compute_metrics(y_true, y_pred, y_proba_2d)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_undefined_roc_auc_single_class():
    """Test that single-class y_true does not crash and handles roc_auc safely."""
    y_true = [1, 1, 1]
    y_pred = [1, 1, 1]
    y_proba = [0.9, 0.8, 0.95]

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["roc_auc"] == 0.0
    assert metrics["accuracy"] == 1.0


def test_evaluate_model_pipeline(tmp_path):
    """Test full evaluate_model function and schema of evaluation.json."""
    # 1. Create synthetic test.csv
    n_samples = 20
    data = {feat: np.random.rand(n_samples) for feat in FEATURES}
    data["url"] = [f"https://test-{i}.com" for i in range(n_samples)]
    data["label"] = [0 if i % 2 == 0 else 1 for i in range(n_samples)]
    df_test = pd.DataFrame(data)

    test_csv = tmp_path / "test.csv"
    df_test.to_csv(test_csv, index=False)

    # 2. Train and save a tiny dummy model
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(df_test[FEATURES], df_test["label"])
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    # 3. Create metadata
    metadata = {
        "model_name": "DummyTestModel",
        "version": "1",
        "threshold": 0.5,
    }
    meta_path = tmp_path / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    out_path = tmp_path / "evaluation.json"

    # 4. Evaluate
    res = evaluate_model(
        model_path=str(model_path),
        metadata_path=str(meta_path),
        test_path=str(test_csv),
        output_path=str(out_path),
    )

    # 5. Verify JSON file and structure
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert saved["model_name"] == "DummyTestModel"
    assert saved["model_version"] == "1"
    assert saved["test_rows"] == n_samples
    assert "metrics" in saved
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]:
        assert m in saved["metrics"]
    assert "confusion_matrix" in saved
    for cm_key in ["tn", "fp", "fn", "tp"]:
        assert cm_key in saved["confusion_matrix"]
    assert "classification_report" in saved
