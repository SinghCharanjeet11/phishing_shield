"""Tests for model saving/loading, versioning and model selection.

Uses only synthetic data and temporary folders: no dataset, no real model,
no shared mlflow.db (CI has none of them).
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.training import model_io
from src.training.model_io import (
    REQUIRED_METADATA_KEYS,
    ModelNotFoundError,
    load_model,
    read_metadata,
    save_model,
)
from src.training.train import build_metadata, select_best, train_and_select

N_FEATURES = 16
COLUMNS = [f"f{i}" for i in range(N_FEATURES)]


# ---------- helpers ----------

def make_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic, roughly separable data with 16 numeric columns."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, N_FEATURES)) + y[:, None] * 1.5
    return pd.DataFrame(X, columns=COLUMNS), pd.Series(y, name="label")


def simple_metrics(y_true, y_pred, y_proba) -> dict:
    """Stand-in for Arman's compute_metrics with the same output keys."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def make_metadata(**overrides) -> dict:
    meta = {
        "model_name": "LogisticRegression",
        "mlflow_run_id": "test-run",
        "mlflow_experiment": "PhishingShield",
        "trained_at": "2026-01-01T00:00:00Z",
        "dataset": "synthetic",
        "train_rows": 200,
        "test_rows": 50,
        "features": COLUMNS,
        "threshold": 0.5,
        "metrics": {"f1": 0.9},
        "candidates": [],
    }
    meta.update(overrides)
    return meta


@pytest.fixture
def tiny_model():
    X, y = make_data()
    return LogisticRegression(max_iter=500).fit(X, y)


# ---------- model_io ----------

def test_save_and_load_roundtrip(tmp_path, tiny_model):
    written = save_model(tiny_model, make_metadata(), tmp_path)

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "model_metadata.json").exists()
    assert written["version"] == "1"

    model, metadata = load_model(tmp_path)
    assert all(k in metadata for k in REQUIRED_METADATA_KEYS)
    assert metadata["features"] == COLUMNS

    X, _ = make_data(n=10, seed=1)
    proba = model.predict_proba(X)
    assert proba.shape == (10, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_version_increments_and_previous_is_archived(tmp_path, tiny_model):
    save_model(tiny_model, make_metadata(model_name="first"), tmp_path)
    second = save_model(tiny_model, make_metadata(model_name="second"), tmp_path)

    assert second["version"] == "2"
    assert read_metadata(tmp_path)["model_name"] == "second"

    archived = tmp_path / "archive" / "v1"
    assert (archived / "model.joblib").exists()
    old_meta = json.loads((archived / "model_metadata.json").read_text(encoding="utf-8"))
    assert old_meta["model_name"] == "first"
    assert old_meta["version"] == "1"


def test_version_in_input_metadata_is_ignored(tmp_path, tiny_model):
    written = save_model(tiny_model, make_metadata(version="99"), tmp_path)
    assert written["version"] == "1"


def test_missing_metadata_keys_raise(tmp_path, tiny_model):
    meta = make_metadata()
    del meta["mlflow_run_id"]
    with pytest.raises(ValueError, match="mlflow_run_id"):
        save_model(tiny_model, meta, tmp_path)
    assert not (tmp_path / "model.joblib").exists()


def test_load_model_missing_raises_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "does-not-exist")


def test_model_dir_env_var_is_used(tmp_path, monkeypatch, tiny_model):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "from-env"))
    save_model(tiny_model, make_metadata())
    assert model_io.get_model_dir() == tmp_path / "from-env"
    assert (tmp_path / "from-env" / "model.joblib").exists()
    load_model()  # no argument: reads MODEL_DIR


# ---------- selection rule ----------

def result(name: str, f1: float, recall: float, fpr: float) -> dict:
    return {"model_name": name, "metrics": {"f1": f1, "recall": recall, "false_positive_rate": fpr}}


def test_select_best_prefers_highest_f1_within_fpr_limit():
    results = [
        result("high_f1_but_noisy", f1=0.99, recall=0.99, fpr=0.30),
        result("good", f1=0.90, recall=0.88, fpr=0.05),
        result("ok", f1=0.85, recall=0.95, fpr=0.04),
    ]
    assert select_best(results, max_fpr=0.10)["model_name"] == "good"


def test_select_best_breaks_f1_ties_on_recall():
    results = [
        result("lower_recall", f1=0.90, recall=0.85, fpr=0.05),
        result("higher_recall", f1=0.90, recall=0.92, fpr=0.05),
    ]
    assert select_best(results)["model_name"] == "higher_recall"


def test_select_best_falls_back_when_no_model_meets_fpr():
    results = [
        result("a", f1=0.80, recall=0.80, fpr=0.20),
        result("b", f1=0.85, recall=0.80, fpr=0.25),
    ]
    assert select_best(results, max_fpr=0.10)["model_name"] == "b"


def test_select_best_rejects_empty():
    with pytest.raises(ValueError):
        select_best([])


# ---------- training with MLflow (temporary tracking store) ----------

def test_train_and_select_logs_runs_and_saves_best(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("PhishingShield-test")

    X_train, y_train = make_data(n=200, seed=0)
    X_test, y_test = make_data(n=80, seed=1)

    best, results = train_and_select(
        X_train, y_train, X_test, y_test, metric_fn=simple_metrics, max_fpr=1.0
    )

    assert {r["model_name"] for r in results} == {"LogisticRegression", "RandomForest"}
    assert best in results
    for r in results:
        run = mlflow.get_run(r["mlflow_run_id"])
        assert "test_f1" in run.data.metrics

    metadata = build_metadata(best, results, COLUMNS, "synthetic", len(X_train), len(X_test))
    written = save_model(best["model"], metadata, tmp_path / "model")
    model, loaded = load_model(tmp_path / "model")

    assert loaded["mlflow_run_id"] == best["mlflow_run_id"]
    assert len(loaded["candidates"]) == 2
    assert written["version"] == "1"
    assert model.predict_proba(X_test).shape == (len(X_test), 2)


# ---------- contract with Arman's feature extractor ----------

def test_metadata_features_match_extractor():
    extractor = pytest.importorskip("src.features.extractor")
    if not hasattr(extractor, "FEATURES"):
        pytest.skip("feature extractor not merged yet")
    best = {"model_name": "x", "mlflow_run_id": "r", "metrics": {}}
    meta = build_metadata(best, [], extractor.FEATURES, "LegitPhish", 1, 1)
    assert meta["features"] == extractor.FEATURES
    assert len(extractor.FEATURES) == 16
