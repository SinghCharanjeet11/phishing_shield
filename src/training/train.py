"""Train candidate models, track them in MLflow, save the best one.

Run from the repo root (after `python -m src.data.prepare`):
    python -m src.training.train
    python -m src.training.train --xgboost        # also train XGBoost

Selection rule: among candidates whose phishing-class false-positive rate is
<= MAX_FPR, pick the highest F1; ties are broken on recall. Accuracy is never
the deciding metric.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

# Headless backend: MLflow autolog draws plots off the main thread, which
# crashes the default Tk backend on Windows.
matplotlib.use("Agg")

import mlflow  # noqa: E402
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.model_io import get_model_dir, save_model

EXPERIMENT_NAME = "PhishingShield"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_TRAIN_PATH = "data/processed/train.csv"
DEFAULT_TEST_PATH = "data/processed/test.csv"
DEFAULT_DATASET_NAME = "LegitPhish"
LABEL_COLUMN = "label"  # 1 = phishing, 0 = legitimate
RANDOM_STATE = 42
MAX_FPR = 0.10
THRESHOLD = 0.5

MetricFn = Callable[[Any, Any, Any], dict]


def build_candidates(include_xgboost: bool = False) -> dict[str, tuple[str, Pipeline]]:
    """Return {model_name: (mlflow_run_name, pipeline)}."""
    candidates: dict[str, tuple[str, Pipeline]] = {
        "LogisticRegression": (
            "logreg-baseline",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            ),
        ),
        "RandomForest": (
            "random-forest",
            Pipeline(
                [
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
                        ),
                    )
                ]
            ),
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("--xgboost was given but xgboost is not installed") from exc
        candidates["XGBoost"] = (
            "xgboost",
            Pipeline(
                [
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.1,
                            eval_metric="logloss",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    )
                ]
            ),
        )
    return candidates


def select_best(results: list[dict], max_fpr: float = MAX_FPR) -> dict:
    """Pick the best candidate: FPR <= max_fpr, then max F1, tie -> max recall.

    If no candidate meets the FPR limit, fall back to all candidates (and say so),
    so training still produces a model.
    """
    if not results:
        raise ValueError("no candidate results to select from")
    eligible = [r for r in results if r["metrics"]["false_positive_rate"] <= max_fpr]
    if not eligible:
        print(f"WARNING: no model has false_positive_rate <= {max_fpr}; selecting from all.")
        eligible = results
    return max(eligible, key=lambda r: (r["metrics"]["f1"], r["metrics"]["recall"]))


def _numeric_metrics(metrics: dict) -> dict[str, float]:
    """Keep only scalar numeric metrics (MLflow cannot log nested values)."""
    return {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric_fn: MetricFn,
    include_xgboost: bool = False,
    max_fpr: float = MAX_FPR,
    run_params: dict[str, Any] | None = None,
) -> tuple[dict, list[dict]]:
    """Train every candidate in its own MLflow run and return (best, all_results).

    Assumes the MLflow tracking URI and experiment are already set.
    Each result: {"model_name", "mlflow_run_id", "model", "metrics"}.
    """
    mlflow.sklearn.autolog(log_models=True, silent=True)
    results: list[dict] = []

    for model_name, (run_name, pipeline) in build_candidates(include_xgboost).items():
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({"model_name": model_name, "threshold": THRESHOLD, **(run_params or {})})
            pipeline.fit(X_train, y_train)

            proba = pipeline.predict_proba(X_test)[:, 1]
            pred = (proba >= THRESHOLD).astype(int)
            metrics = _numeric_metrics(metric_fn(y_test, pred, proba))
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

            results.append(
                {
                    "model_name": model_name,
                    "mlflow_run_id": run.info.run_id,
                    "model": pipeline,
                    "metrics": metrics,
                }
            )
            print(f"[{model_name}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    mlflow.sklearn.autolog(disable=True)
    return select_best(results, max_fpr=max_fpr), results


def build_metadata(
    best: dict,
    results: list[dict],
    features: list[str],
    dataset: str,
    train_rows: int,
    test_rows: int,
) -> dict[str, Any]:
    """Metadata in the schema agreed in TASKS_CHARAN_MLOPS_CI.md (version is set by save_model)."""
    return {
        "model_name": best["model_name"],
        "version": None,
        "mlflow_run_id": best["mlflow_run_id"],
        "mlflow_experiment": EXPERIMENT_NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dataset": dataset,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "features": list(features),
        "threshold": THRESHOLD,
        "metrics": best["metrics"],
        "candidates": [
            {"model_name": r["model_name"], "mlflow_run_id": r["mlflow_run_id"], "metrics": r["metrics"]}
            for r in results
        ],
    }


def tag_selected_run(metadata: dict[str, Any]) -> None:
    """Mark the selected MLflow run and attach the metadata file to it."""
    client = MlflowClient()
    run_id = metadata["mlflow_run_id"]
    client.set_tag(run_id, "selected", "true")
    client.set_tag(run_id, "model_version", metadata["version"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model_metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        client.log_artifact(run_id, str(path))


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _load_split(path: str, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    if not Path(path).exists():
        raise SystemExit(f"'{path}' not found. Run `python -m src.data.prepare` first.")
    df = pd.read_csv(path)
    missing = [c for c in [*features, LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise SystemExit(f"'{path}' is missing columns {missing}. Re-run `python -m src.data.prepare`.")
    return df[features], df[LABEL_COLUMN].astype(int)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train and select the PhishingShield model.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-dir", default=None, help="defaults to $MODEL_DIR or artifacts/model")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--max-fpr", type=float, default=MAX_FPR)
    parser.add_argument("--xgboost", action="store_true", help="also train XGBoost")
    args = parser.parse_args(argv)

    # Arman's modules: the single source of truth for features and metrics.
    from src.evaluation.evaluate import compute_metrics
    from src.features.extractor import FEATURES

    X_train, y_train = _load_split(args.train_path, FEATURES)
    X_test, y_test = _load_split(args.test_path, FEATURES)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_params = {
        "dataset": args.dataset_name,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": len(FEATURES),
        "random_state": RANDOM_STATE,
        "max_fpr": args.max_fpr,
        "git_commit": _git_commit(),
    }
    best, results = train_and_select(
        X_train, y_train, X_test, y_test,
        metric_fn=compute_metrics,
        include_xgboost=args.xgboost,
        max_fpr=args.max_fpr,
        run_params=run_params,
    )

    metadata = build_metadata(
        best, results, FEATURES, args.dataset_name, len(X_train), len(X_test)
    )
    metadata = save_model(best["model"], metadata, args.model_dir)
    tag_selected_run(metadata)

    print(
        f"\nSelected {metadata['model_name']} (version {metadata['version']}, "
        f"run {metadata['mlflow_run_id']}) -> {get_model_dir(args.model_dir)}"
    )
    return metadata


if __name__ == "__main__":
    main()
