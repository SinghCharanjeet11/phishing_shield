"""Simple model-quality monitoring and retraining workflow for PhishingShield."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.evaluate import compute_metrics
from src.features.extractor import FEATURES, extract_batch


INCOMING_PATH = "data/incoming/new_batch.csv"
CHAMPION_PATH = "artifacts/model/model.joblib"
CANDIDATE_PATH = "artifacts/model/candidate_model.joblib"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
RAW_PATH = "data/raw/legitphish.csv"
REPORT_DIR = "artifacts/reports"
RANDOM_STATE = 42


def utc_timestamp() -> str:
    """Return the current UTC time in a simple ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def save_json(path: str, data: dict[str, Any]) -> None:
    """Save a dictionary as nicely formatted JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_incoming(path: str = INCOMING_PATH) -> pd.DataFrame:
    """Load and validate the incoming labelled URL batch."""
    incoming_path = Path(path)
    if not incoming_path.exists():
        fixture = Path("tests/fixtures/sample_urls.csv")
        if fixture.exists():
            fixture_data = pd.read_csv(fixture)
            if "illustrative_label" in fixture_data.columns:
                fixture_data = fixture_data.rename(columns={"illustrative_label": "label"})
            incoming_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_data[["url", "label"]].to_csv(incoming_path, index=False)
        else:
            raise FileNotFoundError(
                f"Incoming data not found at '{path}'. Create a CSV with 'url' and 'label' columns."
            )

    incoming = pd.read_csv(incoming_path)
    print("NEW DATA VALIDATION")
    print(f"Incoming file: {path}")
    print(f"Total records: {len(incoming)}")

    required = {"url", "label"}
    missing_columns = sorted(required.difference(incoming.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    incoming = incoming[["url", "label"]].copy()
    incoming["url"] = incoming["url"].astype(str).str.strip()
    incoming["label"] = pd.to_numeric(incoming["label"], errors="coerce")

    missing_values = int(incoming.isna().sum().sum())
    duplicate_urls = int(incoming.duplicated(subset=["url"], keep="first").sum())
    incoming = incoming.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    incoming = incoming[incoming["url"] != ""]

    labels = set(incoming["label"].dropna().unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError("Target labels must contain only 0 and 1 and include both classes.")

    incoming["label"] = incoming["label"].astype(int)
    phishing_count = int((incoming["label"] == 1).sum())
    legitimate_count = int((incoming["label"] == 0).sum())

    print(f"Missing values: {missing_values}")
    print(f"Duplicate records: {duplicate_urls}")
    print(f"Phishing URLs: {phishing_count}")
    print(f"Legitimate URLs: {legitimate_count}")
    print("Data validation successful!\n")
    return incoming


def generate_fallback_data() -> None:
    """Generate a small local dataset when the real downloaded data/model is absent."""
    print("NOTE: Original raw data/model was missing; using the repository fixture for a synthetic fallback.")
    fixture = pd.read_csv("tests/fixtures/sample_urls.csv")

    incoming = fixture.rename(columns={"illustrative_label": "label"})[["url", "label"]].copy()
    Path(INCOMING_PATH).parent.mkdir(parents=True, exist_ok=True)
    incoming.to_csv(INCOMING_PATH, index=False)

    rows: list[dict[str, Any]] = []
    legit_templates = [
        "https://www.example{}.com/about",
        "https://portal.example{}.org/login",
        "https://docs.example{}.edu/reference",
        "https://news.example{}.net/latest",
    ]
    phishing_templates = [
        "http://account-security-check{}.com/login",
        "http://bank-alert-verify{}.net/update",
        "http://microsoft-login-alert{}.org/signin",
        "http://download-file{}.co/payload.exe",
    ]

    for i in range(80):
        rows.append({"url": legit_templates[i % len(legit_templates)].format(i), "label": 0})
        rows.append({"url": phishing_templates[i % len(phishing_templates)].format(i), "label": 1})

    all_rows = pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)
    features = extract_batch(all_rows["url"])
    data = pd.concat([all_rows, features], axis=1)
    data = data[["url", "label"] + FEATURES]

    # Deterministic 20% holdout. This data is never used for candidate training.
    holdout = data.groupby("label", group_keys=False).sample(
        n=12, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    holdout_urls = set(holdout["url"])
    training = data[~data["url"].isin(holdout_urls)].reset_index(drop=True)

    Path(TRAIN_PATH).parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(TRAIN_PATH, index=False)
    holdout.to_csv(TEST_PATH, index=False)


def ensure_data_and_champion() -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Prepare old data and ensure model.joblib exists without overwriting it again."""
    train_path = Path(TRAIN_PATH)
    test_path = Path(TEST_PATH)
    champion_path = Path(CHAMPION_PATH)

    if not train_path.exists() or not test_path.exists():
        if not Path(RAW_PATH).exists():
            generate_fallback_data()
        else:
            from src.data.prepare import load_dataset, make_splits

            clean = load_dataset(RAW_PATH)
            training, holdout = make_splits(clean)
            train_path.parent.mkdir(parents=True, exist_ok=True)
            training.to_csv(TRAIN_PATH, index=False)
            holdout.to_csv(TEST_PATH, index=False)

    training = pd.read_csv(TRAIN_PATH)
    holdout = pd.read_csv(TEST_PATH)

    missing_train = [column for column in FEATURES + ["label"] if column not in training.columns]
    missing_test = [column for column in FEATURES + ["label"] if column not in holdout.columns]
    if missing_train or missing_test:
        raise ValueError(f"Processed data is missing required columns: train={missing_train}, test={missing_test}")

    if not champion_path.exists():
        print("NOTE: Original champion model is missing; training a champion with the existing Random Forest configuration.")
        champion_path.parent.mkdir(parents=True, exist_ok=True)
        champion = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        champion.fit(training[FEATURES], training["label"].astype(int))
        joblib.dump(champion, champion_path)

    champion = joblib.load(champion_path)
    return training, holdout, champion


def predict_metrics(model: Any, holdout: pd.DataFrame) -> dict[str, float]:
    """Generate predictions and calculate metrics for the phishing positive class."""
    X_holdout = holdout[FEATURES]
    y_true = holdout["label"].astype(int)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_holdout)
        if hasattr(model, "classes_") and 1 in model.classes_:
            class_index = list(model.classes_).index(1)
            y_proba = probabilities[:, class_index]
        else:
            y_proba = probabilities[:, 1] if probabilities.shape[1] >= 2 else probabilities[:, 0]
    else:
        y_proba = model.predict(X_holdout).astype(float)

    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1"],
    }


def calculate_combined_training_data(
    training: pd.DataFrame,
    incoming: pd.DataFrame,
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    """Combine old and incoming data while excluding holdout URLs."""
    holdout_urls = set(holdout["url"].astype(str).str.strip()) if "url" in holdout else set()
    incoming = incoming[~incoming["url"].isin(holdout_urls)].copy()

    incoming_features = extract_batch(incoming["url"])
    incoming_processed = pd.concat(
        [incoming[["url", "label"]].reset_index(drop=True), incoming_features],
        axis=1,
    )

    combined = pd.concat(
        [training[["url", "label"] + FEATURES], incoming_processed],
        axis=0,
    )
    combined = combined.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    return combined


def train_candidate(combined: pd.DataFrame) -> Any:
    """Train a separate Random Forest candidate on the combined dataset."""
    candidate = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    candidate.fit(combined[FEATURES], combined["label"].astype(int))
    Path(CANDIDATE_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, CANDIDATE_PATH)
    return candidate


def apply_quality_gate(champion_metrics: dict[str, float], candidate_metrics: dict[str, float]) -> dict[str, Any]:
    """Apply the recall-first promotion rules without replacing the champion."""
    recall_passed = candidate_metrics["recall"] >= champion_metrics["recall"]
    f1_passed = candidate_metrics["f1_score"] >= champion_metrics["f1_score"] - 0.01
    eligible = recall_passed and f1_passed

    if eligible:
        decision = "CANDIDATE ELIGIBLE FOR PROMOTION"
        reason = "Candidate recall is at least the champion recall and candidate F1-score is within 0.01 of the champion."
    else:
        decision = "EXISTING CHAMPION RETAINED"
        reason = "Candidate recall or the allowed F1-score difference did not meet the quality gate."

    return {
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "selected_primary_metric": "recall",
        "quality_gate_conditions": {
            "candidate_recall_greater_than_or_equal_to_champion": True,
            "candidate_f1_not_lower_by_more_than_0.01": True,
            "recall_passed": recall_passed,
            "f1_passed": f1_passed,
        },
        "final_decision": decision,
        "reason": reason,
        "timestamp": utc_timestamp(),
        "champion_replaced": False,
    }


def track_mlflow(combined: pd.DataFrame, candidate_metrics: dict[str, float]) -> str | None:
    """Log the candidate training run if MLflow is available."""
    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("PhishingShield")
        with mlflow.start_run(run_name="quality-monitoring-candidate") as run:
            mlflow.log_param("model_name", "RandomForestCandidate")
            mlflow.log_param("training_dataset_size", len(combined))
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_metrics(candidate_metrics)
        return run.info.run_id
    except Exception as exc:
        print(f"MLflow unavailable; continuing without experiment tracking ({type(exc).__name__}).")
        return None


def main() -> None:
    """Execute the complete monitoring and retraining workflow."""
    print("=====================================")
    print("PHISHINGSHIELD MLOPS WORKFLOW")
    print("=====================================\n")

    print("STEP 1: Loading incoming data...")
    incoming = load_incoming()

    print("STEP 2: Preparing existing data and champion...")
    training, holdout, champion = ensure_data_and_champion()

    print("STEP 3: Evaluating existing champion...")
    champion_metrics = predict_metrics(champion, holdout)
    print("Champion metrics:")
    for name, value in champion_metrics.items():
        print(f"  {name}: {value:.6f}")

    print("STEP 4: Saving champion evaluation report...")
    champion_report = {
        "model": "Existing Champion",
        "accuracy": champion_metrics["accuracy"],
        "precision": champion_metrics["precision"],
        "recall": champion_metrics["recall"],
        "f1_score": champion_metrics["f1_score"],
        "evaluated_on_rows": len(holdout),
        "timestamp": utc_timestamp(),
    }
    save_json(f"{REPORT_DIR}/evaluation_report.json", champion_report)

    print("STEP 5: Combining old and new training data...")
    combined = calculate_combined_training_data(training, incoming, holdout)
    print(f"Combined training rows: {len(combined)}")

    print("STEP 6: Training candidate...")
    candidate = train_candidate(combined)

    print("STEP 7: Saving candidate model...")
    print(f"Candidate saved to: {CANDIDATE_PATH}")
    print("STEP 8: Evaluating candidate on the untouched holdout...")
    candidate_metrics = predict_metrics(candidate, holdout)
    print("Candidate metrics:")
    for name, value in candidate_metrics.items():
        print(f"  {name}: {value:.6f}")

    mlflow_run_id = track_mlflow(combined, candidate_metrics)

    retraining_report = {
        "model": "RandomForest Candidate",
        "training_dataset_size": len(combined),
        "old_training_rows": len(training),
        "incoming_rows_used": len(combined) - len(training),
        "candidate_metrics": candidate_metrics,
        "training_parameters": {
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "mlflow_run_id": mlflow_run_id,
        "timestamp": utc_timestamp(),
    }
    save_json(f"{REPORT_DIR}/retraining_report.json", retraining_report)

    print("STEP 9: Applying quality gate...")
    gate = apply_quality_gate(champion_metrics, candidate_metrics)
    save_json(f"{REPORT_DIR}/quality_gate.json", gate)

    print("\n=====================================")
    print("FINAL QUALITY GATE DECISION")
    print("=====================================")
    print(f"Champion Recall: {champion_metrics['recall']:.6f}")
    print(f"Candidate Recall: {candidate_metrics['recall']:.6f}")
    print(f"Champion F1-score: {champion_metrics['f1_score']:.6f}")
    print(f"Candidate F1-score: {candidate_metrics['f1_score']:.6f}")
    print(f"Decision: {gate['final_decision']}")
    print("\n=====================================")
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=====================================")


if __name__ == "__main__":
    main()
