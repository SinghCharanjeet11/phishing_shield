"""Data drift report (Evidently 0.7.x).

Reference = features of the training split.
Current   = a SEPARATE, deliberately shifted batch built by `python -m src.data.prepare`
            (data/processed/current_batch.csv, made from the test split).

This demonstrates drift monitoring on a simulated batch. It is not production
drift detection, since we have no production traffic.

Run from the repo root:
    python -m src.monitoring.drift
Outputs:
    artifacts/reports/drift_report.html   full Evidently report
    artifacts/reports/drift_summary.json  small summary (drifted features, share, timestamp)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

DEFAULT_REFERENCE_PATH = "data/processed/train.csv"
DEFAULT_CURRENT_PATH = "data/processed/current_batch.csv"
DEFAULT_REPORT_PATH = "artifacts/reports/drift_report.html"
DEFAULT_SUMMARY_PATH = "artifacts/reports/drift_summary.json"
REFERENCE_SAMPLE_SIZE = 5000
DRIFT_SHARE = 0.5  # dataset-level drift if at least this share of features drifted
RANDOM_STATE = 42
NOTE = "Simulated drift: current data is a deliberately shifted batch, not production traffic."


def split_column_types(reference: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Binary/flag features (<= 2 distinct values) are treated as categorical, the rest as numerical."""
    categorical = [c for c in features if reference[c].nunique(dropna=True) <= 2]
    numerical = [c for c in features if c not in categorical]
    return numerical, categorical


def _is_drifted(method: str, threshold: float, score: float) -> bool:
    """p-value tests drift when the score is below the threshold; distance tests when at/above it."""
    if "p_value" in method:
        return score < threshold
    return score >= threshold


def summarize(
    snapshot_dict: dict[str, Any],
    reference_rows: int,
    current_rows: int,
    drift_share: float = DRIFT_SHARE,
) -> dict[str, Any]:
    """Turn Evidently's snapshot dict into the small drift_summary.json."""
    per_feature = []
    drifted_count = None
    for metric in snapshot_dict.get("metrics", []):
        config = metric.get("config", {})
        metric_type = config.get("type", "")
        if metric_type.endswith("DriftedColumnsCount"):
            drifted_count = int(metric["value"]["count"])
        elif metric_type.endswith("ValueDrift"):
            method = str(config.get("method", ""))
            threshold = float(config.get("threshold", 0.05))
            score = float(metric["value"])
            per_feature.append(
                {
                    "feature": config.get("column"),
                    "method": method,
                    "threshold": threshold,
                    "score": score,
                    "drifted": _is_drifted(method, threshold, score),
                }
            )

    n_features = len(per_feature)
    if drifted_count is None:
        drifted_count = sum(f["drifted"] for f in per_feature)
    share = drifted_count / n_features if n_features else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "reference_rows": int(reference_rows),
        "current_rows": int(current_rows),
        "n_features": n_features,
        "drifted_features_count": drifted_count,
        "drifted_share": round(share, 4),
        "dataset_drift": share >= drift_share,
        "drift_share_threshold": drift_share,
        "drifted_features": [f["feature"] for f in per_feature if f["drifted"]],
        "features": per_feature,
        "note": NOTE,
    }


def run_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    drift_share: float = DRIFT_SHARE,
):
    """Run the Evidently drift preset on `features`. Returns (snapshot, summary)."""
    missing = {
        name: [c for c in features if c not in df.columns]
        for name, df in (("reference", reference), ("current", current))
    }
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    ref = reference[features]
    cur = current[features]
    numerical, categorical = split_column_types(ref, features)
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)

    report = Report([DataDriftPreset(drift_share=drift_share)])
    snapshot = report.run(
        Dataset.from_pandas(cur, data_definition=definition),
        Dataset.from_pandas(ref, data_definition=definition),
    )
    summary = summarize(snapshot.dict(), len(ref), len(cur), drift_share)
    return snapshot, summary


def _read_csv(path: str, hint: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise SystemExit(f"'{path}' not found. {hint}")
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Generate the PhishingShield data drift report.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--current", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--reference-sample", type=int, default=REFERENCE_SAMPLE_SIZE)
    args = parser.parse_args(argv)

    if Path(args.reference).resolve() == Path(args.current).resolve():
        raise SystemExit("Reference and current data are the same file; drift must use a separate batch.")

    # Arman's feature list: the single source of truth.
    from src.features.extractor import FEATURES

    reference = _read_csv(args.reference, "Run `python -m src.data.prepare` first.")
    current = _read_csv(args.current, "Run `python -m src.data.prepare` to build the shifted batch.")
    if len(reference) > args.reference_sample:
        reference = reference.sample(n=args.reference_sample, random_state=RANDOM_STATE)

    snapshot, summary = run_drift(reference, current, FEATURES)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(args.report)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Drifted features: {summary['drifted_features_count']}/{summary['n_features']} "
        f"(share {summary['drifted_share']:.2f}, dataset drift: {summary['dataset_drift']})"
    )
    print(f"Report: {args.report}\nSummary: {args.summary}\n{NOTE}")
    return summary


if __name__ == "__main__":
    main()
