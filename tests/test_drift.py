"""Tests for the drift report, using synthetic feature data only (no dataset in CI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import _is_drifted, run_drift, split_column_types, summarize

NUMERIC = ["url_len", "dots", "entropy"]
FLAGS = ["has_ip", "https"]
FEATURES = NUMERIC + FLAGS


def make_features(n: int, seed: int, shift: float = 0.0, flag_rate: float = 0.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(NUMERIC))), columns=NUMERIC)
    df["url_len"] += shift
    df["dots"] += shift
    for col in FLAGS:
        df[col] = (rng.random(n) < flag_rate).astype(int)
    return df


def test_binary_columns_are_categorical():
    numerical, categorical = split_column_types(make_features(200, 0), FEATURES)
    assert categorical == FLAGS
    assert numerical == NUMERIC


@pytest.mark.parametrize(
    "method,threshold,score,expected",
    [
        ("K-S p_value", 0.05, 0.001, True),
        ("K-S p_value", 0.05, 0.5, False),
        ("Wasserstein distance (normed)", 0.1, 0.3, True),
        ("Wasserstein distance (normed)", 0.1, 0.01, False),
    ],
)
def test_drift_direction_depends_on_method(method, threshold, score, expected):
    assert _is_drifted(method, threshold, score) is expected


def test_shifted_batch_is_detected():
    reference = make_features(600, seed=0)
    current = make_features(300, seed=1, shift=3.0, flag_rate=0.8)
    _, summary = run_drift(reference, current, FEATURES)

    assert summary["n_features"] == len(FEATURES)
    assert summary["dataset_drift"] is True
    assert {"url_len", "dots"} <= set(summary["drifted_features"])
    assert summary["reference_rows"] == 600
    assert summary["current_rows"] == 300
    assert "Simulated" in summary["note"]


def test_same_distribution_has_no_dataset_drift():
    reference = make_features(600, seed=0)
    current = make_features(300, seed=1)
    _, summary = run_drift(reference, current, FEATURES)
    assert summary["dataset_drift"] is False


def test_html_report_can_be_saved(tmp_path):
    snapshot, _ = run_drift(make_features(300, 0), make_features(200, 1, shift=2.0), FEATURES)
    out = tmp_path / "drift_report.html"
    snapshot.save_html(str(out))
    assert out.exists() and out.stat().st_size > 1000


def test_missing_feature_column_raises():
    current = make_features(100, 1).drop(columns=["entropy"])
    with pytest.raises(ValueError, match="entropy"):
        run_drift(make_features(100, 0), current, FEATURES)


def test_summarize_handles_empty_snapshot():
    summary = summarize({"metrics": []}, reference_rows=0, current_rows=0)
    assert summary["n_features"] == 0
    assert summary["dataset_drift"] is False
