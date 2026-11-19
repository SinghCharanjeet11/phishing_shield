"""Unit tests for the Streamlit dashboard component."""

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest

from frontend.dashboard import (
    _fetch_model_info,
    _load_evaluation_report,
    render_dashboard,
)


def test_dashboard_function_signature():
    """Verify render_dashboard public signature matches requirements."""
    sig = inspect.signature(render_dashboard)
    params = list(sig.parameters.keys())
    assert "api_url" in params
    assert "evaluation_path" in params
    assert sig.parameters["evaluation_path"].default == "artifacts/reports/evaluation.json"


def test_load_evaluation_report_missing_file(tmp_path):
    """Test that missing evaluation report returns None safely without raising."""
    missing = tmp_path / "non_existent.json"
    assert _load_evaluation_report(str(missing)) is None


def test_load_evaluation_report_valid_file(tmp_path):
    """Test loading valid evaluation JSON."""
    valid_file = tmp_path / "eval.json"
    payload = {
        "model_name": "RandomForest",
        "metrics": {"accuracy": 0.95, "f1": 0.94},
    }
    valid_file.write_text(json.dumps(payload), encoding="utf-8")

    res = _load_evaluation_report(str(valid_file))
    assert res == payload


def test_fetch_model_info_api_offline():
    """Test that unreachable API returns None safely without raising."""
    res = _fetch_model_info("http://127.0.0.1:9999")
    assert res is None


def test_fetch_model_info_success():
    """Test parsing successful /model-info response."""
    mock_payload = {
        "model_name": "RandomForest",
        "version": "1",
        "candidates": [
            {"model_name": "RandomForest", "metrics": {"f1": 0.95}},
        ],
    }
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_payload
        mock_get.return_value = mock_response

        res = _fetch_model_info("http://127.0.0.1:8000")
        assert res == mock_payload


@patch("streamlit.metric")
@patch("streamlit.table")
@patch("streamlit.dataframe")
@patch("streamlit.subheader")
@patch("streamlit.info")
@patch("streamlit.warning")
def test_render_dashboard_runs_without_raising(
    mock_warn, mock_info, mock_sub, mock_df, mock_tab, mock_met, tmp_path
):
    """Verify render_dashboard executes cleanly with missing files and offline API."""
    render_dashboard(
        api_url="http://127.0.0.1:9999",
        evaluation_path=str(tmp_path / "no_eval.json"),
    )
    # Should display info / warning instead of crashing
    assert mock_info.called or mock_warn.called
