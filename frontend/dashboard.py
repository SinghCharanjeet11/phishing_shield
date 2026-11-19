"""Streamlit Model Performance Dashboard component for PhishingShield.

Exposes render_dashboard() to display evaluation metrics, confusion matrix,
candidate model comparisons, and model metadata in the Streamlit UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


def _load_evaluation_report(path: str) -> dict[str, Any] | None:
    """Safely load the evaluation JSON report without raising exceptions."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _fetch_model_info(api_url: str) -> dict[str, Any] | None:
    """Fetch model metadata from the API /model-info endpoint."""
    clean_url = api_url.rstrip("/") + "/model-info"
    try:
        response = requests.get(clean_url, timeout=3.5)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                return data
    except Exception:
        return None
    return None


def render_dashboard(
    api_url: str,
    evaluation_path: str = "artifacts/reports/evaluation.json",
) -> None:
    """Render the Model Performance Dashboard inside the Streamlit frontend.

    Strict architectural boundary:
        - Reads only evaluation JSON report and HTTP API data.
        - Does NOT load models or run inference locally.
        - Handles missing reports or offline API with friendly warnings, never tracebacks.

    Args:
        api_url: Base URL of the running FastAPI backend (e.g. http://127.0.0.1:8000).
        evaluation_path: Path to evaluation report JSON file.
    """
    st.markdown("---")
    st.header("📊 Model Performance & Evaluation Dashboard")
    st.caption("Standardized offline test evaluation metrics and candidate model tracking.")

    # 1. Load local evaluation metrics report
    eval_data = _load_evaluation_report(evaluation_path)

    # 2. Fetch API model info for candidates and metadata
    api_info = _fetch_model_info(api_url)

    # Determine display metadata from eval report or API info
    model_name = "Not Available"
    model_version = "Not Available"
    run_id = "Not Available"
    trained_at = "Not Available"

    if eval_data:
        model_name = eval_data.get("model_name", model_name)
        model_version = str(eval_data.get("model_version", model_version))
        trained_at = eval_data.get("evaluated_at", trained_at)

    if api_info:
        model_name = api_info.get("model_name", model_name)
        model_version = str(api_info.get("version", model_version))
        run_id = api_info.get("mlflow_run_id", run_id)
        trained_at = api_info.get("trained_at", trained_at)

    # Section 1: Model Overview
    st.subheader("📌 Selected Production Model Overview")
    col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
    with col_ov1:
        st.markdown(f"**Model Name**\n\n`{model_name}`")
    with col_ov2:
        st.markdown(f"**Version**\n\n`v{model_version}`")
    with col_ov3:
        st.markdown(f"**MLflow Run ID**\n\n`{run_id[:12] if run_id != 'Not Available' else run_id}`")
    with col_ov4:
        st.markdown(f"**Evaluated / Trained At**\n\n`{trained_at[:19] if trained_at != 'Not Available' else trained_at}`")

    # Section 2 & 3: Evaluation Metrics and Confusion Matrix
    if eval_data is None:
        st.info("ℹ️ Evaluation results are not available yet. Please run the evaluation pipeline (`python -m src.evaluation.evaluate`).")
    else:
        metrics = eval_data.get("metrics", {})
        st.subheader("🎯 Primary Evaluation Metrics (Held-out Test Split)")

        # Metric Cards
        col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
        with col_m1:
            acc = metrics.get("accuracy")
            st.metric("Accuracy", f"{acc * 100:.2f}%" if acc is not None else "N/A")
        with col_m2:
            prec = metrics.get("precision")
            st.metric("Precision", f"{prec * 100:.2f}%" if prec is not None else "N/A")
        with col_m3:
            rec = metrics.get("recall")
            st.metric("Recall", f"{rec * 100:.2f}%" if rec is not None else "N/A")
        with col_m4:
            f1 = metrics.get("f1")
            st.metric("F1-Score", f"{f1 * 100:.2f}%" if f1 is not None else "N/A")
        with col_m5:
            auc = metrics.get("roc_auc")
            st.metric("ROC-AUC", f"{auc:.4f}" if auc is not None else "N/A")
        with col_m6:
            fpr = metrics.get("false_positive_rate")
            st.metric("False Pos Rate", f"{fpr * 100:.2f}%" if fpr is not None else "N/A")

        # Section 3: Confusion Matrix
        cm = eval_data.get("confusion_matrix", {})
        if cm:
            st.subheader("🔲 Confusion Matrix Breakdown")
            tn = cm.get("tn", 0)
            fp = cm.get("fp", 0)
            fn = cm.get("fn", 0)
            tp = cm.get("tp", 0)

            cm_df = pd.DataFrame(
                [
                    [tn, fp],
                    [fn, tp],
                ],
                index=["Actual Legitimate (0)", "Actual Phishing (1)"],
                columns=["Predicted Legitimate (0)", "Predicted Phishing (1)"],
            )
            st.table(cm_df)

            st.caption(
                f"**True Negatives (TN):** {tn} | **False Positives (FP):** {fp} | "
                f"**False Negatives (FN):** {fn} | **True Positives (TP):** {tp}"
            )

    # Section 4: Candidate Model Comparison
    st.subheader("⚖️ Candidate Model Comparison")
    if api_info is None:
        st.warning("⚠️ Model comparison information is temporarily unavailable (API offline or unreachable).")
    else:
        candidates = api_info.get("candidates", [])
        if not candidates:
            st.info("No candidate models recorded in metadata.")
        else:
            table_rows = []
            selected_name = api_info.get("model_name", "")

            for cand in candidates:
                cand_name = cand.get("model_name", "Unknown")
                cand_metrics = cand.get("metrics", {})
                is_selected = "✅ Yes" if cand_name == selected_name else "No"

                row = {
                    "Model": cand_name,
                    "Selected": is_selected,
                    "Accuracy": f"{cand_metrics.get('accuracy', 0.0):.4f}" if "accuracy" in cand_metrics else "N/A",
                    "Precision": f"{cand_metrics.get('precision', 0.0):.4f}" if "precision" in cand_metrics else "N/A",
                    "Recall": f"{cand_metrics.get('recall', 0.0):.4f}" if "recall" in cand_metrics else "N/A",
                    "F1-Score": f"{cand_metrics.get('f1', 0.0):.4f}" if "f1" in cand_metrics else "N/A",
                    "ROC-AUC": f"{cand_metrics.get('roc_auc', 0.0):.4f}" if "roc_auc" in cand_metrics else "N/A",
                    "FPR": f"{cand_metrics.get('false_positive_rate', 0.0):.4f}" if "false_positive_rate" in cand_metrics else "N/A",
                    "MLflow Run": cand.get("mlflow_run_id", "")[:8],
                }
                table_rows.append(row)

            df_candidates = pd.DataFrame(table_rows)
            st.dataframe(df_candidates, use_container_width=True)

    # Section 5: Metadata Footer
    if api_info or eval_data:
        st.caption("ℹ️ Model selection rule: Phishing FPR ≤ 0.10, maximize F1-score, break ties on Recall.")
