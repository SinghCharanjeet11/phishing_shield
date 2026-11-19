# PhishingShield: Faculty Demonstration Script

This document details the exact 7-step live demonstration sequence for PhishingShield, aligned with Section 14 of `PROJECT_OVERVIEW.md`. It outlines speaker roles, screen views, execution commands, expected outputs, and seamless presenter transitions.

---

## Team Roster & Speaking Roles
* **Arman (Data & ML Engineer):** Project Introduction, Architecture & Pipeline Walkthrough, Step 3 (Evaluation Metrics Comparison on Dashboard), Limitations, Conclusion.
* **Charan (MLOps & CI Engineer):** Step 1 (Model Training), Step 2 (MLflow Experiment Tracking), Step 4 (Model Versioning & Artifacts), Step 6 (GitHub Actions CI), Step 7 (Evidently Drift Detection).
* **Deeksha (Backend, Frontend & Deployment Engineer):** Step 5 (Docker Compose Startup & Live URL Inference).

---

## 7-Step Live Demonstration Sequence

### Step 0: Welcome & Architecture Overview
* **Speaker:** **Arman**
* **Screen:** Presentation / `README.md` and `docs/architecture.md` (Mermaid diagram).
* **Action:** Introduce PhishingShield ("PhiShark - AI Phishing Analyst"), explain the 48-hour MLOps lifecycle objective, dataset characteristics (LegitPhish from Mendeley), and the 16 lexical URL features extracted by the shared single source of truth (`src/features/extractor.py`).
* **Transition to Charan:** *"I'll now hand over to Charan to demonstrate reproducible training across our candidate models in MLflow."*

---

### Step 1: Train Logistic Regression and Random Forest
* **Speaker:** **Charan**
* **Screen:** Terminal / IDE console.
* **Command:**
  ```bash
  python -m src.training.train
  ```
* **Expected Output:**
  * Terminal displays training progress for `LogisticRegression` (`logreg-baseline`) and `RandomForest` (`random-forest`).
  * Autologging metrics logged per run: Accuracy, Precision, Recall, F1, ROC-AUC, FPR.
  * Model selection rule executes: checks FPR $\le 0.10$, selects highest F1 score (Random Forest).
  * Best model saved to `artifacts/model/model.joblib` and `artifacts/model/model_metadata.json`.
* **Transition:** *"Both candidate runs have finished; let's inspect them inside the local MLflow dashboard."*

---

### Step 2: Show Both Runs in MLflow
* **Speaker:** **Charan**
* **Screen:** Web browser at `http://127.0.0.1:5000` (MLflow UI).
* **Command:**
  ```bash
  mlflow ui --host 127.0.0.1 --port 5000
  ```
* **Expected Output:**
  * Experiment `PhishingShield` displays two completed runs.
  * Comparison view demonstrates hyperparameter differences, training curves, and logged test metrics.
  * Tag `selected: true` appears on the winning model run.
* **Transition to Arman:** *"Now Arman will present the formal test split evaluation metrics and the Streamlit performance dashboard."*

---

### Step 3: Compare Evaluation Metrics on the Dashboard
* **Speaker:** **Arman**
* **Screen:** Streamlit Web UI (`http://localhost:8501`) scrolled to the **Model Performance & Evaluation Dashboard** section.
* **Action / Command:**
  ```bash
  python -m src.evaluation.evaluate
  ```
* **Expected Output:**
  * Generated `artifacts/reports/evaluation.json` displays on the dashboard.
  * Metric cards show: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **False Positive Rate** on held-out test data.
  * Confusion Matrix table details: True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP).
  * Candidate model comparison table displays comparative metrics fetched dynamically from `GET /model-info`.
* **Talking Points:** Phishing-positive label is 1; model selection strictly penalized false positives (FPR $\le 0.10$) while optimizing F1 and Recall, preventing aggressive misclassification of legitimate URLs.
* **Transition to Charan:** *"Charan will now highlight our automated model artifact versioning and metadata schemas."*

---

### Step 4: Show Saved, Versioned Model and Metadata
* **Speaker:** **Charan**
* **Screen:** IDE file tree / JSON viewer showing `artifacts/model/`.
* **Action:** Inspect `artifacts/model/model_metadata.json`.
* **Expected Output:**
  * Displays schema: `model_name`, `version`, `mlflow_run_id`, `trained_at`, `features` (16 names), `threshold`, and `candidates`.
  * Demonstrates version archiving: previous models archived under `artifacts/model/archive/v<old>/`.
* **Transition to Deeksha:** *"Deeksha will now run the entire application container stack using Docker Compose."*

---

### Step 5: Run the App with Docker & Live Predictions
* **Speaker:** **Deeksha**
* **Screen:** Terminal & Web Browser (`http://localhost:8501`).
* **Command:**
  ```bash
  docker compose up --build -d
  ```
* **Action:**
  1. Open Streamlit UI at `http://localhost:8501`.
  2. Input a known legitimate URL (e.g. `https://docs.python.org/3/`) -> Returns **LEGITIMATE** with Low Risk.
  3. Input a phishing test URL (e.g. `http://metamsk01lgiix.godaddysites.com/`) -> Returns **PHISHING** with High Risk.
  4. Showcase the 16 extracted lexical feature values displayed in the interactive breakdown table.
* **Transition to Charan:** *"Let's verify our CI automation by inspecting our GitHub Actions test pipeline."*

---

### Step 6: Push a Code Change and Show GitHub Actions Running Tests
* **Speaker:** **Charan**
* **Screen:** GitHub Repository Actions tab (`https://github.com/SinghCharanjeet11/phishing_shield/actions`).
* **Action:** Show automated CI workflow (`.github/workflows/ci.yml`).
* **Expected Output:**
  * Green checkmark on CI workflow run.
  * Test execution step running `pytest tests/ -v`:
    * `test_features.py` (16 feature checks)
    * `test_prepare.py` (data splitting and drift batch checks)
    * `test_evaluation.py` (metric calculations and report schema)
    * `test_model.py` (synthetic training, save/load, metadata)
    * `test_api.py` (FastAPI endpoint contracts)
* **Transition:** *"Charan will now demonstrate data drift monitoring against our simulated distribution-shifted batch."*

---

### Step 7: Generate a Drift Report from a Changed URL-Feature Batch
* **Speaker:** **Charan**
* **Screen:** Browser showing `artifacts/reports/drift_report.html`.
* **Command:**
  ```bash
  python -m src.monitoring.drift
  ```
* **Expected Output:**
  * Evidently HTML report displays comparison between Reference (`data/processed/train.csv`) and Current (`data/processed/current_batch.csv`, built by Arman from held-out test data with 80% phishing and length bias).
  * Report highlights drifted features (e.g., `url_length`, `query_param_count`, `dot_count`) and flags overall dataset drift.
  * Note acknowledged: *"Simulated drift batch for pipeline demonstration; not production traffic."*
* **Transition to Arman:** *"Arman will conclude our demonstration with key architectural takeaways and operational limitations."*

---

### Conclusion & Operational Limitations
* **Speaker:** **Arman**
* **Summary Points:**
  1. **Lexical Heuristics:** PhishingShield uses URL structural signals as a fast, first-line triage tool. It is not an endpoint antivirus or sandboxing engine.
  2. **Evolving Threat Landscape:** Adversaries continuously adapt domain generation algorithms and URL shorteners; periodic retraining and drift tracking are essential.
  3. **Zero Data Leakage:** Rigorous deduplication prior to splitting ensures reported evaluation metrics reflect true generalization performance.
