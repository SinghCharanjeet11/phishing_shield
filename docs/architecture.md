# PhishingShield: System Architecture & Data Flow

PhishingShield is an end-to-end MLOps machine learning system engineered to detect phishing URLs using 16 lexical and structural heuristics. This document specifies the complete system architecture, data pipelines, module boundaries, and contributor ownership.

---

## 1. High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Data Engineering & Features (Arman)"]
        RawCSV["data/raw/legitphish.csv\n(101,219 raw rows)"]
        Prepare["src/data/prepare.py\n(Deduplication, Mapping, Split)"]
        Extractor["src/features/extractor.py\n(16 Lexical Features)\n[SINGLE SOURCE OF TRUTH]"]
        TrainCSV["data/processed/train.csv\n(80,697 rows, 80%)"]
        TestCSV["data/processed/test.csv\n(20,175 rows, 20%)"]
        ShiftedCSV["data/processed/current_batch.csv\n(3,000 rows, 80% phish)"]
        
        RawCSV --> Prepare
        Prepare --> Extractor
        Extractor --> TrainCSV
        Extractor --> TestCSV
        TestCSV --> Prepare
        Prepare --> ShiftedCSV
    end

    subgraph MLOps_Pipeline["2. Model Training & Monitoring (Charan)"]
        TrainScript["src/training/train.py\n(Logistic Regression vs Random Forest)"]
        MLflow["MLflow Tracking\n(sqlite:///mlflow.db)"]
        BestModel["artifacts/model/model.joblib\n+ model_metadata.json"]
        Evaluate["src/evaluation/evaluate.py\n(Arman / Shared)\nartifacts/reports/evaluation.json"]
        Evidently["src/monitoring/drift.py\nartifacts/reports/drift_report.html"]

        TrainCSV --> TrainScript
        TestCSV --> TrainScript
        TrainScript --> MLflow
        TrainScript --> BestModel
        BestModel --> Evaluate
        TestCSV --> Evaluate
        TrainCSV --> Evidently
        ShiftedCSV --> Evidently
    end

    subgraph Serving_Deployment["3. Serving & Deployment (Deeksha)"]
        FastAPI["FastAPI Backend\nsrc/api/main.py\n(Port 8000)"]
        Streamlit["Streamlit UI\nfrontend/app.py\n(Port 8501)"]
        Dashboard["frontend/dashboard.py\n(Arman Dashboard Component)"]
        DockerCompose["Docker Compose\n(api + frontend containers)"]
        EC2["AWS EC2 Deployment\n(Ubuntu t3.small)"]

        BestModel --> FastAPI
        Extractor --> FastAPI
        FastAPI -- "HTTP REST" --> Streamlit
        Evaluate --> Dashboard
        FastAPI -- "GET /model-info" --> Dashboard
        Dashboard --> Streamlit
        FastAPI --> DockerCompose
        Streamlit --> DockerCompose
        DockerCompose --> EC2
    end

    subgraph CI_Pipeline["4. Quality Assurance & CI (Charan / Arman / Deeksha)"]
        GitHubActions[".github/workflows/ci.yml"]
        PytestSuite["pytest (tests/test_features.py,\ntest_prepare.py, test_model.py,\ntest_api.py, test_evaluation.py)"]
        
        GitHubActions --> PytestSuite
        PytestSuite --> DockerBuild["Docker Build Verification"]
    end
```

---

## 2. Component Ownership & Team Responsibilities

| Contributor | Role | Core Modules & Ownership Responsibilities |
|---|---|---|
| **Arman (Me)** | **Data & ML Engineer** | `src/features/extractor.py` (single source of truth for 16 URL features), `src/data/prepare.py` (cleaning, deduplication, stratified splits), `src/evaluation/evaluate.py` (`compute_metrics` & evaluation report), `frontend/dashboard.py` (model performance UI), `docs/FEATURE_NOTES.md`, architecture documentation, demo script, and integration testing. |
| **Charan** | **MLOps & CI Engineer** | `src/training/train.py` (Logistic Regression baseline, Random Forest primary, XGBoost optional), `src/training/model_io.py`, MLflow experiment tracking, model selection & versioning, `src/monitoring/drift.py` (Evidently data drift), `.github/workflows/ci.yml` (automated testing CI). |
| **Deeksha** | **Backend & Deployment Engineer** | `src/api/main.py` (FastAPI serving: `/`, `/health`, `/predict`, `/model-info`), `src/api/schemas.py`, `frontend/app.py` (main Streamlit detection UI), `Dockerfile`, `docker-compose.yml`, and AWS EC2 deployment. |

---

## 3. Data Flow & Interface Contracts

### 3.1 Feature Extraction Pipeline
* **Input:** Raw URL string (e.g. `https://example.com/login?token=abc`).
* **Normalization:** URLs without schemes are normalized with `http://`.
* **Execution:** `src/features/extractor.py` extracts exactly 16 numeric features:
  1. `url_length`
  2. `has_ip_address`
  3. `dot_count`
  4. `https_flag`
  5. `url_entropy`
  6. `token_count`
  7. `subdomain_count`
  8. `query_param_count`
  9. `tld_length`
  10. `path_length`
  11. `has_hyphen_in_domain`
  12. `number_of_digits`
  13. `tld_popularity`
  14. `suspicious_file_extension`
  15. `domain_name_length`
  16. `percentage_numeric_chars`
* **Invariant:** Both training (`src/training/train.py`) and inference (`src/api/main.py`) call the same `extract_features()` / `extract_batch()` functions. No feature logic is duplicated in the API or frontend.

### 3.2 Model Training & Selection Contract
* Logistic Regression baseline and Random Forest classifiers are trained on `data/processed/train.csv` (80% split).
* Evaluation on `data/processed/test.csv` (20% split) uses `src.evaluation.evaluate.compute_metrics()`.
* **Selection Rule:** Among candidates with phishing false-positive rate $\le 0.10$, select the candidate with highest **F1-score**; break ties on **Recall**. Accuracy alone is never used as the decision criterion.
* The selected model and metadata are serialized to `artifacts/model/model.joblib` and `artifacts/model/model_metadata.json`.

### 3.3 API & UI Integration Contract
* The Streamlit UI communicates with FastAPI strictly over HTTP:
  * `POST /predict`: Sends `{"url": "..."}`, receives prediction, phishing probability, risk level (`LOW`, `MEDIUM`, `HIGH`), and 16 feature values.
  * `GET /model-info`: Returns model name, version, MLflow run ID, metrics, and candidate models.
* `frontend/dashboard.py` renders inside `frontend/app.py`, reading offline test metrics from `artifacts/reports/evaluation.json` and candidate models from `/model-info`.
* If either the API or the evaluation report is temporarily unavailable, friendly status notices are rendered without crashing the application.
